import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sentencepiece as spm
import os
import requests
import time
import hashlib

# Set page config
st.set_page_config(
    page_title="Urdu Span Correction Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Force CPU for better compatibility and performance
device = torch.device('cpu')
torch.set_grad_enabled(False)

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x):
        batch_size, num_heads, seq_len, d_k = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def forward(self, query, key, value, mask=None):
        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, V)
        output = self.combine_heads(output)
        output = self.W_o(output)
        return output

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        cross_attn_output = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        ff_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_layers=6, num_heads=8, d_ff=2048, dropout=0.1, max_len=5000):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_layers=6, num_heads=8, d_ff=2048, dropout=0.1, max_len=5000):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x

class SpanCorruptionTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_layers=6,
                 num_heads=8, d_ff=2048, dropout=0.1, max_len=5000):
        super(SpanCorruptionTransformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_len)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_len)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src):
        return (src != 0).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):
        batch_size, tgt_len = tgt.size()
        tgt_padding_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        return tgt_padding_mask & tgt_sub_mask

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        output = self.fc_out(decoder_output)
        return output

# ============================================================================
# MODEL LOADING AND GENERATION
# ============================================================================

@st.cache_resource(show_spinner=False)
def load_tokenizer():
    """Load tokenizer with caching"""
    try:
        tokenizer = spm.SentencePieceProcessor()
        tokenizer_paths = [
            "unigram_urdu_spm.model",
            "./unigram_urdu_spm.model",
            "models/unigram_urdu_spm.model"
        ]
        
        for path in tokenizer_paths:
            if os.path.exists(path):
                tokenizer.load(path)
                st.success(f"✅ Tokenizer loaded from {path}")
                return tokenizer
        
        st.error("❌ Tokenizer file not found. Please ensure 'unigram_urdu_spm.model' is in your repository.")
        return None
    except Exception as e:
        st.error(f"❌ Tokenizer loading error: {str(e)}")
        return None

def verify_file_hash(file_path, expected_hash):
    """Verify file integrity using SHA256 hash"""
    try:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        file_hash = sha256_hash.hexdigest()
        return file_hash == expected_hash
    except Exception as e:
        st.error(f"❌ Hash verification error: {str(e)}")
        return False

def download_model_with_progress(url, file_path, expected_hash):
    """Download model file with progress bar and hash verification"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with open(file_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = downloaded / total_size
                        progress_bar.progress(min(progress, 1.0))
                        status_text.text(f"Downloading: {downloaded/(1024*1024):.1f}MB / {total_size/(1024*1024):.1f}MB")
        
        progress_bar.empty()
        status_text.empty()
        
        # Verify file hash
        st.info("🔍 Verifying file integrity...")
        if verify_file_hash(file_path, expected_hash):
            st.success("✅ File downloaded and verified successfully!")
            return True
        else:
            st.error("❌ File verification failed! The downloaded file may be corrupted.")
            os.remove(file_path)  # Remove corrupted file
            return False
            
    except Exception as e:
        st.error(f"❌ Download error: {str(e)}")
        return False

@st.cache_resource(show_spinner=False)
def load_model(_tokenizer):
    """Load model with caching and automatic download from GitHub release"""
    if _tokenizer is None:
        return None
        
    try:
        # GitHub release URL and expected hash
        GITHUB_RELEASE_URL = "https://github.com/HafsaShad/urdu-chatbot/releases/download/v1.0/best_span_corruption_model.pth"
        EXPECTED_HASH = "f318b68c180ea9eef88bd1d1bc7b10adf06ebec24bf06173214feaafc5ce0f35"
        MODEL_PATH = "best_span_corruption_model.pth"
        
        # Check if model file exists and is valid
        if os.path.exists(MODEL_PATH):
            st.info("🔍 Found existing model file, verifying integrity...")
            if verify_file_hash(MODEL_PATH, EXPECTED_HASH):
                st.success("✅ Model file verified successfully!")
            else:
                st.warning("⚠️ Model file corrupted or modified. Re-downloading...")
                os.remove(MODEL_PATH)
                if not download_model_with_progress(GITHUB_RELEASE_URL, MODEL_PATH, EXPECTED_HASH):
                    return None
        else:
            # Download model
            st.warning("📥 Model file not found. Downloading from GitHub release...")
            if not download_model_with_progress(GITHUB_RELEASE_URL, MODEL_PATH, EXPECTED_HASH):
                return None

        # Load model checkpoint
        with st.spinner("🔄 Loading model weights..."):
            checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
            config = checkpoint['config']

            # Create model
            model = SpanCorruptionTransformer(
                src_vocab_size=_tokenizer.get_piece_size(),
                tgt_vocab_size=_tokenizer.get_piece_size(),
                d_model=config['d_model'],
                num_layers=config['num_layers'],
                num_heads=config['num_heads'],
                d_ff=config['d_ff'],
                dropout=config['dropout'],
                max_len=config['max_len']
            ).to(device)

            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
        st.success("✅ Model loaded successfully!")
        return model
        
    except Exception as e:
        st.error(f"❌ Model loading error: {str(e)}")
        return None

def generate_response(model, tokenizer, input_text, max_length=50, temperature=0.8):
    """Generate response for given input text"""
    if model is None or tokenizer is None:
        return "Model not loaded properly."
    
    try:
        model.eval()
        
        # Tokenize input
        input_tokens = tokenizer.encode(input_text)
        if not input_tokens:
            return "Unable to process input text."
            
        src = torch.tensor([input_tokens]).to(device)
        src_mask = model.make_src_mask(src)
        
        # Start with SOS token
        tgt = torch.tensor([[tokenizer.bos_id()]]).to(device)
        
        for _ in range(max_length - 1):
            tgt_mask = model.make_tgt_mask(tgt)
            
            with torch.no_grad():
                encoder_output = model.encoder(src, src_mask)
                decoder_output = model.decoder(tgt, encoder_output, src_mask, tgt_mask)
                output = model.fc_out(decoder_output)
                
                # Get last token predictions
                next_token_logits = output[:, -1, :] / temperature
                probabilities = F.softmax(next_token_logits, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(probabilities, 1)
            
            # Append to sequence
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # Stop if EOS token generated
            if next_token.item() == tokenizer.eos_id():
                break
        
        # Decode response
        response_tokens = tgt[0].tolist()
        response = tokenizer.decode(response_tokens)
        
        # Clean up response
        response = response.replace('<s>', '').replace('</s>', '').strip()
        
        return response if response else "No response generated."
        
    except Exception as e:
        return f"Generation error: {str(e)}"

# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    st.title("🤖 Urdu Span Correction Chatbot")
    st.markdown("""
    This chatbot uses a Transformer model trained with **Span Corruption** technique 
    to generate responses in Urdu. The model was trained on Urdu text data and can 
    engage in conversations, answer questions, and generate contextual responses.
    
    **Note:** The model weights (~100MB) will be automatically downloaded from GitHub releases on first run.
    """)
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.markdown("""
    **Model Details:**
    - Architecture: Transformer Encoder-Decoder
    - Training: Span Corruption (T5-style)
    - Vocabulary: 6,000 tokens
    - Language: Urdu
    
    **File Verification:**
    - SHA256: `f318b68c180ea9eef88bd1d1bc7b10adf06ebec24bf06173214feaafc5ce0f35`
    - Downloaded files are automatically verified
    
    **How to use:**
    1. Type your message in Urdu
    2. Click 'Generate Response'
    3. The model will generate a response
    """)
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'resources_loaded' not in st.session_state:
        st.session_state.resources_loaded = False
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'tokenizer' not in st.session_state:
        st.session_state.tokenizer = None

    # Load resources only once
    if not st.session_state.resources_loaded:
        with st.status("🚀 Loading chatbot resources...", expanded=True) as status:
            # Load tokenizer
            status.write("📖 Loading tokenizer...")
            tokenizer = load_tokenizer()
            
            if tokenizer is None:
                st.error("Failed to load tokenizer. Please ensure 'unigram_urdu_spm.model' is in your repository.")
                return
            
            # Load model
            status.write("🧠 Loading model (this may take a minute)...")
            model = load_model(tokenizer)
            
            if model is None:
                st.error("Failed to load model. Please check your internet connection and try again.")
                return
            
            # Store in session state
            st.session_state.tokenizer = tokenizer
            st.session_state.model = model
            st.session_state.resources_loaded = True
            
            status.write("✅ All resources loaded successfully!")
            status.update(label="✅ Chatbot ready!", state="complete", expanded=False)
    
    # Chat interface
    st.subheader("💬 Chat with the Urdu Bot")
    
    # Input area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_input = st.text_input(
            "Type your message in Urdu:",
            placeholder="مثال: تم کہاں رہتے ہو",
            key="user_input"
        )
    
    with col2:
        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=1.0,
            value=0.8,
            help="Higher values make output more random"
        )
    
    # Generate button
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        generate_clicked = st.button("Generate Response", type="primary", use_container_width=True)
    
    with col2:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    if generate_clicked and user_input.strip():
        with st.spinner("🤔 Generating response..."):
            start_time = time.time()
            response = generate_response(
                st.session_state.model, 
                st.session_state.tokenizer, 
                user_input, 
                temperature=temperature
            )
            end_time = time.time()
            
            # Add to chat history
            st.session_state.chat_history.append({
                "user": user_input,
                "bot": response,
                "time": f"{end_time - start_time:.2f}s"
            })
    
    # Display chat history
    st.subheader("📝 Conversation History")
    
    if st.session_state.chat_history:
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.container():
                col1, col2 = st.columns([1, 4])
                
                with col1:
                    st.markdown("**👤 You:**")
                with col2:
                    st.info(chat["user"])
                
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.markdown("**🤖 Bot:**")
                with col2:
                    st.success(chat["bot"])
                
                st.caption(f"Generated in {chat['time']}")
                st.markdown("---")
    else:
        st.info("💡 No conversation yet. Start by typing a message above!")
    
    # Quick examples
    st.subheader("💡 Quick Examples")
    example_cols = st.columns(3)
    
    examples = [
        "تم کہاں رہتے ہو",
        "آپ کیسے ہیں؟", 
        "آج موسم بہت خوبصورت ہے",
        "کیا حال ہے",
        "تمہارا نام کیا ہے",
        "میں  سیکھ رہا ہوں"
    ]
    
    for i, example in enumerate(examples):
        with example_cols[i % 3]:
            if st.button(example, key=f"example_{i}", use_container_width=True):
                st.session_state.user_input = example
                st.rerun()
    
    # Model info
    with st.expander("📊 Model Information"):
        if st.session_state.tokenizer:
            st.write(f"**Device:** {device}")
            st.write(f"**Vocabulary Size:** {st.session_state.tokenizer.get_piece_size()}")
            st.write("**Special Tokens:**")
            st.write(f"  - BOS: {st.session_state.tokenizer.bos_id()}")
            st.write(f"  - EOS: {st.session_state.tokenizer.eos_id()}")
            st.write(f"  - PAD: {st.session_state.tokenizer.pad_id()}")
            st.write(f"  - UNK: {st.session_state.tokenizer.unk_id()}")
        
        st.write(f"**Chat History Length:** {len(st.session_state.chat_history)}")
        
        # Show file info
        if os.path.exists("best_span_corruption_model.pth"):
            file_size = os.path.getsize("best_span_corruption_model.pth") / (1024 * 1024)
            st.write(f"**Model File Size:** {file_size:.1f} MB")
            st.write(f"**Model File Verified:** ✅")

if __name__ == "__main__":
    main()
