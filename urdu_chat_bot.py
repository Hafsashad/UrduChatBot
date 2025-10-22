import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sentencepiece as spm
import random
import os
from typing import List, Tuple

# Set page config
st.set_page_config(
    page_title="Urdu Span Correction Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# MODEL ARCHITECTURE (Same as your Colab code)
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

@st.cache_resource
def load_model_and_tokenizer():
    """Load the trained model and tokenizer"""
    try:
        # Load tokenizer
        tokenizer = spm.SentencePieceProcessor()
        
        # Try different possible paths for the model files
        model_paths = [
            "unigram_urdu_spm.model",
            "./unigram_urdu_spm.model",
            "models/unigram_urdu_spm.model"
        ]
        
        tokenizer_loaded = False
        for model_path in model_paths:
            if os.path.exists(model_path):
                tokenizer.load(model_path)
                tokenizer_loaded = True
                st.success(f"✅ Tokenizer loaded from {model_path}")
                break
        
        if not tokenizer_loaded:
            st.error("❌ Tokenizer model file not found. Please make sure 'unigram_urdu_spm.model' is in your repository.")
            return None, None

        # Load model checkpoint
        checkpoint_paths = [
            "best_span_corruption_model.pth",
            "./best_span_corruption_model.pth", 
            "models/best_span_corruption_model.pth"
        ]
        
        checkpoint_loaded = False
        checkpoint = None
        for checkpoint_path in checkpoint_paths:
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                checkpoint_loaded = True
                st.success(f"✅ Model checkpoint loaded from {checkpoint_path}")
                break
        
        if not checkpoint_loaded:
            st.error("❌ Model checkpoint file not found. Please make sure 'best_span_corruption_model.pth' is in your repository.")
            return None, None

        config = checkpoint['config']
        
        # Create model
        model = SpanCorruptionTransformer(
            src_vocab_size=tokenizer.get_piece_size(),
            tgt_vocab_size=tokenizer.get_piece_size(),
            d_model=config['d_model'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            d_ff=config['d_ff'],
            dropout=config['dropout'],
            max_len=config['max_len']
        ).to(device)

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        st.success("✅ Model loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None

def generate_response(model, tokenizer, input_text, max_length=100, temperature=0.8):
    """Generate response for given input text"""
    model.eval()
    
    # Tokenize input
    input_tokens = tokenizer.encode(input_text)
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
    
    return response

# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    st.title("🤖 Urdu Span Correction Chatbot")
    st.markdown("""
    This chatbot uses a Transformer model trained with **Span Corruption** technique 
    to generate responses in Urdu. The model was trained on Urdu text data and can 
    engage in conversations, answer questions, and generate contextual responses.
    """)
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.markdown("""
    **Model Details:**
    - Architecture: Transformer Encoder-Decoder
    - Training: Span Corruption (T5-style)
    - Vocabulary: 6,000 tokens
    - Language: Urdu
    
    **How to use:**
    1. Type your message in Urdu
    2. Click 'Generate Response'
    3. The model will generate a response
    
    **Example inputs:**
    - تم کہاں رہتے ہو
    - آپ کیسے ہیں؟
    - آج موسم بہت خوبصورت ہے
    """)
    
    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Load model
    with st.spinner("Loading model and tokenizer..."):
        model, tokenizer = load_model_and_tokenizer()
    
    if model is None or tokenizer is None:
        st.error("""
        Unable to load the model. Please ensure you have:
        - `best_span_corruption_model.pth` (trained model weights)
        - `unigram_urdu_spm.model` (sentencepiece tokenizer)
        
        in your repository root or in a 'models/' folder.
        """)
        return
    
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
            help="Higher values make output more random, lower values more deterministic"
        )
    
    # Generate button
    if st.button("Generate Response", type="primary"):
        if user_input.strip():
            with st.spinner("Generating response..."):
                try:
                    response = generate_response(
                        model, 
                        tokenizer, 
                        user_input, 
                        temperature=temperature
                    )
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "user": user_input,
                        "bot": response
                    })
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
        else:
            st.warning("Please enter a message first.")
    
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
                
                st.markdown("---")
    else:
        st.info("No conversation yet. Start by typing a message above!")
    
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
            if st.button(example, key=f"example_{i}"):
                st.session_state.user_input = example
                st.rerun()
    
    # Model info
    with st.expander("📊 Model Information"):
        st.write(f"**Device:** {device}")
        st.write(f"**Vocabulary Size:** {tokenizer.get_piece_size()}")
        st.write("**Special Tokens:**")
        st.write(f"  - BOS: {tokenizer.bos_id()}")
        st.write(f"  - EOS: {tokenizer.eos_id()}")
        st.write(f"  - PAD: {tokenizer.pad_id()}")

if __name__ == "__main__":
    main()
