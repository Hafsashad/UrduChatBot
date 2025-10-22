import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
import math
import time
from pathlib import Path

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="اردو AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS FOR BEAUTIFUL UI
# ============================================
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    /* Chat container */
    .chat-container {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* User message */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 1rem 0;
        max-width: 70%;
        float: right;
        clear: both;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        font-size: 1.1rem;
        direction: rtl;
    }
    
    /* Bot message */
    .bot-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 1rem 0;
        max-width: 70%;
        float: left;
        clear: both;
        box-shadow: 0 4px 15px rgba(245, 87, 108, 0.4);
        font-size: 1.1rem;
        direction: rtl;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Header styling */
    .header-container {
        text-align: center;
        padding: 2rem;
        background: white;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    }
    
    .header-title {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        font-size: 1.3rem;
        color: #666;
        direction: rtl;
    }
    
    /* Stats card */
    .stats-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        text-align: center;
        margin: 1rem 0;
    }
    
    .stats-number {
        font-size: 2.5rem;
        font-weight: bold;
        color: #667eea;
    }
    
    .stats-label {
        font-size: 1rem;
        color: #666;
        margin-top: 0.5rem;
    }
    
    /* Example button */
    .example-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        border: none;
        cursor: pointer;
        font-size: 1rem;
        margin: 0.5rem;
        transition: transform 0.2s;
        direction: rtl;
    }
    
    .example-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Loading animation */
    .loading-dots {
        display: inline-block;
        animation: dots 1.5s infinite;
    }
    
    @keyframes dots {
        0%, 20% { content: '.'; }
        40% { content: '..'; }
        60%, 100% { content: '...'; }
    }
    
    /* Clear float */
    .clearfix::after {
        content: "";
        display: table;
        clear: both;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# MODEL ARCHITECTURE CLASSES
# ============================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
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
        super().__init__()
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
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
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
        super().__init__()
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
        super().__init__()
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
        super().__init__()
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
        super().__init__()
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

# ============================================
# MODEL LOADING AND INFERENCE
# ============================================

import urllib.request
import os

def download_model_from_release(url, output_path):
    """Download model file from GitHub release"""
    if os.path.exists(output_path):
        print(f"✅ Model already exists at {output_path}")
        return output_path
    
    print(f"📥 Starting download from: {url}")
    
    try:
        # Download with progress
        print("Downloading model file... This may take 3-5 minutes...")
        urllib.request.urlretrieve(url, output_path)
        print(f"✅ Model downloaded successfully to {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ Error downloading model: {str(e)}")
        raise Exception(f"Failed to download model: {str(e)}")

@st.cache_resource
def load_model_and_tokenizer():
    """Load model and tokenizer with caching"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # GitHub release URL for the model
    MODEL_URL = "https://github.com/HafsaShad/urdu-chatbot/releases/download/v1.0.0/best_span_corruption_model.pth"
  
    MODEL_PATH = "best_span_corruption_model.pth"
    
    # Download model if not exists
    if not os.path.exists(MODEL_PATH):
        MODEL_PATH = download_model_from_release(MODEL_URL, MODEL_PATH)
    
    # Load tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load('unigram_urdu_spm.model')
    
    # Load model checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=device)
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
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, tokenizer, device, config

def generate_response(model, tokenizer, input_text, device, max_length=100, temperature=0.8):
    """Generate response for given input"""
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
            
            next_token_logits = output[:, -1, :] / temperature
            probabilities = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probabilities, 1)
        
        tgt = torch.cat([tgt, next_token], dim=1)
        
        if next_token.item() == tokenizer.eos_id():
            break
    
    # Decode response
    response_tokens = tgt[0].tolist()
    response = tokenizer.decode(response_tokens)
    response = response.replace('<s>', '').replace('</s>', '').strip()
    
    return response

# ============================================
# INITIALIZE SESSION STATE
# ============================================

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'message_count' not in st.session_state:
    st.session_state.message_count = 0

if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# ============================================
# MAIN APP
# ============================================

def main():
    # Header
    st.markdown("""
    <div class="header-container">
        <div class="header-title">🤖 اردو AI Chatbot</div>
        <div class="header-subtitle">Urdu Span Corruption Language Model • ٹرانسفارمر ماڈل</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # Temperature slider
        temperature = st.slider(
            "Temperature (Creativity)",
            min_value=0.1,
            max_value=2.0,
            value=0.8,
            step=0.1,
            help="Higher values make output more creative but less focused"
        )
        
        # Max length slider
        max_length = st.slider(
            "Max Response Length",
            min_value=20,
            max_value=150,
            value=100,
            step=10
        )
        
        st.markdown("---")
        
        # Stats
        st.markdown("### 📊 Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number">{st.session_state.message_count}</div>
                <div class="stats-label">Messages</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number">{len(st.session_state.chat_history)}</div>
                <div class="stats-label">Conversations</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.message_count = 0
            st.rerun()
        
        st.markdown("---")
        
        # Model info
        st.markdown("### ℹ️ Model Info")
        st.info("""
        **Architecture:** Transformer Encoder-Decoder
        
        **Training:** Span Corruption
        
        **Language:** Urdu
        
        **Tokenizer:** SentencePiece Unigram
        """)
    
    # Load model
    if not st.session_state.model_loaded:
        with st.spinner('🔄 Loading AI model... Please wait...'):
            try:
                model, tokenizer, device, config = load_model_and_tokenizer()
                st.session_state.model = model
                st.session_state.tokenizer = tokenizer
                st.session_state.device = device
                st.session_state.config = config
                st.session_state.model_loaded = True
                st.success('✅ Model loaded successfully!')
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error loading model: {str(e)}")
                st.stop()
    
    # Example prompts
    st.markdown("### 💡 Try These Examples")
    
    examples = [
        "تم کہاں رہتے ہو",
        "آپ کیسے ہیں؟",
        "آج موسم بہت خوبصورت ہے",
        "کیا حال ہے",
        "میں ٹھیک ہوں"
    ]
    
    cols = st.columns(len(examples))
    for idx, (col, example) in enumerate(zip(cols, examples)):
        with col:
            if st.button(example, key=f"example_{idx}", use_container_width=True):
                # Add user message
                st.session_state.chat_history.append({"role": "user", "content": example})
                st.session_state.message_count += 1
                
                # Generate response
                with st.spinner('🤔 Thinking...'):
                    response = generate_response(
                        st.session_state.model,
                        st.session_state.tokenizer,
                        example,
                        st.session_state.device,
                        max_length=max_length,
                        temperature=temperature
                    )
                
                # Add bot response
                st.session_state.chat_history.append({"role": "bot", "content": response})
                st.session_state.message_count += 1
                st.rerun()
    
    st.markdown("---")
    
    # Chat container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display chat history
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="clearfix">
                    <div class="user-message">
                        👤 {message["content"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="clearfix">
                    <div class="bot-message">
                        🤖 {message["content"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; color: #666;">
            <h2>👋 خوش آمدید! Welcome!</h2>
            <p style="font-size: 1.2rem;">Start chatting in Urdu by typing a message below</p>
            <p style="font-size: 1.1rem; direction: rtl;">نیچے اردو میں پیغام لکھ کر بات چیت شروع کریں</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Input area
    st.markdown("### 💬 Your Message")
    
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input(
            "Type your Urdu message here...",
            key="user_input",
            label_visibility="collapsed",
            placeholder="اردو میں اپنا پیغام یہاں لکھیں..."
        )
    
    with col2:
        send_button = st.button("📤 Send", use_container_width=True, type="primary")
    
    # Handle message sending
    if send_button and user_input.strip():
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.message_count += 1
        
        # Generate response
        with st.spinner('🤔 Generating response...'):
            response = generate_response(
                st.session_state.model,
                st.session_state.tokenizer,
                user_input,
                st.session_state.device,
                max_length=max_length,
                temperature=temperature
            )
        
        # Add bot response
        st.session_state.chat_history.append({"role": "bot", "content": response})
        st.session_state.message_count += 1
        
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>Powered by PyTorch & Streamlit | Urdu Span Corruption Transformer</p>
        <p style="direction: rtl;">پائی ٹارچ اور سٹریم لِٹ کے ذریعے تیار کردہ | اردو ٹرانسفارمر ماڈل</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()


