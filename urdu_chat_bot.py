import streamlit as st
import requests
import os
import torch
import torch.nn.functional as F
import sentencepiece as spm
import math
import torch.nn as nn
import time

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
# DOWNLOAD AND LOAD MODEL FROM GITHUB RELEASES
# ============================================================================

def download_file_from_github(url, filename):
    """Download any file from GitHub with progress tracking"""
    if os.path.exists(filename):
        file_size = os.path.getsize(filename) / (1024 * 1024)
        return True
    
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        if total_size == 0:
            return False
        
        downloaded = 0
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
        
        return os.path.exists(filename)
            
    except Exception as e:
        return False

def download_all_files():
    """Download all required files from GitHub"""
    files_to_download = {
        'best_span_corruption_model.pth': 'https://github.com/Hafsashad/UrduChatBot/releases/download/v1.0.0/best_span_corruption_model.pth',
        'unigram_urdu_spm.model': 'https://github.com/Hafsashad/UrduChatBot/raw/main/unigram_urdu_spm.model',
        'unigram_urdu_spm.vocab': 'https://github.com/Hafsashad/UrduChatBot/raw/main/unigram_urdu_spm.vocab'
    }
    
    all_downloaded = True
    for filename, url in files_to_download.items():
        if not download_file_from_github(url, filename):
            all_downloaded = False
    
    return all_downloaded

@st.cache_resource
def load_model():
    """Load model and tokenizer with comprehensive error handling"""
    
    # Download all required files first
    if not download_all_files():
        return None, None, None, None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Load tokenizer
        if not os.path.exists('unigram_urdu_spm.model'):
            return None, None, None, None
        
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.load('unigram_urdu_spm.model')
        vocab_size = tokenizer.get_piece_size()
        
        # Load model checkpoint
        if not os.path.exists('best_span_corruption_model.pth'):
            return None, None, None, None
        
        checkpoint = torch.load('best_span_corruption_model.pth', map_location=device)
        
        # Check checkpoint structure
        if 'config' not in checkpoint:
            return None, None, None, None
        
        config = checkpoint['config']
        
        # Create model
        model = SpanCorruptionTransformer(
            src_vocab_size=vocab_size,
            tgt_vocab_size=vocab_size,
            d_model=config['d_model'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            d_ff=config['d_ff'],
            dropout=config['dropout'],
            max_len=config['max_len']
        ).to(device)
        
        # Load weights
        if 'model_state_dict' not in checkpoint:
            return None, None, None, None
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, tokenizer, device, config
        
    except Exception as e:
        return None, None, None, None

# ============================================================================
# RESPONSE GENERATION
# ============================================================================

def generate_response(model, tokenizer, input_text, device, max_length=100, temperature=0.8):
    """Generate response using the loaded model"""
    try:
        model.eval()
        
        # Tokenize input
        input_tokens = tokenizer.encode(input_text)
        if not input_tokens:
            return "معاف کیجئے، میں اس ان پٹ کو سمجھ نہیں سکا۔"
        
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
        
        # Clean up response
        response = response.replace('<s>', '').replace('</s>', '').strip()
        
        return response if response else "معاف کیجئے، میں جواب نہیں بنا سکا۔"
    
    except Exception as e:
        return f"جواب بنانے میں مسئلہ: {str(e)}"

def get_demo_response(user_input):
    """Demo responses when model is not available"""
    responses = {
        "ہیلو": "ہیلو! آپ کیسے ہیں؟ میں آپ کی کس طرح مدد کر سکتا ہوں؟",
        "آپ کیسے ہیں": "میں ٹھیک ہوں، شکریہ! آپ سنائیں؟",
        "تم کہاں رہتے ہو": "میں ایک AI چیٹ بوٹ ہوں اور ڈیجیٹل دنیا میں رہتا ہوں!",
        "آج موسم کیسا ہے": "میں باہر نہیں دیکھ سکتا، لیکن امید ہے کہ موسم اچھا ہے!",
        "کیا تم سکول جاتے ہو": "نہیں، میں ایک AI ماڈل ہوں جو Urdu میں بات چیت کرتا ہے!",
        "میرا نام علی ہے": "آپ سے مل کر بہت خوشی ہوئی علی!",
        "شکریہ": "خوش آمدید! کوئی اور سوال؟",
        "خدا حافظ": "خدا حافظ! بات کر کے اچھا لگا!",
        "کیا حال ہے": "سب ٹھیک ہے! آپ کے دن کیسے گزر رہے ہیں؟",
        "تمہارا نام کیا ہے": "میں Urdu Transformer Chatbot ہوں!",
        "کیا کر رہے ہو": "آپ سے بات کر رہا ہوں! آپ کیسے ہیں؟",
        "hi": "Hello! I'm an Urdu chatbot. You can talk to me in Urdu!",
        "hello": "Hello! Nice to meet you!",
        "how are you": "I'm doing great! How about you?",
        "what is your name": "I'm Urdu Transformer Chatbot!",
        "thank you": "You're welcome! 😊"
    }
    
    user_input_lower = user_input.lower()
    
    # Find best match
    for key, response in responses.items():
        if key.lower() in user_input_lower:
            return response
    
    # Default responses
    default_responses = [
        "یہ دلچسپ ہے! میں اس کے بارے میں مزید جاننا چاہوں گا۔",
        "معاف کیجئے، میں ابھی اس سوال کا جواب نہیں دے سکتا۔",
        "کیا آپ اسے مختلف طریقے سے پوچھ سکتے ہیں؟",
        "میں Urdu میں بات چیت کے لیے تربیت یافتہ ہوں!",
        "آپ کا سوال بہت اچھا ہے! میں اس پر کام کر رہا ہوں۔",
        "I'm still learning! Try asking me something in Urdu.",
        "That's an interesting question! I'm optimized for Urdu conversations."
    ]
    
    import random
    return random.choice(default_responses)

# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="Urdu Conversational Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    # Custom CSS for beautiful styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.8rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1rem;
            font-weight: bold;
        }
        .sub-header {
            font-size: 1.4rem;
            color: #666;
            text-align: center;
            margin-bottom: 2rem;
        }
        .user-message {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 16px;
            border-radius: 18px 18px 5px 18px;
            margin: 8px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            max-width: 80%;
            margin-left: auto;
        }
        .bot-message {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 12px 16px;
            border-radius: 18px 18px 18px 5px;
            margin: 8px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            max-width: 80%;
        }
        .status-box {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            border-left: 5px solid #28a745;
            margin: 10px 0;
        }
        .demo-box {
            background: #fff3cd;
            padding: 15px;
            border-radius: 10px;
            border-left: 5px solid #ffc107;
            margin: 10px 0;
        }
        .stButton button {
            width: 100%;
            border-radius: 8px;
            font-weight: bold;
            height: 3em;
        }
        .chat-container {
            height: 500px;
            overflow-y: auto;
            padding: 10px;
            background: #fafafa;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Urdu Conversational Chatbot</h1>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Transformer with Multi-Head Attention | Span Corruption Training</div>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'model_loaded' not in st.session_state:
        with st.spinner('🚀 Loading AI Model... This may take a minute for first time.'):
            st.session_state.model, st.session_state.tokenizer, st.session_state.device, st.session_state.config = load_model()
            st.session_state.model_loaded = True
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ℹ️ About")
        
        if st.session_state.model is not None:
            st.markdown('<div class="status-box">', unsafe_allow_html=True)
            st.success("**✅ AI Model Active**")
            st.write(f"• **Layers**: {st.session_state.config['num_layers']}")
            st.write(f"• **Heads**: {st.session_state.config['num_heads']}")
            st.write(f"• **Model Dim**: {st.session_state.config['d_model']}")
            st.write(f"• **Vocab Size**: {st.session_state.tokenizer.get_piece_size()}")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="demo-box">', unsafe_allow_html=True)
            st.warning("**🔧 Demo Mode Active**")
            st.write("Using predefined responses")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### 💡 Example Prompts")
        example_prompts = [
            "ہیلو، آپ کیسے ہیں؟",
            "آج موسم کیسا ہے؟",
            "تم کہاں رہتے ہو؟",
            "کیا تم سکول جاتے ہو؟",
            "میرا نام علی ہے",
            "خدا حافظ"
        ]
        
        for prompt in example_prompts:
            if st.button(prompt, use_container_width=True):
                st.session_state.user_input = prompt
                st.rerun()
        
        st.markdown("---")
        st.markdown("### 📊 Chat Info")
        st.write(f"**Messages**: {len(st.session_state.chat_history)}")
        
        if st.button("🗑️ Clear Chat History", use_container_width=True, type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

    # Main Chat Interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 💬 Conversation")
        
        # Display chat history
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for user_msg, bot_msg in st.session_state.chat_history:
            st.markdown(f'<div class="user-message"><strong>👤 You:</strong> {user_msg}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="bot-message"><strong>🤖 Bot:</strong> {bot_msg}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # User input
        col_input1, col_input2 = st.columns([4, 1])
        with col_input1:
            user_input = st.text_input(
                "**Type your message:**",
                key="user_input",
                placeholder="Type in Urdu or English...",
                label_visibility="collapsed"
            )
        with col_input2:
            send_button = st.button("🚀 Send", use_container_width=True)
        
        if (send_button or user_input) and user_input.strip():
            if st.session_state.model is not None:
                with st.spinner("🤖 Generating response..."):
                    response = generate_response(
                        st.session_state.model,
                        st.session_state.tokenizer,
                        user_input,
                        st.session_state.device
                    )
            else:
                response = get_demo_response(user_input)
            
            st.session_state.chat_history.append((user_input, response))
            st.rerun()
    
    with col2:
        st.markdown("### 📈 Statistics")
        
        if st.session_state.model is not None:
            st.success("**Model Status**: ✅ Loaded")
        else:
            st.warning("**Model Status**: 🔧 Demo Mode")
        
        st.info(f"**Total Messages**: {len(st.session_state.chat_history)}")
        
        if st.session_state.chat_history:
            st.markdown("**Recent Messages**:")
            for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history[-3:]):
                st.text(f"You: {user_msg[:20]}..." if len(user_msg) > 20 else f"You: {user_msg}")
                st.text(f"Bot: {bot_msg[:20]}..." if len(bot_msg) > 20 else f"Bot: {bot_msg}")
                if i < len(st.session_state.chat_history[-3:]) - 1:
                    st.write("---")

if __name__ == "__main__":
    main()
