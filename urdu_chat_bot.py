# urdu_chatbot_app.py
import streamlit as st
import torch
import torch.nn.functional as F
import sentencepiece as spm
import math
import torch.nn as nn
import os

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
# STREAMLIT APP
# ============================================================================

@st.cache_resource
def load_model():
    """Load the trained model and tokenizer"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load('unigram_urdu_spm.model')
    
    # Load model checkpoint
    try:
        checkpoint = torch.load('best_span_corruption_model.pth', map_location=device)
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
        
        return model, tokenizer, device, config
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

def generate_response(model, tokenizer, input_text, device, max_length=100, temperature=0.8):
    """Generate response for given input text"""
    if model is None:
        return "Model not loaded properly."
    
    model.eval()
    
    try:
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
    
    except Exception as e:
        return f"Error generating response: {str(e)}"

def main():
    st.set_page_config(
        page_title="Urdu Conversational Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .urdu-text {
            font-size: 1.2rem;
            direction: rtl;
            text-align: right;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .user-message {
            background-color: #d1ecf1;
            padding: 10px;
            border-radius: 10px;
            margin: 5px 0;
            border-left: 5px solid #17a2b8;
        }
        .bot-message {
            background-color: #e2e3e5;
            padding: 10px;
            border-radius: 10px;
            margin: 5px 0;
            border-left: 5px solid #6c757d;
        }
        .stButton button {
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Urdu Conversational Chatbot</h1>', unsafe_allow_html=True)
    st.markdown("### Transformer with Multi-Head Attention | Span Corruption Training")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.info(
            "This chatbot uses a custom Transformer model trained with "
            "span corruption for Urdu language conversations."
        )
        
        st.header("⚙️ Model Info")
        
        # Load model info
        if 'model_info' not in st.session_state:
            with st.spinner('Loading model info...'):
                model, tokenizer, device, config = load_model()
                if config:
                    st.session_state.model_info = {
                        'layers': config['num_layers'],
                        'heads': config['num_heads'], 
                        'd_model': config['d_model'],
                        'vocab_size': tokenizer.get_piece_size() if tokenizer else 0
                    }
        
        if 'model_info' in st.session_state:
            st.write(f"• **Layers**: {st.session_state.model_info['layers']}")
            st.write(f"• **Heads**: {st.session_state.model_info['heads']}")
            st.write(f"• **Model Dim**: {st.session_state.model_info['d_model']}")
            st.write(f"• **Vocab Size**: {st.session_state.model_info['vocab_size']}")
        
        st.header("💡 Example Prompts")
        example_prompts = [
            "ہیلو، آپ کیسے ہیں؟",
            "آج موسم کیسا ہے؟",
            "تم کہاں رہتے ہو؟",
            "کیا تم سکول جاتے ہو؟",
            "میرا نام علی ہے"
        ]
        
        for prompt in example_prompts:
            if st.button(prompt, key=prompt):
                st.session_state.user_input = prompt
                st.rerun()
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'model_loaded' not in st.session_state:
        with st.spinner('Loading AI model... This may take a moment.'):
            st.session_state.model, st.session_state.tokenizer, st.session_state.device, _ = load_model()
            st.session_state.model_loaded = True
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Chat")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
                st.markdown(f'<div class="user-message"><strong>👤 You:</strong> {user_msg}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="bot-message"><strong>🤖 Bot:</strong> {bot_msg}</div>', unsafe_allow_html=True)
        
        # User input
        user_input = st.text_input(
            "Type your message in Urdu:",
            key="user_input",
            placeholder="اردو میں اپنا پیغام لکھیں... (Type in Urdu)"
        )
        
        # Buttons
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🚀 Send", type="primary", use_container_width=True) and user_input:
                if st.session_state.model_loaded and st.session_state.model:
                    with st.spinner("Generating response..."):
                        response = generate_response(
                            st.session_state.model,
                            st.session_state.tokenizer,
                            user_input,
                            st.session_state.device
                        )
                        st.session_state.chat_history.append((user_input, response))
                        st.rerun()
                else:
                    st.error("Model not loaded properly. Please refresh the page.")
        
        with col_btn2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
    
    with col2:
        st.subheader("📊 Statistics")
        
        st.write(f"**Messages in chat**: {len(st.session_state.chat_history)}")
        
        if st.session_state.chat_history:
            st.write("**Recent conversation**:")
            for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history[-3:]):
                st.text(f"You: {user_msg[:20]}..." if len(user_msg) > 20 else f"You: {user_msg}")
                st.text(f"Bot: {bot_msg[:20]}..." if len(bot_msg) > 20 else f"Bot: {bot_msg}")
                st.write("---")
        
        # File check
        st.write("**File Status**:")
        required_files = {
            'unigram_urdu_spm.model': 'Tokenizer',
            'unigram_urdu_spm.vocab': 'Vocabulary', 
            'best_span_corruption_model.pth': 'Model Weights'
        }
        
        for file, desc in required_files.items():
            if os.path.exists(file):
                st.success(f"✅ {desc}")
            else:
                st.error(f"❌ {desc}")

if __name__ == "__main__":
    main()