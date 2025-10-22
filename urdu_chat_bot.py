import streamlit as st
import requests
import os
import torch
import sentencepiece as spm

def download_model_from_release():
    """Download model file from GitHub Releases"""
    model_url = "https://github.com/Hafsashad/UrduChatBot/releases/download/mytag/best_span_corruption_model.pth"
    model_filename = "best_span_corruption_model.pth"
    
    if not os.path.exists(model_filename):
        st.info("📥 Downloading model file from GitHub Releases... This may take a few minutes.")
        try:
            # Download with progress
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            downloaded = 0
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with open(model_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = downloaded / total_size
                            progress_bar.progress(progress)
                            status_text.text(f"Downloaded: {downloaded}/{total_size} bytes ({progress:.1%})")
            
            progress_bar.empty()
            status_text.empty()
            st.success("✅ Model downloaded successfully!")
            return True
            
        except Exception as e:
            st.error(f"❌ Failed to download model: {e}")
            return False
    else:
        st.success("✅ Model file found!")
        return True

@st.cache_resource
def load_model():
    """Load model and tokenizer"""
    # First download model if needed
    if not download_model_from_release():
        return None, None, None, None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Load tokenizer
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.load('unigram_urdu_spm.model')
        
        # Load model checkpoint
        checkpoint = torch.load('best_span_corruption_model.pth', map_location=device)
        config = checkpoint['config']
        
        # Create model (make sure SpanCorruptionTransformer class is defined)
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

def main():
    st.set_page_config(
        page_title="Urdu Conversational Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    # Header
    st.markdown('<h1 style="text-align: center; color: #1f77b4;">🤖 Urdu Conversational Chatbot</h1>', unsafe_allow_html=True)
    st.markdown("### Transformer with Multi-Head Attention")
    
    # Initialize session state
    if 'model_loaded' not in st.session_state:
        with st.spinner('Loading model from GitHub Releases...'):
            st.session_state.model, st.session_state.tokenizer, st.session_state.device, st.session_state.config = load_model()
            st.session_state.model_loaded = True
    
    if st.session_state.model is None:
        st.error("❌ Model could not be loaded. Using demo mode.")
        # Add your demo mode code here
        demo_mode()
    else:
        st.success("✅ Model loaded successfully!")
        # Your main chat interface code here
        chat_interface()

def demo_mode():
    """Demo mode when model fails to load"""
    st.warning("🔧 Running in demo mode with predefined responses")
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Demo responses
    def get_demo_response(user_input):
        responses = {
            "ہیلو": "ہیلو! آپ کیسے ہیں؟",
            "آپ کیسے ہیں": "میں ٹھیک ہوں، شکریہ!",
            "تم کہاں رہتے ہو": "میں کمپیوٹر میں رہتا ہوں!",
            "آج موسم کیسا ہے": "آج موسم بہت خوبصورت ہے!",
        }
        return responses.get(user_input, "میں ابھی سیکھ رہا ہوں۔ براہ کرم دوبارہ کوشش کریں!")
    
    # Chat interface
    user_input = st.text_input("Type your message in Urdu:", placeholder="اردو میں لکھیں...")
    
    if st.button("Send") and user_input:
        response = get_demo_response(user_input)
        st.session_state.chat_history.append((user_input, response))
    
    # Display chat history
    for user_msg, bot_msg in st.session_state.chat_history:
        st.write(f"**You:** {user_msg}")
        st.write(f"**Bot:** {bot_msg}")
        st.write("---")

def chat_interface():
    """Main chat interface when model loads successfully"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Your existing chat interface code
    user_input = st.text_input("Type your message in Urdu:", placeholder="اردو میں لکھیں...")
    
    if st.button("Send") and user_input:
        # Use your model to generate response
        response = generate_response(st.session_state.model, st.session_state.tokenizer, user_input, st.session_state.device)
        st.session_state.chat_history.append((user_input, response))
    
    # Display chat history
    for user_msg, bot_msg in st.session_state.chat_history:
        st.write(f"**You:** {user_msg}")
        st.write(f"**Bot:** {bot_msg}")
        st.write("---")

def generate_response(model, tokenizer, input_text, device):
    """Generate response using the model"""
    try:
        # Your existing generate_response function
        model.eval()
        
        # Tokenize input
        input_tokens = tokenizer.encode(input_text)
        src = torch.tensor([input_tokens]).to(device)
        src_mask = model.make_src_mask(src)
        
        # Start with SOS token
        tgt = torch.tensor([[tokenizer.bos_id()]]).to(device)
        
        for _ in range(100 - 1):
            tgt_mask = model.make_tgt_mask(tgt)
            
            with torch.no_grad():
                encoder_output = model.encoder(src, src_mask)
                decoder_output = model.decoder(tgt, encoder_output, src_mask, tgt_mask)
                output = model.fc_out(decoder_output)
                
                next_token_logits = output[:, -1, :] / 0.8
                probabilities = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probabilities, 1)
            
            tgt = torch.cat([tgt, next_token], dim=1)
            
            if next_token.item() == tokenizer.eos_id():
                break
        
        response_tokens = tgt[0].tolist()
        response = tokenizer.decode(response_tokens)
        response = response.replace('<s>', '').replace('</s>', '').strip()
        
        return response
    
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    main()

