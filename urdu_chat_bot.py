import streamlit as st
import os
import sys

# Simple debug app to test Streamlit deployment
st.set_page_config(page_title="Debug Test", page_icon="🔍")

st.title("🔍 Streamlit Deployment Debug")

st.success("✅ Streamlit is working!")

# Check Python version
st.write("### 🐍 Python Info")
st.write(f"Python Version: {sys.version}")
st.write(f"Python Path: {sys.executable}")

# Check current directory
st.write("### 📁 Current Directory")
st.write(f"Current Dir: {os.getcwd()}")

# List files in current directory
st.write("### 📄 Files in Current Directory")
try:
    files = os.listdir(".")
    for file in files:
        size = os.path.getsize(file) if os.path.isfile(file) else "DIR"
        st.write(f"- {file} ({size} bytes)")
except Exception as e:
    st.error(f"Error listing files: {e}")

# Check if model files exist
st.write("### 🎯 Model Files Check")

required_files = [
    "unigram_urdu_spm.model",
    "unigram_urdu_spm.vocab"
]

for file in required_files:
    if os.path.exists(file):
        size = os.path.getsize(file) / 1024  # KB
        st.success(f"✅ {file} found ({size:.2f} KB)")
    else:
        st.error(f"❌ {file} NOT FOUND")

# Check if torch can be imported
st.write("### 📦 Package Imports")

packages_to_test = [
    "torch",
    "sentencepiece", 
    "numpy"
]

for package in packages_to_test:
    try:
        __import__(package)
        st.success(f"✅ {package} imported successfully")
    except ImportError as e:
        st.error(f"❌ {package} import failed: {e}")

# Test model download
st.write("### 🌐 Model Download Test")

if st.button("Test Model Download"):
    import urllib.request
    
    MODEL_URL = "https://github.com/HafsaShad/urdu-chatbot/releases/download/v1.0/best_span_corruption_model.pth"
    MODEL_PATH = "best_span_corruption_model.pth"
    
    if os.path.exists(MODEL_PATH):
        size = os.path.getsize(MODEL_PATH) / (1024*1024)  # MB
        st.success(f"✅ Model already exists ({size:.2f} MB)")
    else:
        try:
            with st.spinner("Downloading model..."):
                st.info(f"Downloading from: {MODEL_URL}")
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
                size = os.path.getsize(MODEL_PATH) / (1024*1024)
                st.success(f"✅ Model downloaded successfully! ({size:.2f} MB)")
        except Exception as e:
            st.error(f"❌ Download failed: {e}")

# Memory info
st.write("### 💾 System Info")
try:
    import psutil
    mem = psutil.virtual_memory()
    st.write(f"Total Memory: {mem.total / (1024**3):.2f} GB")
    st.write(f"Available Memory: {mem.available / (1024**3):.2f} GB")
    st.write(f"Used Memory: {mem.percent}%")
except:
    st.info("psutil not available for memory info")

st.write("---")
st.success("🎉 If you see this, Streamlit is deployed correctly!")
st.info("Next step: Deploy the full app (urdu_chatbot_app.py)")
