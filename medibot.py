import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
import os
from datetime import datetime
from dotenv import load_dotenv
import base64
import tempfile
from audio_recorder_streamlit import audio_recorder
from PIL import Image
import io

# Import custom modules
from voice_of_the_patient import transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_elevenlabs, text_to_speech_with_gtts
from brain_of_the_doctor import encode_image, analyze_image_with_query

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="Medibot",
    page_icon="🏥",
    layout="wide"
)

load_dotenv()

# ====================== INITIALIZE SESSION STATE ======================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "audio_response" not in st.session_state:
    st.session_state.audio_response = None
if "processing_voice" not in st.session_state:
    st.session_state.processing_voice = False
if "last_audio_bytes" not in st.session_state:
    st.session_state.last_audio_bytes = None
if "last_text_input" not in st.session_state:
    st.session_state.last_text_input = ""
if "image_upload_key" not in st.session_state:
    st.session_state.image_upload_key = 0
if "processed_query" not in st.session_state:
    st.session_state.processed_query = False
if "pending_image" not in st.session_state:
    st.session_state.pending_image = None
if "pending_text" not in st.session_state:
    st.session_state.pending_text = ""
if "menu_open" not in st.session_state:
    st.session_state.menu_open = False
if "record_voice" not in st.session_state:
    st.session_state.record_voice = False
if "upload_image" not in st.session_state:
    st.session_state.upload_image = False

# ====================== CONFIG ======================
DB_FAISS_PATH = "vectorstore/db_faiss"
TOP_K = 5
MAX_OUTPUT_TOKENS = 600
TEMPERATURE = 0.3

# ====================== INIT GROQ CLIENT ======================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

# ====================== LOAD FAISS ======================
@st.cache_resource
def load_vectorstore():
    try:
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    except Exception as e:
        return None

db = load_vectorstore()

# ====================== HELPER FUNCTIONS ======================
def generate_answer(question: str):
    """Generate text answer from RAG pipeline"""
    if not db:
        return "Sorry, the medical database is currently unavailable. Please try again later."
    
    try:
        docs = db.similarity_search(question, k=TOP_K)
        if not docs:
            return "I couldn't find specific information about your query in my medical database. Please consult with a healthcare professional for personalized advice."
        
        context_text = "\n\n".join([doc.page_content[:800] for doc in docs])
        
        system_message = {
            "role": "system",
            "content": (
                "You are Medibot, a professional medical assistant AI. "
                "Provide accurate, helpful medical information based only on the provided medical literature. "
                "Be empathetic, clear, and concise. If you cannot answer from the provided context, say so clearly."
            )
        }
        
        user_message = {
            "role": "user",
            "content": f"Question: {question}\n\nMedical Literature Context:\n{context_text}"
        }
        
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[system_message, user_message],
            temperature=TEMPERATURE,
            max_completion_tokens=MAX_OUTPUT_TOKENS
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return "I'm experiencing technical difficulties. Please try again in a moment."


def process_audio_input(audio_bytes):
    """Process audio input and return transcription"""
    try:
        if len(audio_bytes) < 1000:
            return None
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
            tmp_audio.write(audio_bytes)
            tmp_audio_path = tmp_audio.name
        
        transcription = transcribe_with_groq(GROQ_API_KEY, tmp_audio_path)
        os.unlink(tmp_audio_path)
        
        return transcription
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None


def generate_voice_response(text: str):
    """Generate voice output from text"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
            tmp_audio_path = tmp_audio.name
        
        try:
            text_to_speech_with_elevenlabs(text, tmp_audio_path)
        except:
            text_to_speech_with_gtts(text, tmp_audio_path)
        
        with open(tmp_audio_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
        
        os.unlink(tmp_audio_path)
        
        return audio_bytes
    except Exception as e:
        st.error(f"Error generating voice: {str(e)}")
        return None


def process_image_with_text(image, text_query):
    """Process image with text query"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
            image.save(tmp_img.name, format="JPEG")
            tmp_img_path = tmp_img.name
        
        encoded_image = encode_image(tmp_img_path)
        medical_context = f"{text_query}\n\nProvide medical insights about this image. Be specific and educational."
        analysis = analyze_image_with_query(medical_context, encoded_image)
        
        os.unlink(tmp_img_path)
        
        return analysis
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None


# ====================== CUSTOM CSS ======================
st.markdown("""
<style>
    /* Hide Streamlit stuff */
    .stApp > footer, .stApp > header, .stDeployButton, .stDecoration, #MainMenu {display: none !important;}
    
    .stApp {background: #0f1419 !important;}
    .main .block-container {padding: 0 !important; max-width: 100% !important;}
    
    html, body, .stApp {
    height: 100%;
    margin: 0;
    padding: 0;
    overflow: hidden; /* prevent page scroll */
    }

    .whatsapp-container {
        display: flex;
        flex-direction: column;
        height: 100vh; /* exactly viewport height, not more */
        overflow: hidden;
        margin: 0;
        background: #0f1419;
    }



    /* WhatsApp Layout */
    .whatsapp-container {
        height: 100vh;
        display: flex;
        flex-direction: column;
        max-width: 100%;
        margin: 0;
        background: #0f1419;
        position: relative;
    }
    
    /* Fixed Header */
    .whatsapp-header {
        background: linear-gradient(135deg, #075e54, #128c7e);
        color: white;
        padding: 70px 20px 15px 20px;
        text-align: center;
        border-bottom: 2px solid #128c7e;
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 1000;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-sizing: border-box;
    }
    
    .chatbot-title {
        font-size: 18px;
        font-weight: 500;
        margin: 0;
        display: block;
        text-align: center;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }

    .chatbot-subtitle {
        font-size: 12px;
        margin-top: 4px;
        font-weight: 400;
        opacity: 0.95;
    }
    
    /* Chat Area - Scrollable */
    .chat-area {
    flex: 1;
    background: #0b141a;
    overflow-y: auto;
    padding: 20px;
    margin-top: 160px;     /* header height */
    margin-bottom: 120px;  /* space for input + disclaimer */
    position: relative;
    box-sizing: border-box;
    }

    
    /* Fixed Medical Watermark */
    .chat-area::before {
        content: '';
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 200px;
        height: 200px;
        background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200"><g opacity="0.02"><circle cx="100" cy="100" r="80" fill="none" stroke="white" stroke-width="3"/><path d="M70 100h60M100 70v60" stroke="white" stroke-width="6" stroke-linecap="round"/><ellipse cx="100" cy="65" rx="20" ry="12" fill="white"/><path d="M80 65c0-6 9-12 20-12s20 6 20 12" fill="none" stroke="white" stroke-width="1.5"/></g></svg>');
        background-repeat: no-repeat;
        background-position: center;
        background-size: contain;
        z-index: 1;
        pointer-events: none;
    }
    
    /* Messages */
    .message {
        margin-bottom: 15px;
        display: flex;
        align-items: flex-end;
        position: relative;
        z-index: 2;
    }
    
    .message.sent {
        justify-content: flex-end;
    }
    
    .message-bubble {
        max-width: 65%;
        padding: 10px 14px;
        border-radius: 8px;
        position: relative;
        word-wrap: break-word;
        font-size: 14px;
        line-height: 1.4;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    
    .sent .message-bubble {
        background: #005c4b;
        color: white;
        border-bottom-right-radius: 2px;
    }
    
    .received .message-bubble {
        background: #202c33;
        color: #e9edef;
        border-bottom-left-radius: 2px;
    }
    
    .message-time {
        font-size: 11px;
        opacity: 0.6;
        margin-top: 2px;
        text-align: right;
    }
    
    /* Image in message */
    .message-image {
        max-width: 100%;
        border-radius: 8px;
        margin-bottom: 5px;
    }
    
    /* Welcome */
    .welcome {
        text-align: center;
        color: #8696a0;
        padding: 40px 20px;
        position: relative;
        z-index: 2;
    }
    
    .welcome-icon {
        font-size: 60px;
        margin-bottom: 20px;
        opacity: 0.7;
        animation: wave 2s ease-in-out infinite;
    }
    
    .welcome h2 {
        font-size: 28px;
        margin: 0 0 15px 0;
        color: #e9edef;
        font-weight: 400;
    }
    
    .welcome p {
        font-size: 16px;
        margin: 0;
        opacity: 0.8;
        line-height: 1.5;
    }
    
    @keyframes wave {
        0%, 100% { transform: rotate(0deg); }
        25% { transform: rotate(-10deg); }
        75% { transform: rotate(10deg); }
    }
    
    /* Input Area - Fixed at bottom */
    .input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: #1f2c34;
        padding: 15px 20px;
        z-index: 1001;
        border-top: 1px solid #2a3942;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.3);
    }
    
    .input-wrapper {
        display: flex;
        align-items: center;
        gap: 10px;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .search-input {
        flex: 1;
        background: #2a3942;
        border: none;
        border-radius: 20px;
        padding: 12px 20px;
        color: white;
        font-size: 14px;
        outline: none;
    }
    
    .search-input::placeholder {
        color: #8696a0;
    }
    
    .input-buttons {
        display: flex;
        gap: 8px;
        align-items: center;
    }
    
    .input-button {
        background: #005c4b;
        color: white;
        border: none;
        border-radius: 50%;
        width: 45px;
        height: 45px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        font-size: 18px;
        transition: all 0.2s ease;
    }
    
    .input-button:hover {
        background: #128c7e;
        transform: scale(1.05);
    }
    
    .input-button.secondary {
        background: #2a3942;
    }
    
    .input-button.secondary:hover {
        background: #3d4b52;
    }
    
    /* Menu Options - Hidden by default */
    .menu-options {
        position: fixed;
        bottom: 0;
        right: 20px;
        background: #233138;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.4);
        z-index: 1002;
        display: none;
        flex-direction: column;
        gap: 8px;
        min-width: 120px;
    }
    
    .menu-options.show {
        display: flex;
    }
    
    .menu-option {
        background: #005c4b;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 15px;
        cursor: pointer;
        font-size: 14px;
        text-align: left;
        transition: all 0.2s ease;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .menu-option:hover {
        background: #128c7e;
    }
    
    /* Voice recorder styling */
    .audio-recorder {
        margin: 10px 0;
        background: #2a3942 !important;
        border-radius: 20px !important;
    }
    
    /* Image uploader styling */
    .uploadedImage {
        border-radius: 10px;
        margin: 10px 0;
    }
    
    /* Disclaimer - Fixed at very bottom */
    .disclaimer {
        background: #34495e;
        color: #ecf0f1;
        padding: 8px 20px;
        text-align: center;
        font-size: 11px;
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        z-index: 1002;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.2);
        font-weight: 400;
        height: 30px;
        box-sizing: border-box;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .disclaimer strong {
        font-weight: 600;
        color: #f39c12;
    }

    /* Hide menu items by default using Streamlit classes */
    div[data-testid="stHorizontalBlock"] .stButton:nth-child(3) {
        position: fixed;
        bottom: 120px;
        right: 30px;
        width: auto;
        z-index: 1002;
        background: #233138;
        border-radius: 10px;
        padding: 5px 15px;
    }


    /* Voice recorder custom styling */
    /* Simple voice recorder styling */
    .audio-recorder {
        margin: 20px 0 !important;
        border-radius: 10px !important;
    }

    /* Center the voice recording section */
    section[data-testid="stVerticalBlock"] {
        text-align: center;
}


</style>
""", unsafe_allow_html=True)

# ====================== MAIN CONTAINER ======================
st.markdown('<div class="whatsapp-container">', unsafe_allow_html=True)

# Header
st.markdown('''
<div class="whatsapp-header">
    <h1 class="chatbot-title">🎙️ Medibot - Multimodal AI Medical Assistant</h1>
    <p class="chatbot-subtitle">Voice • Text • Image | Professional Healthcare Information</p>
</div>
''', unsafe_allow_html=True)

# Chat area
st.markdown('<div class="chat-area">', unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown('''
    <div class="welcome">
        <div class="welcome-icon">👋</div>
        <h2>Hello! I'm Medibot</h2>
        <p>Ask me anything using voice, text, or images!</p>
    </div>
    ''', unsafe_allow_html=True)
else:
    for message in st.session_state.messages:
        timestamp = datetime.now().strftime("%H:%M")
        
        if message["role"] == "user":
            content_html = message["content"]
            
            if "image" in message:
                img_b64 = message["image"]
                # Display image first, then text below it
                st.markdown(f'''
                <div class="message sent">
                    <div class="message-bubble">
                        <img src="data:image/jpeg;base64,{img_b64}" class="message-image"/>
                        <div style="margin-top: 10px;">{content_html}</div>
                        <div class="message-time">{timestamp}</div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
            else:
                # Regular text message without image
                st.markdown(f'''
                <div class="message sent">
                    <div class="message-bubble">
                        {content_html}
                        <div class="message-time">{timestamp}</div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
        
        else:  # This is the missing part - assistant messages
            st.markdown(f'''
            <div class="message received">
                <div class="message-bubble">
                    {message["content"]}
                    <div class="message-time">{timestamp}</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
            if "audio" in message and message["audio"]:
                st.audio(message["audio"], format="audio/wav")

st.markdown('</div>', unsafe_allow_html=True)

# ====================== FIXED INPUT AREA ======================
st.markdown('<div class="input-container">', unsafe_allow_html=True)
st.markdown('<div class="input-wrapper">', unsafe_allow_html=True)

# Create columns for input area
col1, col2, col3 = st.columns([20, 1, 1])



with col1:
    # Text input - use dynamic key to reset the input
    text_input_key = st.session_state.get('text_input_key', 'text_input_0')
    text_input = st.text_input(
        "Type your message here...",
        key=text_input_key,
        label_visibility="collapsed",
        placeholder="Type your message here..."
    )

with col2:
    # Send button
    if st.button("➤", key="send_button", help="Send message", use_container_width=True):
        if text_input:
            st.session_state.last_text_input = text_input
            st.session_state.processed_query = False
            st.rerun()

with col3:
    # Menu button
    if st.button("⋮", key="menu_button", help="Voice & Image options", use_container_width=True):
        st.session_state.menu_open = not st.session_state.menu_open
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ====================== FLOATING MENU ======================
if st.session_state.menu_open:
    st.markdown('<div class="menu-options show">', unsafe_allow_html=True)
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        if st.button("🎙️ Voice", key="voice_menu", use_container_width=True):
            st.session_state.menu_open = False
            st.session_state.record_voice = True
            st.rerun()
    
    with col_m2:
        if st.button("📷 Image", key="image_menu", use_container_width=True):
            st.session_state.menu_open = False
            st.session_state.upload_image = True
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# ====================== VOICE RECORDER ======================
if st.session_state.record_voice:
    # Check if there's a pending image to provide context
    if st.session_state.pending_image:
        st.markdown("---")
        st.markdown("### 🎙️ Record Voice for Image")
        st.info("📷 You have an image attached. Describe what you want to know about it.")
    else:
        st.markdown("---")
        st.markdown("### 🎙️ Voice Recording")
    
    # Rest of your voice recorder code remains the same...
    audio_bytes = audio_recorder(
        text="Click the microphone to start recording",
        recording_color="#e74c3c",
        neutral_color="#128c7e",
        icon_name="microphone",
        key="voice_recorder"
    )
    
    st.write("")  # Add some space
    
    if st.button("Cancel Recording"):
        st.session_state.record_voice = False
        st.rerun()
    
    st.markdown("---")

    # Process audio
    if audio_bytes and audio_bytes != st.session_state.get('last_audio_bytes'):
        if len(audio_bytes) > 2000:
            st.session_state.last_audio_bytes = audio_bytes
            
            with st.spinner("Processing your voice..."):
                transcription = process_audio_input(audio_bytes)
                if transcription:
                    st.session_state.pending_text = transcription
                    st.session_state.record_voice = False
                    # Force immediate processing when we have both image and voice
                    if st.session_state.pending_image:
                        st.session_state.processed_query = False
                    st.rerun()
        else:
            st.warning("Please record longer audio (at least 2-3 seconds)")
            st.session_state.record_voice = False
# ====================== IMAGE UPLOADER ======================
if st.session_state.upload_image:
    st.markdown("### 📷 Image Upload")
    uploaded_image = st.file_uploader(
        "Upload Medical Image",
        type=["jpg", "jpeg", "png"],
        key=f"image_upload_{st.session_state.image_upload_key}"
    )
    
    if uploaded_image is not None:
        st.session_state.pending_image = Image.open(uploaded_image)
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        st.session_state.upload_image = False
        st.rerun()
    
    if st.button("Cancel", key="cancel_image"):
        st.session_state.upload_image = False
        st.rerun()

# ====================== PROCESS INPUTS ======================
user_query = None
is_voice_input = False
image_to_process = None

# Handle text input
if text_input and text_input != st.session_state.get('last_text_input', ''):
    user_query = text_input
    st.session_state.last_text_input = text_input
    st.session_state.processed_query = False

# Handle pending voice input with text
if st.session_state.pending_text and text_input:
    user_query = text_input + " " + st.session_state.pending_text
    is_voice_input = True
    st.session_state.pending_text = ""
    st.session_state.last_text_input = text_input
    st.session_state.processed_query = False

# Handle pending voice input alone
if st.session_state.pending_text and not text_input:
    # Create columns to put the cancel button next to the info message
    col1, col2 = st.columns([15, 1])
    with col1:
        st.info(f"🎙️ Voice input detected: \"{st.session_state.pending_text}\"\n\nYou can type more text or just press Enter to send this voice message.")
    with col2:
        if st.button("×", key="cancel_voice_btn", help="Cancel voice input"):
            st.session_state.pending_text = ""
            st.rerun()

# Handle image with voice (when voice is recorded while image is pending) - PUT THIS FIRST
if st.session_state.pending_image and st.session_state.pending_text:
    user_query = st.session_state.pending_text
    image_to_process = st.session_state.pending_image
    is_voice_input = True
    st.session_state.pending_image = None
    st.session_state.pending_text = ""
    st.session_state.last_text_input = ""
    st.session_state.processed_query = False

# Handle image with text
if st.session_state.pending_image and text_input:
    user_query = text_input
    image_to_process = st.session_state.pending_image
    is_voice_input = False
    st.session_state.pending_image = None
    st.session_state.last_text_input = text_input
    st.session_state.processed_query = False

# Display pending image options
if st.session_state.pending_image and not text_input and not st.session_state.pending_text:
    st.info("📷 Image attached. You can type a message or use voice recording to describe the image.")
    
    # Add voice recording option specifically for the image
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎤 Record Voice for Image", use_container_width=True):
            st.session_state.record_voice = True
            st.rerun()
    with col2:
        if st.button("📝 Type Message", use_container_width=True):
            # This will focus on the text input
            st.rerun()


# Generate response
if user_query and not st.session_state.get('processed_query', False):
    st.session_state.processed_query = True
    
    user_message = {"role": "user", "content": user_query}
    
    if image_to_process:
    # Resize image to make it smaller
        max_size = (250, 250)  # Adjust this size as needed
        resized_image = image_to_process.copy()
        
        # Convert RGBA to RGB if needed (for PNG images with transparency)
        if resized_image.mode == 'RGBA':
            resized_image = resized_image.convert('RGB')
        
        resized_image.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        buffered = io.BytesIO()
        resized_image.save(buffered, format="JPEG", optimize=True, quality=85)
        img_b64 = base64.b64encode(buffered.getvalue()).decode()
        user_message["image"] = img_b64
    
    st.session_state.messages.append(user_message)
    
    with st.spinner("🤔 Thinking..."):
        if image_to_process:
            response_text = process_image_with_text(image_to_process, user_query)
        else:
            response_text = generate_answer(user_query)
        
        assistant_message = {"role": "assistant", "content": response_text}
        
        if is_voice_input:
            audio_response = generate_voice_response(response_text)
            if audio_response:
                assistant_message["audio"] = audio_response
        
        st.session_state.messages.append(assistant_message)
    
    # Clear all input states properly
    st.session_state.processing_voice = False
    st.session_state.last_text_input = ""  # Reset the comparison text
    st.session_state.image_upload_key += 1
    st.session_state.pending_image = None
    st.session_state.pending_text = ""
    
    # Use a unique key for the text input to force reset
    st.session_state.text_input_key = f"text_input_{st.session_state.image_upload_key}"
    st.rerun()

# ====================== DISCLAIMER ======================
st.markdown('''
<div class="disclaimer">
    ⚠️ <strong>Medical Disclaimer:</strong> This AI provides general health information for educational purposes only. Always consult qualified healthcare professionals for medical advice, diagnosis, or treatment.
</div>
''', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)