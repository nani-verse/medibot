import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
import os
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# ---------------- CONFIG ----------------
DB_FAISS_PATH = "vectorstore/db_faiss"
TOP_K = 5
MAX_OUTPUT_TOKENS = 600
TEMPERATURE = 0.3

# ---------------- INIT GROQ CLIENT ----------------
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------------- LOAD FAISS ----------------
@st.cache_resource
def load_vectorstore():
    try:
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    except Exception as e:
        return None

db = load_vectorstore()

# ---------------- LLM Answer ----------------
def generate_answer(question: str):
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

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Medibot",
    page_icon="🏥",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
    
    /* Hide Streamlit stuff */
    .stApp > footer, .stApp > header, .stDeployButton, .stDecoration, #MainMenu {display: none !important;}
}



    .stApp {background: #0f1419 !important;}
    .main .block-container {padding: 0 !important; max-width: 100% !important;}
    
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
        padding: 35px 20px;
        text-align: center;
        border-bottom: 2px solid #128c7e;
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 1000;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        height: 150px;
        box-sizing: border-box;
    }
    
    .chatbot-title {
        font-size: 18px;
        font-weight: 500;
        margin: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .chatbot-subtitle {
        font-size: 14px;
        opacity: 0.95;
        margin-top: 6px;
        font-weight: 400;
        letter-spacing: 0.3px;
    }
    
    /* Chat Area */
    .chat-area {
        flex: 1;
        background: #0b141a;
        overflow-y: auto;
        padding: 20px;
        margin-top: 85px;
        margin-bottom: 70px;
        position: relative;
        min-height: calc(100vh - 155px);
        background-image: 
            radial-gradient(circle at 20% 20%, rgba(18, 140, 126, 0.03) 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, rgba(18, 140, 126, 0.02) 0%, transparent 50%);
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
    
    /* Welcome */
    .welcome {
        text-align: center;
        color: #8696a0;
        margin-top: 120px;
        padding: 60px 20px;
        position: relative;
        z-index: 2;
    }
    
    .welcome-icon {
        font-size: 80px;
        margin-bottom: 30px;
        opacity: 0.7;
        animation: wave 2s ease-in-out infinite;
    }
    
    .welcome h2 {
        font-size: 32px;
        margin: 0 0 20px 0;
        color: #e9edef;
        font-weight: 400;
    }
    
    .welcome p {
        font-size: 18px;
        margin: 0;
        opacity: 0.8;
        line-height: 1.5;
    }
    
    @keyframes wave {
        0%, 100% { transform: rotate(0deg); }
        25% { transform: rotate(-10deg); }
        75% { transform: rotate(10deg); }
    }
    
    /* Input Area - Fixed */


    
    .input-wrapper {
        max-width: 1200px;
        margin: 0 auto;
        display: flex;
        align-items: center;
        gap: 15px;
        height: 100%;
        width: 100%;
    }
    
    /* Style the Streamlit input */
    .stChatInput {
        flex: 1;
        margin: 0 !important;
        width: 100% !important;
    }
    
   
    /* Enhance only the Streamlit chat input */
    .stChatInput > div {
    background: #ffffff !important;
    border: 2px solid #e0e0e0 !important;
    border-radius: 25px !important;
    margin: 0 10px 8px 10px !important;  /* balanced margin */
    box-shadow: 0 2px 8px rgba(0,0,0,0.12);
    height: 48px !important;
    display: flex !important;
    align-items: center !important;
    padding: 0 !important;  /* remove unwanted padding */
    }

.stChatInput > div:focus-within {
    border-color: #128c7e !important;
    box-shadow: 0 2px 15px rgba(18, 140, 126, 0.3);
}

    .stChatInput input {
    flex: 1 !important;
    border: none !important;
    background: transparent !important;
    padding: 0 15px 0 20px !important;  /* left padding is here */
    font-size: 15px !important;
    color: #2c3e50 !important;
}



    .stChatInput input::placeholder {
    color: #7f8c8d !important;
    padding-left: 5px !important;  
    }

    .stChatInput button {
    background: linear-gradient(135deg, #128c7e, #075e54) !important;
    border: none !important;
    border-radius: 50% !important;
    width: 40px !important;
    height: 40px !important;
    margin: 0 0 0 0 !important;  /* aligned, no big gap */
    color: #fff !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.2);
    }

.stChatInput button:hover {
    transform: scale(1.05) !important;
    box-shadow: 0 5px 15px rgba(18, 140, 126, 0.4) !important;
}

    
    .stChatInput button svg {
        width: 18px !important;
        height: 18px !important;
        margin: 0 !important;
    }
    
    /* Fixed Disclaimer */
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
        z-index: 1001;
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Main container
st.markdown('<div class="whatsapp-container">', unsafe_allow_html=True)

# Header - Fixed at top
st.markdown('''
<div class="whatsapp-header">
    <h1 class="chatbot-title"> Medibot - AI Medical Assistant</h1>
    <p class="chatbot-subtitle">Professional Healthcare Information & Support</p>
</div>
''', unsafe_allow_html=True)

# Chat area
st.markdown('<div class="chat-area">', unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown('''
    <div class="welcome">
        <div class="welcome-icon">👋</div>
        <h2>Hello! I'm Medibot</h2>
        <p>Ask me anything about health and medicine</p>
    </div>
    ''', unsafe_allow_html=True)
else:
    for message in st.session_state.messages:
        timestamp = datetime.now().strftime("%H:%M")
        
        if message["role"] == "user":
            st.markdown(f'''
            <div class="message sent">
                <div class="message-bubble">
                    {message["content"]}
                    <div class="message-time">{timestamp}</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div class="message received">
                <div class="message-bubble">
                    {message["content"]}
                    <div class="message-time">{timestamp}</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Input area - Fixed above disclaimer
st.markdown('<div class="input-area">', unsafe_allow_html=True)
st.markdown('<div class="input-wrapper">', unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask me anything about health and medicine..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner(""):
        response = generate_answer(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Fixed Disclaimer at bottom
st.markdown('''
<div class="disclaimer">
    ⚠️ <strong>Medical Disclaimer:</strong> This AI provides general health information for educational purposes only. Always consult qualified healthcare professionals for medical advice, diagnosis, or treatment.
</div>
''', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)