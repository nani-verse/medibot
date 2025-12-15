🩺 Medibot – Multimodal Medical RAG Assistant

Medibot is a multimodal medical question-answering system built using Retrieval-Augmented Generation (RAG).
It supports text, voice, and image inputs and generates grounded medical responses by retrieving information from verified medical documents.


🚀 Key Features

📄 Document-grounded answers using FAISS-based semantic search
🎙️ Voice input support via Speech-to-Text (Groq Whisper)
🖼️ Medical image analysis using multimodal LLMs
🔊 Optional voice output using Text-to-Speech
⚡ Low-latency inference powered by Groq LLMs
🧠 RAG architecture to minimize hallucinations

🏗️ Architecture Overview

1. User provides Text / Voice / Image input
2. Voice is converted to text (STT)
3. If image is present → Multimodal analysis
4. Text query is embedded and searched in FAISS
5. Retrieved context + query sent to LLM
6. Model generates a grounded medical response
7. (Optional) Response converted to speech (TTS)


🧩 Tech Stack

Python - Core backend language
Streamlit	- Interactive web interface
LangChain	- RAG orchestration framework
FAISS	- Vector database for medical knowledge retrieval
HuggingFace	- Semantic text embeddings
Groq LLMs	- High-performance language models
Groq Whisper	- Speech-to-Text (STT)
ElevenLabs / gTTS -	Text-to-Speech (TTS)

⚙️ Setup Instructions

1️⃣ Clone Repository

-> git clone 
-> cd medibot

2️⃣ Create Virtual Environment

-> python -m venv venv
-> source venv/bin/activate   # Linux / Mac
-> venv\Scripts\activate      # Windows

3️⃣ Install Dependencies
-> pip install -r requirements.txt

🔑 Environment Variables

Set the following API keys:(Use .env file if preferred)

1. export GROQ_API_KEY=your_groq_api_key
2. export ELEVENLABS_API_KEY=your_elevenlabs_api_key

🌐 Run Web Application
streamlit run medibot.py
