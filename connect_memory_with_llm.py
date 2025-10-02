# medical_rag_bot_groq.py
# Terminal-based Medical RAG Bot using FAISS + Groq
# Updated version: safer cleaning, similarity scores, better error handling

import os
import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq

# ---------------- CONFIG ----------------
DB_FAISS_PATH = "vectorstore/db_faiss"
TOP_K = 5
MAX_OUTPUT_TOKENS = 800
TEMPERATURE = 0.3

# ---------------- INIT GROQ CLIENT ----------------
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
print("✅ Groq client initialized successfully")

# ---------------- STEP 1: Load FAISS DB ----------------
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if not os.path.exists(DB_FAISS_PATH):
    raise ValueError("❌ FAISS DB not found! Please run create_memory_for_llm.py first.")

db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
print("✅ FAISS DB loaded successfully")

# ---------------- Helper: build context prompt ----------------
def build_context_prompt(question, docs):
    contexts = []
    for i, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page", "N/A")
        book = doc.metadata.get("book_title", "Unknown Book")
        text = doc.page_content.strip()[:800]
        contexts.append(f"Source {i} ({book}, p. {page}): {text}")
    context_text = "\n\n".join(contexts)

    prompt = f"""Question: {question}

Use ONLY the information in the Documents section to answer the question. Keep your answer concise and in plain language.

Documents:
{context_text}
"""
    return prompt

# ---------------- Helper: clean model answer ----------------
def clean_answer(answer: str) -> str:
    # Remove only page references like (p. 123)
    answer = re.sub(r'\(.*?\bp\.?\s*\d+.*?\)', '', answer, flags=re.IGNORECASE)
    # Remove "Source:" lines
    answer = re.sub(r'(?mi)^\s*source[s]?:.*$', '', answer)
    # Remove inline "Source: ..." text
    answer = re.sub(r'(?i)\s*source[s]?:.*', '', answer)
    # Collapse extra whitespace
    answer = re.sub(r'\n{2,}', '\n', answer).strip()
    answer = re.sub(r'[ \t]{2,}', ' ', answer).strip()
    return answer

# ---------------- STEP 2: Generate answer ----------------
def generate_answer(prompt: str) -> str:
    system_message = {
        "role": "system",
        "content": (
            "You are a concise, factual medical assistant. "
            "Answer the user's question using ONLY the provided documents. "
            "Do NOT include book titles, page numbers, source citations, or any 'Source' text inside the answer body. "
            "Provide a clear single best answer in plain language (2–6 sentences). "
            "If the documents do not contain enough information to be certain, say you are unsure."
        )
    }
    user_message = {"role": "user", "content": prompt}

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[system_message, user_message],
            temperature=TEMPERATURE,
            max_completion_tokens=MAX_OUTPUT_TOKENS
        )
        raw = response.choices[0].message.content
        cleaned = clean_answer(raw)
        return cleaned if cleaned else raw.strip()
    except Exception as e:
        return f"❌ Error generating response: {e}"

# ---------------- STEP 3: Chat loop ----------------
print("🩺 Medical RAG Summarization Bot — type 'exit' to quit")

while True:
    question = input("\n💬 Your question: ").strip()
    if not question:
        continue
    if question.lower() in ["exit", "quit"]:
        print("👋 Exiting bot. Stay healthy!")
        break

    # Search with scores
    docs_and_scores = db.similarity_search_with_score(question, k=TOP_K)
    if not docs_and_scores:
        print("❌ No relevant documents found.")
        continue

    docs = [doc for doc, _ in docs_and_scores]
    prompt = build_context_prompt(question, docs)

    answer = generate_answer(prompt)
    print("\n💡 Answer:\n", answer)

    # Show retrieved sources with similarity scores
    print(f"\n📚 Retrieved {len(docs_and_scores)} sources:")
    for i, (doc, score) in enumerate(docs_and_scores, start=1):
        book = doc.metadata.get("book_title", "Unknown Book")
        page = doc.metadata.get("page", "N/A")
        snippet = doc.page_content.replace("\n", " ")[:200] + "..."
        print(f"  [{i}] {book} (p. {page}) | score={score:.3f} | {snippet}")
