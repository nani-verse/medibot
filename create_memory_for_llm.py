import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from tqdm import tqdm   # ✅ for progress bar

# ---------------- CONFIG ----------------
DATA_PATH = "data/"                     # Folder containing PDFs
DB_FAISS_PATH = "vectorstore/db_faiss"  # FAISS DB path

# ---------------- Load PDFs ----------------
def load_new_pdf_files(folder_path=DATA_PATH, existing_books=set()):
    loader = DirectoryLoader(folder_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    new_documents = []
    for doc in documents:
        source_file = os.path.basename(doc.metadata.get("source", "unknown"))
        book_title = os.path.splitext(source_file)[0]

        # Skip if already in existing FAISS DB
        if book_title in existing_books:
            continue

        doc.metadata["book_title"] = book_title
        if "page" not in doc.metadata:
            doc.metadata["page"] = "N/A"
        new_documents.append(doc)

    return new_documents

# ---------------- Split into chunks ----------------
def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        add_start_index=True
    )

    chunks = []
    print(f"🔹 Splitting {len(documents)} documents into chunks...")
    for i, doc in enumerate(tqdm(documents, desc="Chunking progress")):
        chunks.extend(text_splitter.split_documents([doc]))
    print(f"✅ Total chunks created: {len(chunks)}")
    return chunks

# ---------------- Load or create FAISS DB ----------------
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if os.path.exists(DB_FAISS_PATH):
    print("📂 Existing FAISS DB found, loading...")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

    # Get existing book titles from the DB
    existing_books = set(doc.metadata.get("book_title") for doc_id, doc in db.docstore._dict.items())
    print(f"📚 Already in DB: {existing_books}")

    # Load only new books
    new_docs = load_new_pdf_files(DATA_PATH, existing_books)
    if new_docs:
        print(f"📄 Loaded {len(new_docs)} new PDF pages")

        # Split into chunks
        new_chunks = create_chunks(new_docs)

        # Embedding with progress bar
        print("⚙️ Generating embeddings for new chunks...")
        db_new = FAISS.from_documents(tqdm(new_chunks, desc="Embedding progress"), embedding_model)

        db.merge_from(db_new)
        print(f"✅ Merged {len(new_chunks)} new chunks into FAISS DB")
    else:
        print("✅ No new books to add")
else:
    print("🆕 No FAISS DB found, creating new DB from all PDFs...")
    all_docs = load_new_pdf_files(DATA_PATH)
    chunks = create_chunks(all_docs)

    # Embedding with progress bar
    print("⚙️ Generating embeddings for all chunks...")
    db = FAISS.from_documents(tqdm(chunks, desc="Embedding progress"), embedding_model)
    print(f"✅ Created new FAISS DB with {len(chunks)} chunks")

# ---------------- Save DB ----------------
os.makedirs(DB_FAISS_PATH, exist_ok=True)
db.save_local(DB_FAISS_PATH)
print(f"✅ FAISS DB saved at {DB_FAISS_PATH}")
