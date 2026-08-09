import os
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

current_db_path = "./chroma_db_default"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_retriever():
    if not os.path.exists(current_db_path):
        return None
    db = Chroma(persist_directory=current_db_path, embedding_function=embeddings)
    return db.as_retriever(search_type="mmr", search_kwargs={'k': 10, 'fetch_k': 30})

def clear_database_safe():
    global current_db_path
    new_id = uuid.uuid4().hex[:8]
    current_db_path = f"./chroma_db_{new_id}"
    if not os.path.exists("data"):
        os.makedirs("data")

def add_pdf_to_knowledge_base(file_path: str):
    """
    Safely adds a file to the database.
    Returns (success_bool, message)
    """
    try:
        # 1. Double check extension
        if not file_path.lower().endswith(".pdf"):
            return False, "Not a PDF file."

        # 2. Try to load the PDF
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        if not docs:
            return False, "PDF is empty or unreadable."

        filename = os.path.basename(file_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        split_docs = text_splitter.split_documents(docs)

        for doc in split_docs:
            doc.metadata["source"] = filename

        db = Chroma(persist_directory=current_db_path, embedding_function=embeddings)
        db.add_documents(split_docs)

        return True, f"Added {len(split_docs)} chunks."

    except Exception as e:
        # Catch image errors, decryption errors, etc.
        print(f"Error processing {file_path}: {e}")
        return False, f"Error: {str(e)}"