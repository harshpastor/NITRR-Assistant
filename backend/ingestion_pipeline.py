import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
# CHANGED: Import Google Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# Ensure you have GOOGLE_API_KEY in your .env file
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

def load_documents(docs_path="docs"):
    """Load all text files from the docs directory"""
    print(f"Loading documents from {docs_path}...")
    
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist.")
    
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader
    )
    
    documents = loader.load()
    
    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}.")
    
    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    """Split documents into smaller chunks"""
    print("Splitting documents into chunks...")
    
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print("Creating embeddings and storing in ChromaDB...")
        
    # CHANGED: Use Google Embeddings
    # 'models/text-embedding-004' is a common Gemini embedding model
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory, 
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("--- Finished creating vector store ---")
    return vectorstore

def main():
    print("=== RAG Document Ingestion Pipeline (Gemini) ===\n")
    
    docs_path = "docs"
    persistent_directory = "db/chroma_db"
    
    # Check if vector store exists
    if os.path.exists(persistent_directory):
        print(f"⚠️  Directory {persistent_directory} exists.")
        print("Note: If you are switching models, you MUST delete this folder first to re-embed.")
    
    print("Initializing vector store...\n")
    
    documents = load_documents(docs_path)  
    chunks = split_documents(documents)
    vectorstore = create_vector_store(chunks, persistent_directory)
    
    print("\n✅ Ingestion complete! Documents ready for RAG.")

if __name__ == "__main__":
    main()