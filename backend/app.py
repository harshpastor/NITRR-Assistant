#!/usr/bin/env python3
import os
import time
import uuid
import glob
import sqlite3
import fitz
 
from typing import List, Dict, Any
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from flask import send_from_directory
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import numpy as np

# Vector store
import chromadb
from chromadb.config import Settings

# PDF parsing
# from pypdf import PdfReader
import pymupdf4llm

# Google Generative AI SDK
import google.generativeai as genai

load_dotenv()

# ---- Configuration ----
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GEMINI_CHAT_MODEL = os.environ.get("GEMINI_CHAT_MODEL", "models/gemini-2.5-flash-lite")
GEMINI_EMBED_MODEL = os.environ.get("GEMINI_EMBED_MODEL", "models/text-embedding-004")

CHROMA_DIR = os.environ.get("CHROMA_DIR", os.path.abspath("./chroma"))
DOCS_DIR = os.environ.get("DOCS_DIR", os.path.abspath("./docs")) # New: Local docs folder
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*")

# ---- Flask setup ----
app = Flask(__name__)
CORS(app, resources={r"*": {"origins": ALLOWED_ORIGINS}})

# ---- Google GenAI client setup ----
if not GOOGLE_API_KEY:
    print("[WARN] GOOGLE_API_KEY is not set.")
else:
    genai.configure(api_key=GOOGLE_API_KEY)

# ---- Chroma client ----
# This is persistent. Data is saved to ./chroma folder.
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))

def collection_name_for_program(program: str) -> str:
    return f"ordinance_{program}".lower()

def get_collection(program: str):
    name = collection_name_for_program(program)
    try:
        col = chroma_client.get_collection(name=name)
    except Exception:
        col = chroma_client.create_collection(name=name, metadata={"hnsw:space": "cosine"})
    return col

# ------------ Re-ranking -------------
print("Loading Cross-Encoder model... (this may take a moment)")
# This model is optimized for re-ranking (scoring query-doc pairs)
CROSS_ENCODER = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Global cache to hold BM25 indices per program so we don't rebuild every time
# Structure: { "btech": { "model": BM25Okapi, "texts": [str], "metadatas": [dict] } }
BM25_CACHE = {}
def invalidate_bm25_cache(program):
    """Call this after ingestion to force a rebuild of the keyword index."""
    if program in BM25_CACHE:
        del BM25_CACHE[program]
def get_bm25_data(program):
    """
    Fetches all documents from Chroma for a specific program and builds/returns
    the BM25 index. Uses caching to be fast.
    """
    if program in BM25_CACHE:
        return BM25_CACHE[program]
    
    print(f"Building BM25 index for {program}...")
    col = get_collection(program)
    
    # Fetch all documents to build the keyword index
    # Note: In production with millions of docs, you'd use a separate search engine like Elastic.
    # For a college project (thousands of chunks), in-memory is fine.
    all_docs = col.get() # Fetches IDs, metadatas, documents
    
    texts = all_docs['documents']
    metadatas = all_docs['metadatas']
    
    if not texts:
        return None

    # Tokenize corpus for BM25
    tokenized_corpus = [doc.lower().split() for doc in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Cache it
    data = {
        "model": bm25,
        "texts": texts,
        "metadatas": metadatas
    }
    BM25_CACHE[program] = data
    return data

# ------------ Utilities -------------
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports.db")
def init_db():
    """Initialize the SQLite database for reports."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS missing_reports (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            program TEXT,
            failed_query TEXT,
            suggested_document TEXT,
            chat_context TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Initialize DB on startup
init_db()
def embed_texts(texts: List[str], task_type="retrieval_document") -> List[List[float]]:
    if not texts:
        return []
    embeddings = []
    # Batching could be added here for large lists
    for text in texts:
        try:
            result = genai.embed_content(
                model=GEMINI_EMBED_MODEL,
                content=text,
                task_type=task_type
            )
            embeddings.append(result['embedding'])
        except Exception as e:
            print(f"Error embedding text chunk: {e}")
            # Append a zero vector or skip (simple skip here to avoid crash)
            pass 
    return embeddings

def simple_chunks(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    text = text.replace("\r\n", "\n")
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 2 <= max_chars:
            buf = (buf + "\n\n" + p).strip() if buf else p
        else:
            if buf:
                chunks.append(buf)
            if len(p) > max_chars:
                start = 0
                while start < len(p):
                    end = min(start + max_chars, len(p))
                    chunks.append(p[start:end])
                    start = end - overlap if end < len(p) else end
            else:
                buf = p
    if buf:
        chunks.append(buf)

    with_overlap = []
    for i, c in enumerate(chunks):
        if i == 0 or overlap <= 0:
            with_overlap.append(c)
        else:
            prev = chunks[i-1]
            tail = prev[-overlap:]
            with_overlap.append((tail + "\n" + c).strip())
    return with_overlap

def file_already_ingested(col, filename: str) -> bool:
    """Check if a file with this title already exists in metadata."""
    if not filename: return False
    # Chroma 'get' can filter by metadata
    existing = col.get(where={"title": filename}, include=["metadatas"])
    return len(existing['ids']) > 0

def process_and_ingest(text, title, program, effective_from=None, source_url=None, filename_on_disk=None):
    # ... (date/url defaults logic remains same) ...
    if not effective_from: effective_from = datetime.now().strftime('%Y-%m-%d')
    if not source_url: source_url = "empty"
    
    chunks = simple_chunks(text, max_chars=1900, overlap=220)
    embeddings = embed_texts(chunks, task_type="retrieval_document")
    col = get_collection(program)
    ids = [str(uuid.uuid4()) for _ in chunks]
    
    # Check if title exists to avoid duplicates (optional, based on your preference)
    # col.delete(where={"title": title}) 

    metadatas = [{
        "title": title,
        "filename": filename_on_disk or "unknown.pdf", # <--- NEW FIELD
        "section": "",
        "program": program,
        "effective_from": effective_from,
        "source_url": source_url,
    } for _ in chunks]

    col.upsert(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
    return len(chunks)

def extract_markdown_per_page(path):
    """
    Extracts Markdown text from a PDF, keeping track of page numbers.
    Returns a list of dicts: [{'text': '...', 'page': 1}, ...]
    """
    data = []
    doc = fitz.open(path)
    
    # pymupdf4llm currently converts the whole doc. 
    # We can simulate page-by-page by passing specific page indices if supported,
    # or just use standard pymupdf text extraction for page awareness if markdown is tricky per page.
    # Fortunately, pymupdf4llm.to_markdown accepts `pages`.
    
    for i in range(len(doc)):
        try:
            # Convert single page to markdown
            text = pymupdf4llm.to_markdown(path, pages=[i])
            if text.strip():
                data.append({"text": text, "page": i + 1}) # 1-based index
        except Exception as e:
            print(f"Error reading page {i}: {e}")
            
    return data

# 2. UPDATE: Chunker to respect page boundaries
def process_and_ingest_pages(pages_data, title, program, effective_from=None, source_url=None, filename_on_disk=None):
    if not effective_from: effective_from = datetime.now().strftime('%Y-%m-%d')
    if not source_url: source_url = "empty"
    
    all_chunks = []
    all_embeddings = []
    all_metadatas = []
    all_ids = []
    
    for entry in pages_data:
        text = entry['text']
        page_num = entry['page']
        
        # Split this page's text into chunks
        page_chunks = simple_chunks(text, max_chars=1000, overlap=100) # Smaller chunks for better page precision
        
        if not page_chunks: continue
        
        # Embed chunks
        embeddings = embed_texts(page_chunks, task_type="retrieval_document")
        
        for i, chunk in enumerate(page_chunks):
            all_chunks.append(chunk)
            all_embeddings.append(embeddings[i])
            all_ids.append(str(uuid.uuid4()))
            all_metadatas.append({
                "title": title,
                "filename": filename_on_disk or "unknown.pdf",
                "page": page_num, # <--- NEW: STORE PAGE NUMBER
                "section": "",
                "program": program,
                "effective_from": effective_from,
                "source_url": source_url,
            })

    if not all_chunks:
        return 0

    col = get_collection(program)
    # Delete old version to avoid duplicates
    col.delete(where={"title": title}) 
    
    # Batch upsert (Chroma handles large batches well, but safe to do all at once for moderate docs)
    col.upsert(ids=all_ids, documents=all_chunks, embeddings=all_embeddings, metadatas=all_metadatas)
    return len(all_chunks)


def gemini_chat_answer(question: str, context_chunks: List[Dict[str, Any]], history: List[Dict[str, str]]) -> str:
    # Improved System Prompt
   
    system_prompt = (
        "You are an intelligent Academic Ordinance Assistant.\n"
        "Your goal is to answer student questions based ONLY on the provided context.\n\n"
        "RULES:\n"
        "1. If the user's question is ambiguous, ask a polite clarifying question.\n"
        "2. If the user answers a previous clarification, combine it with chat history.\n"
        "3. Always cite the source title if you find the answer.\n"
        "4. If the answer is NOT in the context, output EXACTLY this phrase: 'NO_DATA: I could not find that information.'\n" 
        "5. Be concise and clear with your answers.\n"
        "6. Make the answer more structured and pointwise wherever possible\n"
    )
    
    # ... (rest of the function remains exactly the same)
    
    # Format Context
    lines = []
    for i, c in enumerate(context_chunks, 1):
        meta = c.get("metadata", {})
        title = meta.get("title") or "Doc"
        lines.append(f"[{i}] Source: {title}\nContent: {c['text']}")
    ctx = "\n\n".join(lines)

    # Format Chat History (Limit to last 3 turns to save tokens)
    chat_history_text = ""
    if history:
        # Take last 3 messages
        recent_history = history[-3:] 
        for msg in recent_history:
            role = "Student" if msg.get("role") == "user" else "Assistant"
            chat_history_text += f"{role}: {msg.get('text')}\n"

    # Create the final prompt
    model = genai.GenerativeModel(
        model_name=GEMINI_CHAT_MODEL,
        system_instruction=system_prompt
    )
    
    user_prompt = (
        f"CONTEXT DOCUMENTS:\n{ctx}\n\n"
        f"CHAT HISTORY:\n{chat_history_text}\n"
        f"CURRENT QUESTION: {question}\n\n"
        "ANSWER:"
    )
    
    try:
        resp = model.generate_content(user_prompt)
        return resp.text.strip()
    except Exception as e:
        return f"Error from Gemini: {str(e)}"

def guess_missing_document(query: str, history: List[Dict[str, str]]) -> str:
    """
    Asks Gemini to analyze the failed query and history to suggest a missing document title.
    """
    system_prompt = (
        "You are an expert Librarian. The user asked a question that could not be answered "
        "by the current database. Your job is to analyze the query and chat history to "
        "predict EXACTLY what kind of specific document is missing.\n"
        "Return ONLY a short, specific document title (e.g., 'B.Tech Fee Structure 2025', "
        "'Hostel Rules & Regulations', 'Academic Calendar'). Do not add any conversational text."
    )
    
    # Format history for context
    chat_context = ""
    for msg in history[-3:]: # Look at last 3 messages
        role = "Student" if msg.get("role") == "user" else "AI"
        chat_context += f"{role}: {msg.get('text')}\n"
    
    user_prompt = (
        f"CHAT HISTORY:\n{chat_context}\n"
        f"FAILED QUERY: {query}\n\n"
        "SUGGESTED DOCUMENT TITLE:"
    )
    
    model = genai.GenerativeModel(
        model_name=GEMINI_CHAT_MODEL,
        system_instruction=system_prompt
    )
    
    try:
        resp = model.generate_content(user_prompt)
        return resp.text.strip()
    except Exception:
        return "Unknown Document"
    

# ------------- Endpoints ----------------
@app.route("/documents/<path:filename>", methods=["GET"])
def get_document(filename):
    """Serve the PDF file to the user."""
    return send_from_directory(DOCS_DIR, filename)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "model": GEMINI_CHAT_MODEL, "db_path": CHROMA_DIR})

@app.route("/db_status", methods=["GET"])
def db_status():
    """Returns a summary of what is currently stored in the DB."""
    stats = {}
    try:
        cols = chroma_client.list_collections()
        for c in cols:
            count = c.count()
            # Get a few unique titles to show what's there
            # (Fetching all metadata might be slow if DB is huge, limiting to 10 items to sample)
            sample = c.get(limit=10, include=["metadatas"])
            titles = list(set([m.get("title") for m in sample['metadatas']]))
            stats[c.name] = {
                "chunk_count": count,
                "sample_files": titles
            }
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    return jsonify({"collections": stats, "persistent_dir": CHROMA_DIR})

@app.route("/ingest_local", methods=["POST"])
def ingest_local():
    """
    Scans the ./docs folder (or configured DOCS_DIR) and ingests all PDFs.
    JSON Body: {"program": "btech"}
    """
    t0 = time.time()
    data = request.get_json(force=True, silent=True) or {}
    program = data.get("program")

    if not program:
        return jsonify({"error": "Program (e.g., btech) is required"}), 400

    if not os.path.exists(DOCS_DIR):
        return jsonify({"error": f"Directory {DOCS_DIR} not found."}), 404

    pdf_files = glob.glob(os.path.join(DOCS_DIR, "*.pdf"))
    if not pdf_files:
        return jsonify({"message": "No PDF files found in docs folder."}), 200

    col = get_collection(program)
    results = []

    for path in pdf_files:
        filename_on_disk = os.path.basename(path)
        filename = filename_on_disk
        
        # Check if already exists
        if file_already_ingested(col, filename):
            results.append(f"Skipped {filename} (Already exists)")
            continue

        try:
            # CHANGED: Use the new page-aware extractor
            pages_data = extract_markdown_per_page(path)
            
            chunk_count = process_and_ingest_pages(
                pages_data, 
                filename, 
                program, 
                filename_on_disk=filename
            )
            results.append(f"Ingested {filename} ({chunk_count} chunks)")
        except Exception as e:
            results.append(f"Failed {filename}: {str(e)}")

    # Clear BM25 cache so the new files are searchable via keywords immediately
    invalidate_bm25_cache(program)

    return jsonify({
        "ok": True, 
        "summary": results, 
        "ms": int((time.time() - t0) * 1000)
    })

@app.route("/ingest_pdf", methods=["POST"])
def ingest_pdf():
    t0 = time.time()
    f = request.files.get("file")
    program = request.form.get("program")
    title = request.form.get("title") or ""
    effective_from = request.form.get("effective_from") or ""
    source_url = request.form.get("source_url") or ""
    
    if not f or not program:
        return jsonify({"error": "file and program are required"}), 400

    if not os.path.exists(DOCS_DIR): os.makedirs(DOCS_DIR)

    original_filename = secure_filename(f.filename)
    save_path = os.path.join(DOCS_DIR, original_filename)
    f.save(save_path)

    final_title = title or original_filename

    try:
        # CHANGED: Extract Pages
        pages_data = extract_markdown_per_page(save_path)
        
        chunk_count = process_and_ingest_pages(
            pages_data, 
            final_title, 
            program, 
            effective_from, 
            source_url, 
            filename_on_disk=original_filename
        )
        
        # Clear cache
        invalidate_bm25_cache(program)
        
        return jsonify({"ok": True, "chunks": chunk_count, "pages": len(pages_data), "ms": int((time.time() - t0) * 1000)})
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500
    
    
@app.route("/ask", methods=["POST"])
def ask():
    t0 = time.time()
    data = request.get_json(force=True, silent=True) or {}
    program = data.get("program")
    query = data.get("query", "").strip()
    history = data.get("history", [])

    if not program:
        return jsonify({"mode": "ERROR", "answer": "Please provide 'program'."})
    if not query:
        return jsonify({"mode": "GENERIC", "answer": "Please provide a question."})

    col = get_collection(program)
    if col.count() == 0:
        return jsonify({
            "mode": "EMPTY_DB", 
            "answer": f"No documents found for '{program}'. Please upload a PDF or use /ingest_local.",
            "sources": []
        })

    try:
        # --- 1. PREPARE SEARCH QUERY ---
        search_query = query
        if history and len(query.split()) < 3:
            last_user_msg = next((m['text'] for m in reversed(history) if m['role'] == 'user'), "")
            if last_user_msg:
                search_query = f"{last_user_msg} {query}"

        # --- 2. VECTOR SEARCH (Semantic) ---
        # Get top 10 candidates from Vector DB
        q_emb = embed_texts([search_query], task_type="retrieval_query")[0]
        res_vec = col.query(query_embeddings=[q_emb], n_results=10, include=["documents", "metadatas"])
        
        vector_candidates = []
        if res_vec['documents']:
            for doc, meta in zip(res_vec['documents'][0], res_vec['metadatas'][0]):
                vector_candidates.append({"text": doc, "metadata": meta, "origin": "vector"})

        # --- 3. KEYWORD SEARCH (BM25) ---
        # Get top 10 candidates from Keyword Match
        keyword_candidates = []
        bm25_data = get_bm25_data(program)
        if bm25_data:
            tokenized_query = search_query.lower().split()
            # Get scores for all docs
            doc_scores = bm25_data["model"].get_scores(tokenized_query)
            # Find top 10 indices
            top_n_indices = np.argsort(doc_scores)[-10:]
            for idx in reversed(top_n_indices):
                if doc_scores[idx] > 0: # Only include if there is at least some match
                    keyword_candidates.append({
                        "text": bm25_data["texts"][idx],
                        "metadata": bm25_data["metadatas"][idx],
                        "origin": "keyword"
                    })

        # --- 4. MERGE & DEDUPLICATE ---
        unique_candidates = {}
        for c in vector_candidates + keyword_candidates:
            unique_candidates[c['text']] = c # Use text as key to remove duplicates
        
        merged_list = list(unique_candidates.values())

        if not merged_list:
            return jsonify({"mode": "NO_MATCH", "answer": "No relevant documents found.", "sources": []})

        # --- 5. RE-RANKING (Cross-Encoder) ---
        # Score every candidate against the original query
        pairs = [[query, doc['text']] for doc in merged_list]
        scores = CROSS_ENCODER.predict(pairs)

        # Attach scores
        for i, doc in enumerate(merged_list):
            doc['score'] = float(scores[i])
        
        # Sort by Score (High to Low) and take Top 5
        merged_list.sort(key=lambda x: x['score'], reverse=True)
        top_chunks = merged_list[:5]

        # Debugging: Print scores to see what's happening
        print(f"\n--- Results for: {query} ---")
        for i, c in enumerate(top_chunks):
            print(f"{i+1}. [{c['origin']}] Score: {c['score']:.2f} | {c['text'][:40]}...")

    except Exception as e:
        print(f"Search Error: {e}")
        return jsonify({"error": str(e)}), 500

    # --- 6. LLM GENERATION ---
    is_failure = False
    
    # Cross-Encoder Score Threshold (approx > -4.0 implies relevance for this model)
    if top_chunks and top_chunks[0]['score'] > -13.0:
        answer = gemini_chat_answer(query, top_chunks, history)
        sources = [c['metadata'] for c in top_chunks[:3]]
        mode = "RAG"
        
        if "NO_DATA" in answer:
            is_failure = True
            answer = "Sorry, I am unable to find that information in the current documents. Kindly contact the college administration."
            sources = []
    else:
        answer = "I couldn't find relevant details in the provided documents."
        sources = []
        mode = "NO_MATCH"
        is_failure = True

    # --- 7. REPORTING (SQLite) ---
    if is_failure:
        print(f"⚠️ FAILURE DETECTED: {query}")
        suggested_doc = guess_missing_document(query, history)
        report_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        chat_context_str = "\n".join([f"{m.get('role')}: {m.get('text')}" for m in history[-3:]])

        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute('''
                INSERT INTO missing_reports (id, timestamp, program, failed_query, suggested_document, chat_context)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (report_id, timestamp, program, query, suggested_doc, chat_context_str))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error saving report: {e}")

    return jsonify({
        "mode": mode, 
        "answer": answer, 
        "sources": sources, 
        "latency_ms": int((time.time() - t0) * 1000)
    })

@app.route("/admin/reports", methods=["GET"])
def get_reports():
    """Fetch all reports from SQLite."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row # Allows accessing columns by name
        c = conn.cursor()
        c.execute("SELECT * FROM missing_reports ORDER BY timestamp DESC")
        rows = c.fetchall()
        conn.close()
        
        # Convert to list of dicts
        reports = [dict(row) for row in rows]
        return jsonify(reports)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/admin/reports/<report_id>", methods=["DELETE"])
def delete_report(report_id):
    """Delete a specific report."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("DELETE FROM missing_reports WHERE id = ?", (report_id,))
        conn.commit()
        conn.close()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == "__main__":
    # Startup log
    print(f"--- Server Starting ---")
    print(f"DB Path: {CHROMA_DIR}")
    try:
        cols = chroma_client.list_collections()
        if not cols:
            print("Status: Database is empty.")
        else:
            print(f"Status: Found {len(cols)} collections.")
            for c in cols:
                print(f" - {c.name}: {c.count()} chunks")
    except Exception as e:
        print(f"Warning: Could not check DB status: {e}")
        
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=True)