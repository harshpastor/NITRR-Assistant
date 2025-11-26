#!/usr/bin/env python3
import os
import time
import uuid
import glob

from typing import List, Dict, Any
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from flask import send_from_directory

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

# ------------ Utilities -------------

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

# ... (imports remain the same)

# 1. Update this function to accept history
def gemini_chat_answer(question: str, context_chunks: List[Dict[str, Any]], history: List[Dict[str, str]]) -> str:
    # Improved System Prompt
    system_prompt = (
        "You are an intelligent Academic Ordinance Assistant.\n"
        "Your goal is to answer student questions based ONLY on the provided context.\n\n"
        "RULES:\n"
        "1. If the user's question is ambiguous (e.g., 'how many semesters?' without specifying context), "
        "DO NOT guess. Instead, ask a polite clarifying question (e.g., 'Do you mean per year or for the entire program?').\n"
        "2. If the user answers a previous clarification (e.g., they just say 'Total'), combine it with the chat history "
        "to understand they mean 'Total semesters in the program'.\n"
        "3. Always cite the source title if you find the answer.\n"
        "4. If the answer is not in the context, say 'Sorry, I am able find that information, Kindly contact the college administration.'"
        "5. Be concise and clear with your answers. Make the answer more structured and pointwise wherever possible\n"
    )
    
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

# ... (Previous code remains)

# ... (Rest of code remains)

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
            text = pymupdf4llm.to_markdown(path)
            
            chunk_count = process_and_ingest(text, filename, program, filename_on_disk=filename)
            # full_text = []
            # for page in reader.pages:
            #     full_text.append(page.extract_text() or "")
            # text = "\n\n".join(full_text).strip()
            
            chunk_count = process_and_ingest(
                text, 
                filename, 
                program, 
                filename_on_disk=filename_on_disk # <--- PASSING FILENAME
            )
            results.append(f"Ingested {filename_on_disk} ({chunk_count} chunks)")
        except Exception as e:
            results.append(f"Failed {filename_on_disk}: {str(e)}")

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

    # Ensure DOCS_DIR exists
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)

    # Secure and Save file
    original_filename = secure_filename(f.filename)
    save_path = os.path.join(DOCS_DIR, original_filename)
    f.save(save_path) # <--- SAVING FILE PHYSICALLY

    # Use user-provided title OR filename as the display title
    final_title = title or original_filename

    try:
        # Re-read the saved file for processing
        # reader = PdfReader(save_path)
        # full_text = []
        # for page in reader.pages:
        #     full_text.append(page.extract_text() or "")
        # text = "\n\n".join(full_text).strip()
        
        # # Pass the filename_on_disk to the helper
        # chunk_count = process_and_ingest(
        #     text, 
        #     final_title, 
        #     program, 
        #     effective_from, 
        #     source_url, 
        #     filename_on_disk=original_filename # <--- PASSING FILENAME
        # )
        text = pymupdf4llm.to_markdown(save_path)
        
        chunk_count = process_and_ingest(
            text, 
            final_title, 
            program, 
            effective_from, 
            source_url, 
            filename_on_disk=original_filename
        )
        
        return jsonify({"ok": True, "chunks": chunk_count, "ms": int((time.time() - t0) * 1000)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ask", methods=["POST"])
def ask():
    t0 = time.time()
    data = request.get_json(force=True, silent=True) or {}
    program = data.get("program")
    query = data.get("query", "").strip()
    history = data.get("history", []) # <--- Get history from frontend

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
        # --- IMPROVEMENT: CONTEXTUAL SEARCH ---
        # If the query is very short (like "Total"), it might be an answer to a previous question.
        # We append the previous user message to the search query to get better vector results.
        search_query = query
        if history and len(query.split()) < 3:
            last_user_msg = next((m['text'] for m in reversed(history) if m['role'] == 'user'), "")
            if last_user_msg:
                # Combine for search: "How many semesters? Total"
                search_query = f"{last_user_msg} {query}"
        
        # Embed the (possibly enhanced) query
        q_emb = embed_texts([search_query], task_type="retrieval_query")[0]
        
        res = col.query(query_embeddings=[q_emb], n_results=5, include=["documents", "metadatas", "distances"])
        
        docs = res['documents'][0]
        metas = res['metadatas'][0]
        dists = res['distances'][0]
        
        top_chunks = [{"text": d, "metadata": m, "distance": dist} for d, m, dist in zip(docs, metas, dists)]
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # RAG Logic
    # We lowered the threshold slightly to allow the LLM to see context and decide if it's ambiguous
    if top_chunks and top_chunks[0]["distance"] < 0.55: 
        # Pass history to the chat function
        answer = gemini_chat_answer(query, top_chunks, history)
        sources = [c['metadata'] for c in top_chunks[:3]]
        mode = "RAG"
    else:
        answer = "I couldn't find relevant details in the provided documents."
        sources = []
        mode = "NO_MATCH"

    return jsonify({
        "mode": mode, 
        "answer": answer, 
        "sources": sources, 
        "latency_ms": int((time.time() - t0) * 1000)
    })


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