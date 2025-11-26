# College RAG Chatbot (OpenAI version)

This bundle swaps **Ollama** for the **OpenAI API** for generation and embeddings.

## Structure
```
college-rag-openai/
  backend/   # Flask + ChromaDB + OpenAI
  frontend/  # Vite + React (Tailwind via CDN)
```

## 1) Backend
See `backend/README.md`, or quick run:
```bash
cd backend
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# set env (or copy .env.example and export)
export OPENAI_API_KEY=sk-...
export OPENAI_CHAT_MODEL=gpt-4o-mini
export OPENAI_EMBED_MODEL=text-embedding-3-small
export CHROMA_DIR=./chroma
export PORT=8080

python app.py
```

### Test
- Ingest: POST `http://localhost:8080/ingest_pdf` (multipart) with fields `file`, `program` (`btech` or `barch`), optional `title`, `effective_from`, `source_url`.
- Ask: POST `http://localhost:8080/ask` with JSON `{"session_id":"test","program":"btech","query":"What is attendance requirement?"}`.

## 2) Frontend
```bash
cd frontend
npm i
npm run dev  # open the URL printed by Vite
```
Update the **Backend URL** field in the header if your backend URL differs.

## Notes
- ChromaDB persists under `backend/chroma/` by default.
- The UI preserves all **variable names** and request/response shapes used earlier.
- Confidence gating switches between **RAG** and **GENERIC** modes; tweak threshold in `/ask` as needed.
