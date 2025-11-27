/**
 * ============================================================================
 * SECTION 1: IMPORTS & CONFIGURATION
 * ============================================================================
 */
import React, { useEffect, useMemo, useRef, useState } from "react";

// Markdown & Styling Libraries
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// PDF Rendering Libraries
import { pdfjs, Document, Page } from 'react-pdf';
import 'react-pdf/dist/Page/AnnotationLayer.css';
import 'react-pdf/dist/Page/TextLayer.css';

// Configure PDF.js worker (Required for Vite/Webpack environments to render PDFs)
pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

/**
 * ============================================================================
 * SECTION 2: UTILITY FUNCTIONS
 * ============================================================================
 */

/**
 * Utility to conditionally join class names.
 * Filters out falsey values (null, undefined, false) to keep class strings clean.
 * @param {...string} classes - List of class strings or conditions.
 */
function cx(...classes) {
  return classes.filter(Boolean).join(" ");
}

/**
 * ============================================================================
 * SECTION 3: CUSTOM HOOKS
 * ============================================================================
 */

/**
 * Generates or retrieves a persistent Session ID from LocalStorage.
 * This ID is used to track conversation history context on the backend.
 */
function useSessionId() {
  const [sid, setSid] = useState("");
  useEffect(() => {
    try {
      const existing = window.localStorage.getItem("ordinance.sid");
      if (existing) {
        setSid(existing);
        return;
      }
    } catch {}
    // Generate new ID if none exists
    const id = (crypto?.randomUUID?.() || `s_${Date.now()}_${Math.random().toString(36).slice(2,8)}`);
    setSid(id);
    try { window.localStorage.setItem("ordinance.sid", id); } catch {}
  }, []);
  return sid;
}

/**
 * ============================================================================
 * SECTION 4: UI ATOM COMPONENTS (Small, Reusable)
 * ============================================================================
 */

/**
 * Displays the user or AI avatar icon with role-based styling.
 */
function Avatar({ role }) {
  return (
    <div className={cx(
      "h-8 w-8 shrink-0 rounded-full flex items-center justify-center text-xs font-semibold shadow",
      role === "user" ? "bg-blue-600 text-white" : "bg-gray-900 text-white"
    )}>
      {role === "user" ? "You" : "AI"}
    </div>
  );
}

/**
 * Animated typing indicator dots displayed while the AI is generating a response.
 */
function TypingDots() {
  return (
    <div className="flex items-end gap-3">
      <Avatar role="assistant" />
      <div className="bg-gray-100 text-gray-900 rounded-2xl rounded-bl-md shadow-sm px-4 py-3">
        <span className="inline-flex gap-1">
          <span className="h-2 w-2 rounded-full bg-gray-500 animate-bounce [animation-delay:-0.2s]"></span>
          <span className="h-2 w-2 rounded-full bg-gray-500 animate-bounce [animation-delay:-0.1s]"></span>
          <span className="h-2 w-2 rounded-full bg-gray-500 animate-bounce"></span>
        </span>
      </div>
    </div>
  );
}

/**
 * Toggle buttons to switch between different programs (B.Tech / B.Arch).
 */
function ProgramToggle({ value, onChange, disabled }) {
  return (
    <div className="inline-flex gap-2 rounded-2xl bg-gray-100 p-1">
      {[
        {key: "btech", label: "B.Tech"},
        {key: "barch", label: "B.Arch"},
      ].map(opt => (
        <button
          key={opt.key}
          disabled={disabled}
          onClick={() => onChange(opt.key)}
          className={cx(
            "px-3 py-1.5 rounded-xl text-sm font-medium transition",
            value === opt.key ? "bg-white shadow border border-gray-200" : "text-gray-600 hover:bg-white"
          )}
          aria-pressed={value === opt.key}
        >{opt.label}</button>
      ))}
    </div>
  );
}

/**
 * ============================================================================
 * SECTION 5: FEATURE COMPONENTS (Complex Logic)
 * ============================================================================
 */

/**
 * Modal overlay to view a PDF document.
 * Includes page navigation props and zoom scaling.
 */
function PdfModal({ url, pageNumber, onClose }) {
  const [numPages, setNumPages] = useState(null);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 animate-in fade-in duration-200">
      <div className="relative bg-white rounded-2xl w-full max-w-4xl h-[90vh] flex flex-col overflow-hidden shadow-2xl">
        
        {/* Modal Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 bg-gray-50">
          <h3 className="text-sm font-semibold text-gray-700">Document Viewer (Page {pageNumber})</h3>
          <button 
            onClick={onClose} 
            className="p-1.5 rounded-full hover:bg-gray-200 transition text-gray-500"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* PDF Rendering Area */}
        <div className="flex-1 overflow-auto bg-gray-100 flex justify-center p-4">
          <Document
            file={url}
            onLoadSuccess={({ numPages }) => setNumPages(numPages)}
            className="shadow-lg"
            loading={<div className="text-gray-500 text-sm">Loading PDF...</div>}
          >
            <Page 
              pageNumber={Number(pageNumber) || 1} 
              renderTextLayer={true} 
              renderAnnotationLayer={true}
              scale={1.2} 
              className="rounded-lg overflow-hidden bg-white"
            />
          </Document>
        </div>
      </div>
    </div>
  );
}

/**
 * Displays a citation source.
 * Handles logic to open either an external URL or the internal PDF Viewer modal.
 */
function SourceCard({ s, backendUrl, onView }) {
  const isLocal = s.filename && !s.source_url?.startsWith("http");
  let link = null;
  
  if (s.source_url && s.source_url.startsWith("http")) {
    link = s.source_url;
  } else if (s.filename) {
    link = `${backendUrl}/documents/${s.filename}`;
  }

  return (
    <div className="rounded-2xl border border-gray-200 p-4 bg-white shadow-sm hover:shadow-md transition-shadow">
      <div className="text-sm font-semibold text-gray-900 line-clamp-2">
        {s.title || "(Untitled)"}
      </div>
      
      <div className="mt-2 space-y-1">
        <div className="text-xs text-gray-500 flex justify-between">
            <span>Program:</span> 
            <span className="font-medium text-gray-700">{(s.program || "—").toUpperCase()}</span>
        </div>
        <div className="text-xs text-gray-500 flex justify-between">
            <span>Effective:</span>
            <span className="font-medium text-gray-700">{s.effective_from || "—"}</span>
        </div>
        <div className="text-xs text-gray-500 flex justify-between mt-1 pt-1 border-t border-gray-50">
          <span>Page:</span>
          <span className="font-medium text-gray-700">{s.page || "1"}</span>
        </div>
      </div>

      <button 
        onClick={(e) => {
          e.preventDefault();
          if (isLocal) {
            onView(s.filename, s.page); 
          } else if (link) {
            window.open(link, "_blank");
          }
        }}
        className="mt-3 flex items-center justify-center w-full rounded-xl bg-blue-50 px-3 py-2 text-xs font-medium text-blue-700 hover:bg-blue-100 transition-colors"
      >
        <svg className="mr-1.5 h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
        </svg>
        {isLocal ? `View on Page ${s.page || 1}` : "Open External Link"}
      </button>
    </div>
  );
}

/**
 * Chat message bubble.
 * Handles Markdown rendering (with GFM tables) and AI Feedback buttons (up/down vote).
 */
function MessageBubble({ role, text, relatedQuery, program, backendUrl }) {
  const isUser = role === "user";
  const [vote, setVote] = useState(null); // 'up' | 'down' | null

  const handleVote = async (rating) => {
    if (vote) return; // Prevent double voting
    setVote(rating);

    try {
      await fetch(`${backendUrl}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          program: program,
          query: relatedQuery,
          response: text,
          rating: rating
        })
      });
    } catch (e) {
      console.error("Feedback failed", e);
    }
  };
  
  return (
    <div className={cx("flex items-end gap-3", isUser ? "justify-end" : "justify-start")}>
      {!isUser && <Avatar role={role} />}
      <div className={cx(
        "max-w-[85%] rounded-2xl px-5 py-3 text-sm leading-relaxed shadow-sm overflow-hidden group relative",
        isUser ? "bg-blue-600 text-white rounded-br-md" : "bg-gray-100 text-gray-900 rounded-bl-md"
      )}>
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            ul: ({node, ...props}) => <ul className="list-disc pl-4 mb-2 space-y-1" {...props} />,
            ol: ({node, ...props}) => <ol className="list-decimal pl-4 mb-2 space-y-1" {...props} />,
            li: ({node, ...props}) => <li className="pl-1" {...props} />,
            p: ({node, ...props}) => <p className="mb-2 last:mb-0" {...props} />,
            strong: ({node, ...props}) => <span className="font-bold" {...props} />,
            a: ({node, ...props}) => (
              <a 
                {...props} 
                target="_blank" 
                rel="noreferrer" 
                className={`underline ${isUser ? 'text-white' : 'text-blue-600 hover:text-blue-800'}`} 
              />
            ),
            // Custom Table Styling for GFM
            table: ({node, ...props}) => (
              <div className="overflow-x-auto my-4 rounded-lg border border-gray-200">
                <table className="min-w-full divide-y divide-gray-200 bg-white text-sm" {...props} />
              </div>
            ),
            thead: ({node, ...props}) => <thead className="bg-gray-50" {...props} />,
            tbody: ({node, ...props}) => <tbody className="divide-y divide-gray-200 bg-white" {...props} />,
            tr: ({node, ...props}) => <tr className="" {...props} />,
            th: ({node, ...props}) => (
              <th className="px-3 py-3.5 text-left text-xs font-semibold text-gray-900 uppercase tracking-wider" {...props} />
            ),
            td: ({node, ...props}) => (
              <td className="whitespace-normal px-3 py-4 text-gray-700" {...props} />
            ),
          }}
        >
          {text}
        </ReactMarkdown>

        {/* Feedback Buttons (Only for AI responses) */}
        {!isUser && (
          <div className="mt-2 pt-2 border-t border-gray-200/50 flex gap-3 opacity-100 sm:opacity-0 sm:group-hover:opacity-100 transition-opacity">
            <button 
              onClick={() => handleVote("up")}
              disabled={vote !== null}
              className={cx(
                "flex items-center gap-1 text-xs transition",
                vote === "up" ? "text-green-600 font-medium" : "text-gray-400 hover:text-green-600"
              )}
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L7 11v9m7-10h-2M7 20H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5" />
              </svg>
              {vote === "up" ? "Helpful" : ""}
            </button>
            <button 
               onClick={() => handleVote("down")}
               disabled={vote !== null}
               className={cx(
                 "flex items-center gap-1 text-xs transition",
                 vote === "down" ? "text-amber-600 font-medium" : "text-gray-400 hover:text-amber-600"
               )}
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 14H5.236a2 2 0 01-1.789-2.894l3.5-7A2 2 0 018.736 3h4.018a2 2 0 01.485.06l2.94 1.108V11a2 2 0 01-2 2h-2.5m-4 9V20m0 0h2m-2 0H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5M20 7v6a2 2 0 01-2 2h-2.5" />
              </svg>
              {vote === "down" ? "Not Helpful" : ""}
            </button>
          </div>
        )}
      </div>
      {isUser && <Avatar role={role} />}
    </div>
  );
}

/**
 * ============================================================================
 * SECTION 6: ADMIN & DASHBOARD COMPONENTS
 * ============================================================================
 */

/**
 * Admin view to list reported missing content failures.
 * Allows deleting reports after resolution.
 */
function ReportsList({ backendUrl }) {
  const [reports, setReports] = useState([]);
  const [visible, setVisible] = useState(false);

  const fetchReports = async () => {
    try {
      const res = await fetch(`${backendUrl}/admin/reports`);
      const data = await res.json();
      setReports(data);
    } catch (e) {
      console.error("Failed to fetch reports", e);
    }
  };

  const handleDelete = async (id, e) => {
    e.stopPropagation(); 
    if (!confirm("Are you sure you want to delete this report?")) return;

    try {
      await fetch(`${backendUrl}/admin/reports/${id}`, { method: "DELETE" });
      setReports(prev => prev.filter(r => r.id !== id));
    } catch (error) {
      alert("Failed to delete report");
    }
  };

  return (
    <div className="mt-6 border-t border-gray-100 pt-4">
      <button 
        onClick={() => {
            if(!visible) fetchReports();
            setVisible(!visible);
        }}
        className="flex items-center gap-2 text-sm font-medium text-gray-700 w-full hover:bg-gray-50 p-2 rounded-lg transition"
      >
        <div className="relative">
             <span className="flex items-center justify-center w-5 h-5 rounded bg-red-100 text-red-600 text-[10px] font-bold">!</span>
             {reports.length > 0 && !visible && (
                 <span className="absolute -top-1 -right-1 flex h-2.5 w-2.5">
                   <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                   <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-red-500"></span>
                 </span>
             )}
        </div>
        Missing Content Reports
        <span className="ml-auto text-xs text-gray-400">{visible ? "Hide" : "Show"}</span>
      </button>

      {visible && (
        <div className="mt-3 space-y-3 max-h-80 overflow-y-auto pr-1 scrollbar-thin">
          {reports.length === 0 && <div className="text-xs text-gray-400 italic p-2">No failures reported yet.</div>}
          
          {reports.map((r) => (
            <div key={r.id} className="group relative p-3 bg-red-50 rounded-xl border border-red-100 text-xs transition hover:shadow-sm hover:border-red-200">
              <button 
                onClick={(e) => handleDelete(r.id, e)}
                className="absolute top-2 right-2 p-1.5 rounded-full bg-white text-gray-400 hover:text-red-600 hover:bg-red-50 opacity-0 group-hover:opacity-100 transition shadow-sm"
                title="Mark as Resolved (Delete)"
              >
                <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
              </button>

              <div className="font-semibold text-gray-800 mb-1 pr-6 flex items-center gap-1.5">
                <svg className="w-3.5 h-3.5 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                   <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                Req: {r.suggested_document}
              </div>
              <div className="text-gray-600 mb-2 leading-relaxed">
                <span className="font-medium text-gray-500">Query:</span> "{r.failed_query}"
              </div>
              <div className="flex justify-between items-center pt-2 border-t border-red-100/50 text-[10px] text-gray-400 uppercase tracking-wider">
                <span className="bg-white px-1.5 py-0.5 rounded border border-red-100">{r.program}</span>
                <span>{r.timestamp}</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/**
 * Admin view to list negative user feedback.
 * Allows deleting/acknowledging feedback.
 */
function FeedbackList({ backendUrl }) {
  const [items, setItems] = useState([]);
  const [visible, setVisible] = useState(false);

  const fetchFeedback = async () => {
    try {
      const res = await fetch(`${backendUrl}/admin/feedback`);
      const data = await res.json();
      setItems(data);
    } catch (e) {
      console.error("Failed to fetch feedback", e);
    }
  };

  const handleDelete = async (id, e) => {
    e.stopPropagation();
    if (!confirm("Delete this feedback alert?")) return;
    try {
      await fetch(`${backendUrl}/admin/feedback/${id}`, { method: "DELETE" });
      setItems(prev => prev.filter(i => i.id !== id));
    } catch (error) { alert("Failed"); }
  };

  return (
    <div className="mt-4 border-t border-gray-100 pt-4">
      <button 
        onClick={() => { if(!visible) fetchFeedback(); setVisible(!visible); }}
        className="flex items-center gap-2 text-sm font-medium text-gray-700 w-full hover:bg-gray-50 p-2 rounded-lg transition"
      >
        <div className="relative">
             <span className="flex items-center justify-center w-5 h-5 rounded bg-amber-100 text-amber-600 text-[10px] font-bold">👎</span>
             {items.length > 0 && !visible && (
                 <span className="absolute -top-1 -right-1 flex h-2.5 w-2.5">
                   <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75"></span>
                   <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-amber-500"></span>
                 </span>
             )}
        </div>
        Negative Feedback
        <span className="ml-auto text-xs text-gray-400">{visible ? "Hide" : "Show"}</span>
      </button>

      {visible && (
        <div className="mt-3 space-y-3 max-h-60 overflow-y-auto pr-1 scrollbar-thin">
          {items.length === 0 && <div className="text-xs text-gray-400 italic p-2">No negative feedback.</div>}
          {items.map((item) => (
            <div key={item.id} className="group relative p-3 bg-amber-50 rounded-xl border border-amber-100 text-xs transition hover:shadow-sm">
               <button 
                onClick={(e) => handleDelete(item.id, e)}
                className="absolute top-2 right-2 p-1.5 rounded-full bg-white text-gray-400 hover:text-red-600 hover:bg-red-50 opacity-0 group-hover:opacity-100 transition shadow-sm"
              >
                <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
              </button>
              <div className="font-medium text-gray-800 mb-1">Query: "{item.query}"</div>
              <div className="text-gray-500 line-clamp-2 mb-2">Bot: {item.response.substring(0, 80)}...</div>
              <div className="text-[10px] text-gray-400 uppercase">{item.timestamp}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/**
 * ============================================================================
 * SECTION 7: MAIN APPLICATION COMPONENT
 * ============================================================================
 */
export default function App() {
  // --- STATE MANAGEMENT ---
  const sessionId = useSessionId();
  const [BACKEND_URL, setBackendUrl] = useState("http://localhost:8080");
  const [program, setProgram] = useState(null);
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [answerSources, setAnswerSources] = useState([]);
  const [lastLatency, setLastLatency] = useState(null);
  const [mode, setMode] = useState(null); 
  
  // PDF Viewer Modal State
  const [viewPdf, setViewPdf] = useState(null); // { url: string, page: number }

  // Ingestion Form State
  const [ingFile, setIngFile] = useState(null);
  const [ingTitle, setIngTitle] = useState("");
  const [ingEff, setIngEff] = useState("");
  const [ingUrl, setIngUrl] = useState("");
  const [ingStatus, setIngStatus] = useState("");

  // Refs & Memoization
  const fileInputRef = useRef(null);
  const canAsk = useMemo(() => !!query.trim() && !!sessionId, [query, sessionId]);
  const chatScrollRef = useRef(null);

  // Suggestions List
  const suggestions = [
    "What is the attendance requirement?",
    "Is a 6-month internship allowed in 8th semester?",
    "How is grades calculated?",
    "How long is the course?"
  ];

  // --- EFFECTS ---

  // Auto-scroll chat to bottom when new messages arrive
  useEffect(() => {
    const el = chatScrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [messages, loading]);

  // --- EVENT HANDLERS ---

  // Handle User Question Submission
  async function handleAsk() {
    if (!canAsk) return;
    if (!program) {
      setMessages(prev => [...prev, { role: "assistant", text: "Please select a stream (B.Tech or B.Arch) above before asking." }]);
      return;
    }
    const q = query.trim();
    const newUserMsg = { role: "user", text: q };
    
    setMessages(prev => [...prev, newUserMsg]);
    setQuery("");
    setLoading(true);
    setAnswerSources([]);

    try {
      const res = await fetch(`${BACKEND_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
            session_id: sessionId, 
            program, 
            query: q,
            history: messages 
        })
      });
      const data = await res.json();
      setMode(data.mode || null);
      setLastLatency(data.latency_ms ?? null);

      if (data.mode === "SELECT_PROGRAM") {
        setMessages(prev => [...prev, { role: "assistant", text: data.message || "Please select a stream: B.Tech or B.Arch." }]);
      } else {
        const text = data.answer || JSON.stringify(data);
        setMessages(prev => [...prev, { role: "assistant", text }]);
        setAnswerSources(Array.isArray(data.sources) ? data.sources : []);
      }
    } catch (err) {
      setMessages(prev => [...prev, { role: "assistant", text: `Error: ${err}` }]);
    } finally {
      setLoading(false);
    }
  }

  // Handle PDF Ingestion Upload
  async function handleIngest() {
    if (!ingFile) { 
        setIngStatus("Choose a PDF first."); 
        return; 
    }
    if (!program) { 
        setIngStatus("Select B.Tech or B.Arch before uploading."); 
        return; 
    }

    let effectiveFrom = ingEff || new Date().toISOString().split('T')[0];
    let sourceUrl = ingUrl || "empty";

    const fd = new FormData();
    fd.append("file", ingFile);
    fd.append("program", program);
    fd.append("title", ingTitle || ingFile.name); // Use file name as fallback title
    fd.append("effective_from", effectiveFrom);
    fd.append("source_url", sourceUrl);

    setIngStatus("Uploading…");

    try {
        const res = await fetch(`${BACKEND_URL}/ingest_pdf`, { method: "POST", body: fd });
        const data = await res.json();
        if (res.ok) {
            setIngStatus(`OK: ${data.chunks ?? "?"} chunks from ${data.pages ?? "?"} pages`);
            setIngFile(null);
            if (fileInputRef.current) fileInputRef.current.value = ""; 
        } else {
            setIngStatus(`Failed: ${data.error || res.status}`);
        }
    } catch (e) {
        setIngStatus(`Error: ${e}`);
    }
  }

  // Handle Pre-set Suggestions
  function handleSuggestionClick(s) {
    setQuery(s);
    setTimeout(() => {
      if (program) handleAsk();
    }, 0);
  }

  // --- RENDER ---
  return (
    <div className="min-h-screen bg-[radial-gradient(ellipse_at_top,rgba(59,130,246,0.08),transparent_50%),linear-gradient(to_bottom,#ffffff,#f8fafc)] text-gray-900">
      
      {/* PDF Modal Render */}
      {viewPdf && (
        <PdfModal 
          url={viewPdf.url} 
          pageNumber={viewPdf.page} 
          onClose={() => setViewPdf(null)} 
        />
      )}

      {/* Header Bar */}
      <header className="sticky top-0 z-10 backdrop-blur supports-[backdrop-filter]:bg-white/70 bg-white/60 border-b border-gray-200">
        <div className="mx-auto max-w-6xl px-4 py-3 flex items-center gap-4">
          <div className="flex items-center gap-3">
            <div className="h-8 w-8 rounded-xl bg-blue-600 text-white flex items-center justify-center text-sm font-bold shadow">NITRR</div>
            <div>
              <div className="text-lg font-semibold leading-5">NITRR - Academic Ordinance Assistant</div>
              <div className="text-[11px] text-gray-500 -mt-0.5">Answering from official documents with citations</div>
            </div>
          </div>
          <div className="ml-auto flex items-center gap-3">
            <ProgramToggle value={program} onChange={setProgram} />
            <div className="hidden sm:flex items-center gap-2 text-xs text-gray-600">
              {mode && <span className="px-2 py-1 rounded-full bg-gray-100 border border-gray-200">{mode}</span>}
              {lastLatency != null && (
                <span className="px-2 py-1 rounded-full bg-gray-100 border border-gray-200">{lastLatency} ms</span>
              )}
            </div>
            <input
              className="w-[16rem] rounded-xl border border-gray-300 px-3 py-1.5 text-sm focus:outline-none focus:ring focus:ring-blue-200"
              value={BACKEND_URL}
              onChange={(e) => setBackendUrl(e.target.value)}
              placeholder="Backend URL (e.g., http://localhost:8080)"
            />
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="mx-auto max-w-6xl px-4 py-6 grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Chat Interface Column */}
        <section className="lg:col-span-2 rounded-3xl border border-gray-200 bg-white shadow-sm p-0 flex flex-col min-h-[72vh]">
          <div className="px-5 pt-4 pb-2 text-xs text-gray-600 border-b border-gray-100">
            Session: <span className="font-mono">{sessionId || "(generating…)"}</span>
          </div>

          {/* Messages Area */}
          <div
            ref={chatScrollRef}
            className="flex-1 overflow-y-auto px-5 py-4 space-y-4 scroll-smooth"
          >
            {messages.length === 0 && (
              <div className="text-sm text-gray-500 bg-gray-50 border border-dashed border-gray-200 rounded-2xl p-4">
                Ask about <span className="font-medium">attendance, internships, CGPA, re-evaluation</span> and more. Select your stream above first.
                <div className="mt-3 flex flex-wrap gap-2">
                  {suggestions.map((s, i) => (
                    <button
                      key={i}
                      onClick={() => handleSuggestionClick(s)}
                      className="text-xs px-3 py-1.5 rounded-full border border-gray-200 bg-gray-50 hover:bg-gray-100"
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            )}
            {messages.map((m, i) => (
              <MessageBubble 
                key={i} 
                role={m.role} 
                text={m.text}
                relatedQuery={m.role === "assistant" && i > 0 ? messages[i-1].text : ""}
                program={program}
                backendUrl={BACKEND_URL}
              />
            ))}
            {loading && <TypingDots />}
          </div>

          {/* Input Area */}
          <div className="px-5 pt-2 pb-4 border-t border-gray-100">
            <div className="flex items-center gap-2">
              <input
                className="flex-1 rounded-2xl border border-gray-300 px-4 py-3 text-sm focus:outline-none focus:ring focus:ring-blue-200"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder={program ? "Type your question…  (Enter to send)" : "Select B.Tech / B.Arch first…"}
                onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleAsk(); } }}
              />
              <button
                onClick={handleAsk}
                disabled={!canAsk || loading || !program}
                className={cx(
                  "px-5 py-3 rounded-2xl text-sm font-medium transition shadow-sm",
                  (!canAsk || loading || !program) ? "bg-gray-200 text-gray-500" : "bg-blue-600 text-white hover:bg-blue-700"
                )}
              >{loading ? "Asking…" : "Send"}</button>
            </div>

            {/* Answer Sources Display */}
            {answerSources.length > 0 && (
              <div className="mt-4">
                <div className="text-sm font-semibold mb-2 flex items-center gap-2">
                  Sources
                  <span className="text-xs text-gray-500">({answerSources.length})</span>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  {answerSources.map((s, i) => (
                    <SourceCard 
                      key={i} 
                      s={s} 
                      backendUrl={BACKEND_URL}
                      onView={(filename, page) => {
                        setViewPdf({
                          url: `${BACKEND_URL}/documents/${filename}`,
                          page: Number(page) || 1
                        });
                      }}
                    />
                  ))}
                </div>
              </div>
            )}
          </div>
        </section>

        {/* Sidebar / Ingestion Column */}
        <aside className="rounded-3xl border border-gray-200 bg-white shadow-sm p-4">
          <div className="text-base font-semibold mb-1">Ingest Ordinance / Notices</div>
          <div className="text-xs text-gray-600 mb-3">Upload a PDF into the selected stream’s index. Add metadata to improve citations.</div>

          <div className="space-y-3">
            <input ref={fileInputRef} type="file" accept="application/pdf" onChange={(e) => setIngFile(e.target.files?.[0] || null)} />
            <input className="w-full rounded-xl border border-gray-300 px-3 py-2 text-sm" placeholder="Title (optional)" value={ingTitle} onChange={(e)=>setIngTitle(e.target.value)} />
            <input className="w-full rounded-xl border border-gray-300 px-3 py-2 text-sm" placeholder="Effective from (YYYY-MM-DD) (optional)" value={ingEff} onChange={(e)=>setIngEff(e.target.value)} />
            <input className="w-full rounded-xl border border-gray-300 px-3 py-2 text-sm" placeholder="Source URL (optional)" value={ingUrl} onChange={(e)=>setIngUrl(e.target.value)} />
            <button
                onClick={handleIngest}
                disabled={!program || loading}
                className={cx(
                    "w-full px-4 py-2 rounded-xl text-sm font-medium transition",
                    (!program || loading) ? "bg-gray-200 text-gray-500" : "bg-green-600 text-white hover:bg-green-700"
                )}
            >
                Upload to {program ? program.toUpperCase() : "(select stream)"}
            </button>

            {ingStatus && <div className="text-xs text-gray-700">{ingStatus}</div>}

            <div className="pt-3 border-t border-gray-100 text-xs text-gray-500">
              Tip: For first run, upload at least one PDF per stream (B.Tech / B.Arch).
            </div>
          </div>
          
          {/* Admin Lists */}
          <ReportsList backendUrl={BACKEND_URL} />
          <FeedbackList backendUrl={BACKEND_URL} />
        </aside>
      </main>

      {/* Footer */}
      <footer className="mx-auto max-w-6xl px-4 py-6 text-xs text-gray-500">
        © {new Date().getFullYear()} Ordinance Assistant
      </footer>
    </div>
  );
}