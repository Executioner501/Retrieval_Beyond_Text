"""
FastAPI server for conversational QA over WhatsApp exports.

Run from the project root:
    uvicorn src.api.server:app --reload

Then open http://localhost:8000/docs to try the endpoints.

Environment variables (optional):
    OLLAMA_MODEL  default llama3.2:3b
    OLLAMA_URL    default http://localhost:11434
    EMBEDDER      minilm (default) or bge-m3
"""

import os
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..generation.ollama_client import DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_URL, OllamaClient
from ..indexing.embedders import get_embedder
from ..pipeline import ConversationQA

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_CHAT = ROOT / "data" / "raw" / "sample_chat.txt"
UI_DIR = ROOT / "ui"

app = FastAPI(title="Context-Aware Conversational RAG", version="0.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_llm = OllamaClient(
    model=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
    base_url=os.getenv("OLLAMA_URL", DEFAULT_OLLAMA_URL),
)
_embedder_name = os.getenv("EMBEDDER", "minilm")
_encoder = None
_chats: Dict[str, Dict] = {}  # chat_id -> {"name": str, "qa": ConversationQA}


def _get_encoder():
    global _encoder
    if _encoder is None:
        _encoder = get_embedder(_embedder_name)
    return _encoder


def _register(path: str, name: str) -> Dict:
    try:
        qa = ConversationQA.from_file(path, llm=_llm, encoder=_get_encoder())
    except ValueError as e:
        raise HTTPException(400, str(e))
    chat_id = uuid.uuid4().hex[:8]
    _chats[chat_id] = {"name": name, "qa": qa}
    return {"chat_id": chat_id, "name": name, **qa.stats()}


class Turn(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    question: str
    history: Optional[List[Turn]] = None


@app.get("/api/health")
def health():
    return {"ollama_available": _llm.is_available(), "ollama_model": _llm.model,
            "embedder": _embedder_name, "chats_loaded": len(_chats)}


@app.post("/api/chats")
def upload_chat(file: UploadFile = File(...)):
    """Upload a WhatsApp .txt export; it is parsed, chunked and indexed in memory."""
    if not (file.filename or "").lower().endswith(".txt"):
        raise HTTPException(400, "Please upload a WhatsApp export (.txt)")
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name
    try:
        return _register(tmp_path, file.filename)
    finally:
        os.unlink(tmp_path)  # the raw chat is not kept on disk


@app.post("/api/chats/demo")
def load_demo_chat():
    """Loads data/raw/sample_chat.txt, handy for demos."""
    return _register(str(SAMPLE_CHAT), SAMPLE_CHAT.name)


@app.get("/api/chats")
def list_chats():
    return [{"chat_id": cid, "name": c["name"], **c["qa"].stats()} for cid, c in _chats.items()]


@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: str):
    if _chats.pop(chat_id, None) is None:
        raise HTTPException(404, "Chat not found")
    return {"deleted": chat_id}


@app.post("/api/chats/{chat_id}/ask")
def ask(chat_id: str, req: AskRequest):
    chat = _chats.get(chat_id)
    if chat is None:
        raise HTTPException(404, "Chat not found. Upload it again (chats are kept in memory only).")
    try:
        history = [t.model_dump() for t in req.history] if req.history else None
        result = chat["qa"].ask(req.question, history=history)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if result.get("error"):
        # Local LLM is offline, but evidence chunks are successfully retrieved
        result["answer"] = (
            f"[Local LLM generation offline: {result['error']}]\n\n"
            f"Displaying top retrieved conversation evidence chunks with speaker attribution and timestamps below."
        )
    return result


# Serve the web UI at / (API routes above take priority)
if UI_DIR.exists():
    app.mount("/", StaticFiles(directory=str(UI_DIR), html=True), name="ui")
