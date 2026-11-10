"""
Switchable embedding backends.

    minilm  sentence-transformers/all-MiniLM-L6-v2   (384-d, reads 256 tokens, English)
    bge-m3  BAAI bge-m3 served by Ollama              (1024-d, reads 8192 tokens, multilingual)

Every backend exposes the same .encode(texts, ...) -> np.ndarray interface as a
SentenceTransformer, so the chunker, vector store and pipeline use them unchanged.

Embeddings are cached on disk (.cache/embeddings/<name>.pkl), so re-running an
evaluation with the slower bge-m3 model does not re-embed the whole chat.
"""

import hashlib
import json
import os
import urllib.error
import urllib.request
from typing import Dict, List, Optional

import numpy as np

CACHE_DIR = os.path.join(".cache", "embeddings")
CACHE_VERSION = 1

# min_relevance: starting guess for the "nothing relevant found" cut-off on the best
# semantic score. Calibrate it from the eval output (see compare_evals.py).
EMBEDDERS = {
    "minilm": {"model": "sentence-transformers/all-MiniLM-L6-v2", "min_relevance": 0.20},
    "bge-m3": {"model": "bge-m3", "min_relevance": 0.40},
    "me5": {"model": "intfloat/multilingual-e5-small", "min_relevance": 0.30},
}


def _normalise(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


class MiniLMEmbedder:
    name = "minilm"

    def __init__(self, model: str = EMBEDDERS["minilm"]["model"]):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError("MiniLM needs sentence-transformers: pip install sentence-transformers") from e
        self.model = SentenceTransformer(model)

    def encode(self, texts: List[str], **_) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                                 show_progress_bar=False, batch_size=64)


class ME5Embedder:
    """Multilingual E5 small embedder. Prepends 'passage: ' or 'query: '."""
    name = "me5"

    def __init__(self, model: str = EMBEDDERS["me5"]["model"]):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError("ME5 needs sentence-transformers: pip install sentence-transformers") from e
        self.model = SentenceTransformer(model)

    def encode(self, texts: List[str], is_query: bool = False, **kwargs) -> np.ndarray:
        prefix = "query: " if is_query else "passage: "
        prefixed = [f"{prefix}{t}" for t in texts]
        batch_size = kwargs.get("batch_size", 64)
        return self.model.encode(prefixed, convert_to_numpy=True, normalize_embeddings=True,
                                 show_progress_bar=False, batch_size=batch_size)


class OllamaEmbedder:
    """Calls Ollama's /api/embed. Run `ollama pull bge-m3` first."""
    name = "bge-m3"

    def __init__(self, model: str = EMBEDDERS["bge-m3"]["model"],
                 base_url: str = "http://localhost:11434", batch_size: int = 32,
                 num_ctx: int = 8192, timeout: int = 600):
        self.model, self.base_url = model, base_url.rstrip("/")
        self.batch_size, self.num_ctx, self.timeout = batch_size, num_ctx, timeout

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        payload = {"model": self.model, "input": texts, "truncate": True,
                   "options": {"num_ctx": self.num_ctx}}
        req = urllib.request.Request(f"{self.base_url}/api/embed",
                                     data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Ollama embed failed ({e.code}): {e.read().decode(errors='ignore')}. "
                               f"Did you run: ollama pull {self.model} ?") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Cannot reach Ollama at {self.base_url} (ollama serve)") from e
        return np.array(data["embeddings"], dtype=np.float32)

    def encode(self, texts: List[str], **_) -> np.ndarray:
        # Ollama rejects empty strings; give them a placeholder
        texts = [t if t.strip() else "." for t in texts]
        out = [self._embed_batch(texts[i:i + self.batch_size])
               for i in range(0, len(texts), self.batch_size)]
        return _normalise(np.vstack(out))


class CachedEmbedder:
    """Wraps any embedder with a persistent text -> vector cache."""

    CACHE_VERSION = CACHE_VERSION

    def __init__(self, inner, cache_name: str):
        self.inner = inner
        self.model_name = getattr(inner, "name", getattr(inner, "model_name", cache_name))
        self.name = self.model_name
        version = getattr(self, "CACHE_VERSION", CACHE_VERSION)
        os.makedirs(CACHE_DIR, exist_ok=True)
        self.path = os.path.join(CACHE_DIR, f"{cache_name}_v{version}.npz")
        self.cache: Dict[str, np.ndarray] = {}
        if os.path.exists(self.path):
            try:
                with np.load(self.path, allow_pickle=False) as data:
                    file_ver = int(data["version"]) if "version" in data else None
                    if file_ver == version:
                        keys = [str(k) for k in data["keys"]]
                        vectors = data["vectors"]
                        for k, v in zip(keys, vectors):
                            self.cache[k] = np.asarray(v, dtype=np.float32)
            except Exception:
                self.cache = {}

    def _key(self, t: str, text: Optional[str] = None, is_query: bool = False) -> str:
        prefix_tag = "q:" if is_query else "p:"
        if isinstance(self, str) and text is not None:
            return hashlib.sha1(f"{self}\0{prefix_tag}{text}".encode("utf-8")).hexdigest()
        if text is not None:
            return hashlib.sha1(f"{t}\0{prefix_tag}{text}".encode("utf-8")).hexdigest()
        m = getattr(self, "model_name", str(self))
        return hashlib.sha1(f"{m}\0{prefix_tag}{t}".encode("utf-8")).hexdigest()

    def encode(self, texts: List[str], is_query: bool = False, **kwargs) -> np.ndarray:
        if not texts:
            dim = next(iter(self.cache.values())).shape[0] if self.cache else 0
            return np.empty((0, dim), dtype=np.float32)

        keys = [self._key(t, is_query=is_query) for t in texts]
        missing = {k: t for k, t in zip(keys, texts) if k not in self.cache}
        if missing:
            try:
                vecs = self.inner.encode(list(missing.values()), is_query=is_query, **kwargs)
            except TypeError:
                vecs = self.inner.encode(list(missing.values()), **kwargs)
            for k, v in zip(missing.keys(), vecs):
                self.cache[k] = np.asarray(v, dtype=np.float32)
            version = getattr(self, "CACHE_VERSION", CACHE_VERSION)
            keys_arr = np.array(list(self.cache.keys()))
            vecs_arr = np.array(list(self.cache.values()), dtype=np.float32)
            np.savez(self.path, keys=keys_arr, vectors=vecs_arr, version=np.array(version))
        return _normalise(np.vstack([self.cache[k] for k in keys]))


def get_embedder(name: str = "minilm", use_cache: bool = True):
    if name == "minilm":
        emb = MiniLMEmbedder()
    elif name == "bge-m3":
        emb = OllamaEmbedder(base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"))
    elif name == "me5":
        emb = ME5Embedder()
    else:
        raise ValueError(f"Unknown embedder '{name}'. Choose from: {', '.join(EMBEDDERS)}")
    return CachedEmbedder(emb, name) if use_cache else emb


def default_min_relevance(name: str) -> float:
    return EMBEDDERS.get(name, {}).get("min_relevance", 0.20)
