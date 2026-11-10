"""
Module 3: Embedding & Vector Storage.
Local-first in-memory vector storage using NumPy normalized cosine similarity
for zero cloud leakage on-device privacy.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from ..chunking.chunk_messages import ConversationChunk


class LocalVectorStore:
    """
    Manages local vector embeddings and indexing of conversational chunks.
    Ensures zero cloud leakage for private personal chat data.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 text_field: str = "attributed_text"):
        self.model_name = model_name
        self.text_field = text_field  # "attributed_text" (with timestamps) or "raw_text" (Sender: text)
        self.chunks: List[ConversationChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(self.model_name)
            except ImportError:
                self._encoder = None
        return self._encoder

    def add_chunks(self, chunks: List[ConversationChunk]):
        """Embeds and indexes a list of conversation chunks."""
        self.chunks = chunks
        texts = [getattr(c, self.text_field) for c in chunks]

        encoder = self._get_encoder()
        if encoder is None:
            raise RuntimeError(
                "sentence-transformers is required: pip install sentence-transformers"
            )
        self.embeddings = encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Performs dense cosine similarity search."""
        if not self.chunks or self.embeddings is None:
            return []

        encoder = self._get_encoder()
        if encoder is None:
            raise RuntimeError(
                "sentence-transformers is required: pip install sentence-transformers"
            )
        q_emb = encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]

        sims = np.dot(self.embeddings, q_emb)
        top_indices = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "chunk": self.chunks[idx],
                "semantic_score": float(sims[idx]),
                "index": int(idx),
            })
        return results
