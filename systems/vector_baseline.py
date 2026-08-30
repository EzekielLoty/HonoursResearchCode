"""Local embedding (all-MiniLM-L6-v2) + top-k cosine-similarity baseline."""

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    # Normal package import
    from .. import config
    from ..benchmark.interface import MemorySystem, MemoryRecord, RetrievalResult
except (ImportError, ValueError):
    # Allow running as a plain script/module without the parent package
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    import config
    from benchmark.interface import MemorySystem, MemoryRecord, RetrievalResult


class VectorBaseline(MemorySystem):
    def __init__(self):
        self._model = SentenceTransformer(config.EMBEDDING_MODEL)
        self._ids: List[str] = []
        self._texts: List[str] = []
        self._embeddings: List[np.ndarray] = []

    @property
    def name(self) -> str:
        return "vector_baseline"

    def add(self, memory: MemoryRecord) -> None:
        vec = self._model.encode(memory.text, normalize_embeddings=True)
        self._ids.append(memory.memory_id)
        self._texts.append(memory.text)
        self._embeddings.append(vec)

    def retrieve(self, query: str, k: int) -> RetrievalResult:
        if not self._embeddings:
            return RetrievalResult()

        query_vec = self._model.encode(query, normalize_embeddings=True)
        matrix = np.stack(self._embeddings)   # (N, dim), rows already unit-normalized
        scores = matrix @ query_vec           # cosine similarity == dot product of unit vectors

        k = min(k, len(self._ids))
        top_idx = np.argsort(-scores)[:k]     # descending, best match first

        return RetrievalResult(
            memory_ids=[self._ids[i] for i in top_idx],
            texts=[self._texts[i] for i in top_idx],
            scores=[float(scores[i]) for i in top_idx],
        )

    def reset(self) -> None:
        self._ids = []
        self._texts = []
        self._embeddings = []
