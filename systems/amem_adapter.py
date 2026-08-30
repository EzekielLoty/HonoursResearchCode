"""Adapter for A-MEM's local `AgenticMemorySystem` (Zettelkasten-style linked notes)."""

from datetime import datetime
from typing import Dict, List

from agentic_memory.memory_system import AgenticMemorySystem

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


class AMemAdapter(MemorySystem):
    def __init__(self):
        self._amem = AgenticMemorySystem(
            model_name=config.EMBEDDING_MODEL,
            llm_backend="openai",
            llm_model=config.GENERATION_MODEL,
        )
        self._id_map: Dict[str, str] = {}

    @property
    def name(self) -> str:
        return "amem"

    def add(self, memory: MemoryRecord) -> None:
        amem_time = datetime.fromisoformat(memory.timestamp).strftime("%Y%m%d%H%M")
        amem_id = self._amem.add_note(memory.text, time=amem_time)
        self._id_map[amem_id] = memory.memory_id

    def retrieve(self, query: str, k: int) -> RetrievalResult:
        hits = self._amem.search_agentic(query, k=k)

        memory_ids: List[str] = []
        texts: List[str] = []
        scores: List[float] = []
        for hit in hits:
            amem_id = hit["id"]
            memory_ids.append(self._id_map.get(amem_id, amem_id))
            texts.append(hit.get("content", ""))
            scores.append(float(hit.get("score", 0.0)))

        return RetrievalResult(memory_ids=memory_ids, texts=texts, scores=scores)

    def reset(self) -> None:
        for amem_id in list(self._id_map.keys()):
            self._amem.delete(amem_id)
        self._id_map = {}
