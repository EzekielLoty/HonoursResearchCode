"""Best-effort adapter for MemPalace — built for mining files, not per-fact storage, so this uses its low-level drawer/search primitives."""

import shutil
import tempfile
from typing import Dict, List

from mempalace.miner import add_drawer, make_drawer_id_from_chunk
from mempalace.palace import get_collection
from mempalace.searcher import search_memories

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

_AGENT_NAME = "benchmark_harness"


class MemPalaceAdapter(MemorySystem):
    def __init__(self):
        self._wing = config.USER_ID
        self._palace_dir = None
        self._collection = None
        self._id_map: Dict[str, str] = {}   # drawer_id -> harness memory_id
        self._open_new_palace()

    @property
    def name(self) -> str:
        return "mempalace"

    def _open_new_palace(self) -> None:
        self._palace_dir = tempfile.mkdtemp(prefix="mempalace_harness_")
        self._collection = get_collection(self._palace_dir, create=True)

    def add(self, memory: MemoryRecord) -> None:
        source_file = f"harness/{memory.memory_id}.txt"
        add_drawer(
            self._collection,
            wing=self._wing,
            room=memory.category,
            content=memory.text,
            source_file=source_file,
            chunk_index=0,
            agent=_AGENT_NAME,
        )
        drawer_id = make_drawer_id_from_chunk(self._wing, memory.category, source_file, 0)
        self._id_map[drawer_id] = memory.memory_id

    def retrieve(self, query: str, k: int) -> RetrievalResult:
        response = search_memories(query, self._palace_dir, wing=self._wing, n_results=k)
        hits = response.get("results", []) or []

        memory_ids: List[str] = []
        texts: List[str] = []
        scores: List[float] = []
        for hit in hits:
            drawer_id = hit["drawer_id"]
            memory_ids.append(self._id_map.get(drawer_id, drawer_id))
            texts.append(hit.get("text", ""))
            scores.append(float(hit.get("similarity", 0.0)))

        return RetrievalResult(memory_ids=memory_ids, texts=texts, scores=scores)

    def reset(self) -> None:
        old_dir = self._palace_dir
        self._open_new_palace()
        self._id_map = {}
        if old_dir:
            shutil.rmtree(old_dir, ignore_errors=True)
