"""Adapter for Mem0's local `Memory` class."""

from typing import Dict, List

from mem0 import Memory

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


class Mem0Adapter(MemorySystem):
    def __init__(self):
        mem0_config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": config.GENERATION_MODEL,
                    "temperature": config.TEMPERATURE,
                },
            },
        }
        self._mem0 = Memory.from_config(mem0_config)
        self._user_id = config.USER_ID
        self._id_map: Dict[str, str] = {}   # mem0 memory id -> harness memory_id

    @property
    def name(self) -> str:
        return "mem0"

    def add(self, memory: MemoryRecord) -> None:
        response = self._mem0.add(
            memory.text,
            user_id=self._user_id,
            metadata={"harness_memory_id": memory.memory_id},
        )
        for entry in self._extract_entries(response):
            self._id_map[entry["id"]] = memory.memory_id

    def retrieve(self, query: str, k: int) -> RetrievalResult:
        response = self._mem0.search(query, top_k=k, filters={"user_id": self._user_id})

        memory_ids: List[str] = []
        texts: List[str] = []
        scores: List[float] = []
        for entry in self._extract_entries(response):
            mem0_id = entry["id"]
            memory_ids.append(self._id_map.get(mem0_id, mem0_id))
            texts.append(entry.get("memory", ""))
            scores.append(float(entry.get("score", 0.0)))

        return RetrievalResult(memory_ids=memory_ids, texts=texts, scores=scores)

    def reset(self) -> None:
        self._mem0.delete_all(user_id=self._user_id)
        self._id_map = {}

    @staticmethod
    def _extract_entries(response) -> list:
        if isinstance(response, dict):
            return response.get("results", []) or []
        return response or []
