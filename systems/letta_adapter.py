"""Adapter for Letta's archival memory (passages API) on a self-hosted server. Only passages.*, never agents.messages.* (no chat/reasoning).

Requires: docker run -d --name letta-benchmark -v letta_pgdata:/var/lib/postgresql/data -p 8283:8283 -e OPENAI_API_KEY="$OPENAI_API_KEY" letta/letta:latest
"""

from typing import Dict, List

from letta_client import Letta

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

_LETTA_BASE_URL = "http://localhost:8283"


class LettaAdapter(MemorySystem):
    def __init__(self):
        self._client = Letta(base_url=_LETTA_BASE_URL)
        self._agent_id = self._create_agent()
        self._id_map: Dict[str, str] = {}

    def _create_agent(self) -> str:
        # embedding= is required — without it search silently returns nothing for full-sentence queries
        agent = self._client.agents.create(
            name=f"benchmark_{config.USER_ID}",
            model=f"openai/{config.GENERATION_MODEL}",
            embedding="openai/text-embedding-3-small",
        )
        return agent.id

    @property
    def name(self) -> str:
        return "letta"

    def add(self, memory: MemoryRecord) -> None:
        passages = self._client.agents.passages.create(self._agent_id, text=memory.text)
        for passage in passages:
            self._id_map[passage.id] = memory.memory_id

    def retrieve(self, query: str, k: int) -> RetrievalResult:
        response = self._client.agents.passages.search(self._agent_id, query=query, top_k=k)

        memory_ids: List[str] = []
        texts: List[str] = []
        scores: List[float] = []
        for rank, item in enumerate(response.results):
            memory_ids.append(self._id_map.get(item.id, item.id))
            texts.append(item.content)
            scores.append(round(1.0 / (rank + 1), 4))   # no score field on Result

        return RetrievalResult(memory_ids=memory_ids, texts=texts, scores=scores)

    def reset(self) -> None:
        self._client.agents.delete(self._agent_id)
        self._agent_id = self._create_agent()
        self._id_map = {}
