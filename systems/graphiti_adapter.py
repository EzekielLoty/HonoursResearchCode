"""Adapter for Zep's Graphiti (temporal knowledge graph), backed by FalkorDB. Retrieves via node search, not edge search.

Requires: docker run -d --name falkordb-benchmark -p 6379:6379 falkordb/falkordb:latest
"""

import asyncio
from datetime import datetime, timezone
from typing import Dict, List

from graphiti_core import Graphiti
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF

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


class GraphitiAdapter(MemorySystem):
    def __init__(self):
        self._loop = asyncio.new_event_loop()
        driver = FalkorDriver(host="localhost", port=6379)
        self._graphiti = Graphiti(graph_driver=driver)
        self._run(self._graphiti.build_indices_and_constraints())
        self._group_id = config.USER_ID
        self._episode_ids: List[str] = []
        self._id_map: Dict[str, List[str]] = {}   # node uuid -> [harness memory_ids]

    def _run(self, coro):
        return self._loop.run_until_complete(coro)

    @property
    def name(self) -> str:
        return "graphiti"

    def add(self, memory: MemoryRecord) -> None:
        reference_time = datetime.fromisoformat(memory.timestamp).replace(tzinfo=timezone.utc)
        result = self._run(self._graphiti.add_episode(
            name=memory.memory_id,
            episode_body=memory.text,
            source=EpisodeType.text,
            source_description=memory.category,
            reference_time=reference_time,
            group_id=self._group_id,
        ))
        episode_uuid = result.episode.uuid
        self._episode_ids.append(episode_uuid)

        touched = self._run(self._graphiti.get_nodes_and_edges_by_episode([episode_uuid]))
        for node in touched.nodes:
            self._id_map.setdefault(node.uuid, []).append(memory.memory_id)

    def retrieve(self, query: str, k: int) -> RetrievalResult:
        node_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        node_config.limit = k
        result = self._run(self._graphiti._search(query=query, config=node_config, group_ids=[self._group_id]))

        memory_ids: List[str] = []
        texts: List[str] = []
        scores: List[float] = []
        for rank, node in enumerate(result.nodes):
            score = round(1.0 / (rank + 1), 4)
            harness_ids = self._id_map.get(node.uuid, [node.uuid])
            for harness_id in harness_ids:
                memory_ids.append(harness_id)
                texts.append(node.summary)
                scores.append(score)

        return RetrievalResult(memory_ids=memory_ids, texts=texts, scores=scores)

    def reset(self) -> None:
        # sweeps by group_id, not just self._episode_ids: FalkorDB is a persistent shared service
        existing = self._run(self._graphiti.retrieve_episodes(
            reference_time=datetime.now(timezone.utc),
            last_n=100_000,
            group_ids=[self._group_id],
        ))
        for episode in existing:
            try:
                self._run(self._graphiti.remove_episode(episode.uuid))
            except Exception:
                pass
        self._episode_ids = []
        self._id_map = {}
