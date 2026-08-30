"""MemorySystem interface (add/retrieve/reset) and the MemoryRecord/RetrievalResult data shapes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class MemoryRecord:
    """One ingested memory fact."""

    memory_id: str
    user_id: str
    timestamp: str
    text: str
    category: str
    subcategory: Optional[str] = None
    session_id: Optional[int] = None


@dataclass
class RetrievalResult:
    """Ranked retrieval output; index 0 is the top match. Parallel lists."""

    memory_ids: List[str] = field(default_factory=list)
    texts: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)

    def __post_init__(self):
        lengths = {len(self.memory_ids), len(self.texts), len(self.scores)}
        if len(lengths) > 1:
            raise ValueError(
                f"RetrievalResult fields must be the same length, got "
                f"memory_ids={len(self.memory_ids)}, texts={len(self.texts)}, "
                f"scores={len(self.scores)}"
            )

    def __len__(self) -> int:
        return len(self.memory_ids)


class MemorySystem(ABC):
    """Interface every memory architecture adapter implements. Storage/retrieval only — no generation."""

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def add(self, memory: MemoryRecord) -> None:
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, query: str, k: int) -> RetrievalResult:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError
