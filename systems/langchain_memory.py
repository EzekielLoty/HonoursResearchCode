"""LangChain/LangGraph memory technique adapters (langchain-classic, verified against the installed package)."""

from typing import Dict, List

from langchain_classic.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationEntityMemory,
    ConversationSummaryMemory,
    VectorStoreRetrieverMemory,
)
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

try:
    from .. import config
    from ..benchmark.interface import MemorySystem, MemoryRecord, RetrievalResult
except (ImportError, ValueError):
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    import config
    from benchmark.interface import MemorySystem, MemoryRecord, RetrievalResult


class _BlobMemoryAdapter(MemorySystem):
    """Base for buffer/window/summary/entity memory: these hold one running
    text blob, not discrete searchable records, so retrieve() always returns
    that single blob as one result rather than a ranked list."""

    def __init__(self, lc_memory):
        self._mem = lc_memory

    def add(self, memory: MemoryRecord) -> None:
        self._mem.save_context({"input": memory.text}, {"output": "Noted."})

    def retrieve(self, query: str, k: int) -> RetrievalResult:
        variables = self._mem.load_memory_variables({"input": query})
        blob = "\n".join(str(v) for v in variables.values() if isinstance(v, str) and v)
        if not blob:
            return RetrievalResult()
        return RetrievalResult(memory_ids=["current_state"], texts=[blob], scores=[1.0])

    def reset(self) -> None:
        self._mem.clear()


class LangChainBufferMemory(_BlobMemoryAdapter):
    """Keeps the full conversation verbatim, unbounded."""

    def __init__(self):
        super().__init__(ConversationBufferMemory())

    @property
    def name(self) -> str:
        return "langchain_buffer"


class LangChainWindowMemory(_BlobMemoryAdapter):
    """Keeps only the last k turns (sliding window)."""

    def __init__(self, k: int = 5):
        super().__init__(ConversationBufferWindowMemory(k=k))

    @property
    def name(self) -> str:
        return "langchain_window"


class LangChainSummaryMemory(_BlobMemoryAdapter):
    """Collapses older turns into a running LLM-generated summary."""

    def __init__(self):
        llm = ChatOpenAI(model=config.GENERATION_MODEL, temperature=config.TEMPERATURE)
        super().__init__(ConversationSummaryMemory(llm=llm))

    @property
    def name(self) -> str:
        return "langchain_summary"


class LangChainEntityMemory(_BlobMemoryAdapter):
    """Extracts and tracks per-entity facts via an LLM."""

    def __init__(self):
        llm = ChatOpenAI(model=config.GENERATION_MODEL, temperature=config.TEMPERATURE)
        super().__init__(ConversationEntityMemory(llm=llm))

    @property
    def name(self) -> str:
        return "langchain_entity"


class LangChainEpisodicMemory(MemorySystem):
    """Each memory is a discrete, independently embedded/retrievable episode
    (VectorStoreRetrieverMemory over an in-memory vector store) — the one
    LangChain memory type here with real ranked semantic retrieval."""

    def __init__(self):
        self._store = InMemoryVectorStore(OpenAIEmbeddings(model="text-embedding-3-small"))
        retriever = self._store.as_retriever(search_kwargs={"k": config.RETRIEVAL_K})
        self._mem = VectorStoreRetrieverMemory(retriever=retriever, return_docs=True)
        self._id_map: Dict[str, str] = {}

    @property
    def name(self) -> str:
        return "langchain_episodic"

    def add(self, memory: MemoryRecord) -> None:
        ids = self._store.add_texts([memory.text], metadatas=[{"harness_memory_id": memory.memory_id}])
        self._id_map[ids[0]] = memory.memory_id

    def retrieve(self, query: str, k: int) -> RetrievalResult:
        docs = self._mem.retriever.invoke(query)[:k]
        memory_ids: List[str] = []
        texts: List[str] = []
        scores: List[float] = []
        for rank, doc in enumerate(docs):
            harness_id = doc.metadata.get("harness_memory_id", doc.id)
            memory_ids.append(self._id_map.get(doc.id, harness_id))
            texts.append(doc.page_content)
            scores.append(round(1.0 / (rank + 1), 4))
        return RetrievalResult(memory_ids=memory_ids, texts=texts, scores=scores)

    def reset(self) -> None:
        self._store = InMemoryVectorStore(OpenAIEmbeddings(model="text-embedding-3-small"))
        retriever = self._store.as_retriever(search_kwargs={"k": config.RETRIEVAL_K})
        self._mem = VectorStoreRetrieverMemory(retriever=retriever, return_docs=True)
        self._id_map = {}
