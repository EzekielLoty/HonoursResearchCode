"""Drives a MemorySystem over the dataset: ingest memories, then retrieve + generate an answer per question."""

import json
import time
from dataclasses import dataclass, field
from typing import List, Optional

try:
    # Normal package import
    from .. import config
    from .interface import MemorySystem, MemoryRecord
    from .llm_client import call_openai_with_retry
except (ImportError, ValueError):
    # Allow running as a plain script/module without the parent package
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    import config
    from benchmark.interface import MemorySystem, MemoryRecord
    from benchmark.llm_client import call_openai_with_retry

from openai import OpenAI

GENERATION_PROMPT = """You are a home assistant. Answer the user's question using ONLY the memories \
listed below. Be concise — answer in a single short sentence or phrase. If the \
memories don't contain the answer, say you don't know.

Memories:
{memories}

Question: {question}
Answer:"""


@dataclass
class QuestionResult:
    """Raw output for one question, before scoring/logging."""

    question_id: str
    category: str
    question: str
    expected_answer: str
    relevant_memory_ids: List[str]
    retrieved_memory_ids: List[str]
    generated_answer: str
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    superseded_memory_ids: Optional[List[str]] = field(default=None)


def load_dataset(path=None) -> List[dict]:
    """Load the JSONL dataset, sorted by timestamp."""
    path = path or config.DATASET_PATH
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records.sort(key=lambda r: r["timestamp"])
    return records


def generate_answer(client: OpenAI, question: str, retrieved_texts: List[str]) -> str:
    """Shared generation step, identical for every memory system."""
    memories_block = "\n".join(f"- {t}" for t in retrieved_texts) or "(no memories retrieved)"
    prompt = GENERATION_PROMPT.format(memories=memories_block, question=question)
    response = call_openai_with_retry(
        client,
        model=config.GENERATION_MODEL,
        temperature=config.TEMPERATURE,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def run(system: MemorySystem, dataset: Optional[List[dict]] = None,
        client: Optional[OpenAI] = None) -> List[QuestionResult]:
    """Ingest all memories in timestamp order, then retrieve + generate for every question."""
    dataset = dataset if dataset is not None else load_dataset()
    client = client or OpenAI()  # reads OPENAI_API_KEY from the environment

    system.reset()
    results = []

    for record in dataset:
        if record["type"] == "memory":
            system.add(MemoryRecord(
                memory_id=record["memory_id"],
                user_id=record["user_id"],
                timestamp=record["timestamp"],
                text=record["text"],
                category=record["category"],
                subcategory=record.get("subcategory"),
                session_id=record.get("session_id"),
            ))

        elif record["type"] == "question":
            t0 = time.perf_counter()
            retrieval = system.retrieve(record["text"], k=config.RETRIEVAL_K)
            t1 = time.perf_counter()

            context_texts = retrieval.texts[:config.MAX_CONTEXT_MEMORIES]
            answer = generate_answer(client, record["text"], context_texts)
            t2 = time.perf_counter()

            results.append(QuestionResult(
                question_id=record["question_id"],
                category=record["category"],
                question=record["text"],
                expected_answer=record["expected_answer"],
                relevant_memory_ids=record["relevant_memory_ids"],
                retrieved_memory_ids=retrieval.memory_ids,
                generated_answer=answer,
                retrieval_latency_ms=(t1 - t0) * 1000,
                generation_latency_ms=(t2 - t1) * 1000,
                total_latency_ms=(t2 - t0) * 1000,
                superseded_memory_ids=record.get("superseded_memory_ids"),
            ))

    return results


if __name__ == "__main__":
    from systems.vector_baseline import VectorBaseline

    system = VectorBaseline()
    results = run(system)

    for r in results:
        print(f"[{r.category}] {r.question_id} {r.question!r} -> {r.generated_answer!r}")
    print(f"\n{len(results)} questions answered by {system.name}")
