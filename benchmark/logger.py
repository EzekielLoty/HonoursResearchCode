"""Scores a run's raw output and writes a per-question CSV + a per-category summary CSV to config.RESULTS_DIR."""

import csv
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

try:
    # Normal package import
    from .. import config
    from . import metrics
    from . import judge as judge_module
    from .interface import MemorySystem
    from .runner import QuestionResult
except (ImportError, ValueError):
    # Allow running as a plain script/module without the parent package
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    import config
    from benchmark import metrics
    from benchmark import judge as judge_module
    from benchmark.interface import MemorySystem
    from benchmark.runner import QuestionResult

from openai import OpenAI

PER_QUESTION_FIELDS = [
    "memory_system", "question_id", "category", "question", "expected_answer",
    "generated_answer", "relevant_memory_ids", "retrieved_memory_ids",
    "recall_at_1", "recall_at_3", "recall_at_5", "precision_at_3", "precision_at_5",
    "mrr", "answer_correct", "retrieval_latency_ms", "generation_latency_ms",
    "total_latency_ms",
]

SUMMARY_FIELDS = ["Memory System", "Category", "Recall@5", "MRR", "Answer Accuracy", "Avg Latency"]


def _join_ids(ids: List[str]) -> str:
    return ";".join(ids)


def score_question(system_name: str, result: QuestionResult, client: OpenAI) -> dict:
    """Combine one QuestionResult with retrieval metrics and the judge's verdict into one CSV row."""
    question_dict = asdict(result)
    retrieval_scores = metrics.score_retrieval(result.retrieved_memory_ids, question_dict)
    correct = judge_module.llm_judge(client, result.question, result.expected_answer, result.generated_answer)

    return {
        "memory_system": system_name,
        "question_id": result.question_id,
        "category": result.category,
        "question": result.question,
        "expected_answer": result.expected_answer,
        "generated_answer": result.generated_answer,
        "relevant_memory_ids": _join_ids(result.relevant_memory_ids),
        "retrieved_memory_ids": _join_ids(result.retrieved_memory_ids),
        **retrieval_scores,
        "answer_correct": correct,
        "retrieval_latency_ms": result.retrieval_latency_ms,
        "generation_latency_ms": result.generation_latency_ms,
        "total_latency_ms": result.total_latency_ms,
    }


def write_results_csv(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PER_QUESTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows: List[dict]) -> List[dict]:
    """Group scored rows by (memory_system, category) and average the metrics."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["memory_system"], row["category"])].append(row)

    summary = []
    for (system_name, category), group in sorted(groups.items()):
        n = len(group)
        summary.append({
            "Memory System": system_name,
            "Category": category,
            "Recall@5": sum(r["recall_at_5"] for r in group) / n,
            "MRR": sum(r["mrr"] for r in group) / n,
            "Answer Accuracy": sum(r["answer_correct"] for r in group) / n,
            "Avg Latency": sum(r["total_latency_ms"] for r in group) / n,
        })
    return summary


def write_summary_csv(summary_rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({
                **row,
                "Recall@5": f"{row['Recall@5']:.4f}",
                "MRR": f"{row['MRR']:.4f}",
                "Answer Accuracy": f"{row['Answer Accuracy']:.4f}",
                "Avg Latency": f"{row['Avg Latency']:.1f}",
            })


def run_and_log(system: MemorySystem, dataset: Optional[List[dict]] = None,
                 client: Optional[OpenAI] = None,
                 results_dir: Optional[Path] = None) -> Tuple[Path, Path]:
    """Run `system` over the dataset, score every question, write both CSVs."""
    from .runner import run

    client = client or OpenAI()
    raw_results = run(system, dataset=dataset, client=client)

    rows = [score_question(system.name, r, client) for r in raw_results]

    results_dir = results_dir or config.RESULTS_DIR
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    per_question_path = results_dir / f"{system.name}_{timestamp}_per_question.csv"
    summary_path = results_dir / f"{system.name}_{timestamp}_summary.csv"

    write_results_csv(rows, per_question_path)
    write_summary_csv(build_summary(rows), summary_path)

    return per_question_path, summary_path


if __name__ == "__main__":
    from systems.vector_baseline import VectorBaseline

    system = VectorBaseline()
    per_question_path, summary_path = run_and_log(system)

    print(f"Wrote {per_question_path}")
    print(f"Wrote {summary_path}")
