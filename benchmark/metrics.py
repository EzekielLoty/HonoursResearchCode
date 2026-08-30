"""Retrieval metrics (Recall@K, Precision@K, MRR) scored against ground-truth memory ids. Answer scoring is in judge.py."""

from collections import defaultdict
from typing import Dict, Iterable, List

try:
    # Normal package import
    from .. import config
except (ImportError, ValueError):
    # Allow running as a plain script/module without the parent package
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    import config


def relevant_ids_for_scoring(question: dict) -> List[str]:
    """Ground-truth relevant ids, respecting config.TEMPORAL_RECALL_INCLUDES_SUPERSEDED."""
    relevant = list(question["relevant_memory_ids"])
    if question["category"] == "temporal_update" and not config.TEMPORAL_RECALL_INCLUDES_SUPERSEDED:
        superseded = set(question.get("superseded_memory_ids", []))
        relevant = [mid for mid in relevant if mid not in superseded]
    return relevant


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Fraction of relevant ids found within the top-k retrieved ids."""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hits = len(top_k & set(relevant_ids))
    return hits / len(relevant_ids)


def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Fraction of the top-k retrieved ids that are actually relevant."""
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = len(set(top_k) & set(relevant_ids))
    return hits / k


def mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Reciprocal rank of the first relevant id in the retrieved list (best
    match ranked first = 1.0); 0.0 if no relevant id was retrieved at all."""
    relevant_set = set(relevant_ids)
    for rank, mid in enumerate(retrieved_ids, start=1):
        if mid in relevant_set:
            return 1.0 / rank
    return 0.0


def score_retrieval(retrieved_ids: List[str], question: dict) -> Dict[str, float]:
    """Score one question's retrieval; keys match logger.py's CSV columns."""
    relevant = relevant_ids_for_scoring(question)
    return {
        "recall_at_1": recall_at_k(retrieved_ids, relevant, 1),
        "recall_at_3": recall_at_k(retrieved_ids, relevant, 3),
        "recall_at_5": recall_at_k(retrieved_ids, relevant, 5),
        "precision_at_3": precision_at_k(retrieved_ids, relevant, 3),
        "precision_at_5": precision_at_k(retrieved_ids, relevant, 5),
        "mrr": mrr(retrieved_ids, relevant),
    }


def mean_by_category(rows: Iterable[dict], value_key: str, category_key: str = "category") -> Dict[str, float]:
    """Average `value_key` across `rows`, grouped by `category_key`."""
    buckets = defaultdict(list)
    for row in rows:
        buckets[row[category_key]].append(row[value_key])
    return {cat: sum(vals) / len(vals) for cat, vals in buckets.items() if vals}


def accuracy_by_category(rows: Iterable[dict], category_key: str = "category") -> Dict[str, float]:
    """Answer accuracy per category."""
    return mean_by_category(rows, value_key="answer_correct", category_key=category_key)
