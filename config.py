"""Shared config: dataset, models, retrieval K, rate limits. Keys come from env/.env, never hardcoded."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

SEED = 42
USER_ID = "user_01"
BASE_DATE = "2026-01-01T08:00:00"
DATASET_VERSION = os.environ.get("DATASET_VERSION", "home_assistant_v1")

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
DATASET_PATH = DATA_DIR / f"{DATASET_VERSION}.jsonl"

# retrieval / generation
RETRIEVAL_K = int(os.environ.get("RETRIEVAL_K", 5))
TEMPERATURE = 0.0
MAX_CONTEXT_MEMORIES = 5

GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "gpt-4o-mini")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# rate-limit handling (benchmark/llm_client.py)
REQUEST_DELAY_SECONDS = float(os.environ.get("REQUEST_DELAY_SECONDS", 6.5))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", 5))
RETRY_BACKOFF_BASE_SECONDS = float(os.environ.get("RETRY_BACKOFF_BASE_SECONDS", 7.0))

# if False, superseded memories are excluded from temporal_update ground truth
TEMPORAL_RECALL_INCLUDES_SUPERSEDED = True


def as_dict():
    """Return a JSON-serialisable snapshot of the config, for logging into results."""
    return {
        "seed": SEED,
        "user_id": USER_ID,
        "base_date": BASE_DATE,
        "dataset_version": DATASET_VERSION,
        "retrieval_k": RETRIEVAL_K,
        "temperature": TEMPERATURE,
        "max_context_memories": MAX_CONTEXT_MEMORIES,
        "generation_model": GENERATION_MODEL,
        "judge_model": JUDGE_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "temporal_recall_includes_superseded": TEMPORAL_RECALL_INCLUDES_SUPERSEDED,
        "request_delay_seconds": REQUEST_DELAY_SECONDS,
        "max_retries": MAX_RETRIES,
        "retry_backoff_base_seconds": RETRY_BACKOFF_BASE_SECONDS,
    }
