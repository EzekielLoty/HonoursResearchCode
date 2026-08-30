# Memory Architecture Benchmark

A research repository for my honours research a benchmark comparing long-term memory architectures for LLM-based personal/home assistants. The memory system is the only independent variable: one shared dataset, generation model, judge, and retrieval K are held constant across every system under test.

## Layout

```
config.py                     # all shared constants (env-overridable)
benchmark/
  interface.py                 # MemorySystem ABC + MemoryRecord/RetrievalResult
  dataset_generator.py         # builds data/home_assistant_v1.jsonl
  dataset_generator_v2.py      # builds data/home_assistant_v2_stress.jsonl
  runner.py                    # ingests memories, runs questions, generates answers
  metrics.py                   # Recall@K / Precision@K / MRR
  judge.py                     # LLM-as-judge answer scoring
  llm_client.py                # shared OpenAI pacing/retry wrapper
  logger.py                    # writes per-question + summary CSVs
systems/
  vector_baseline.py           # local embeddings, cosine top-k
  mem0_adapter.py
  amem_adapter.py
  mempalace_adapter.py
  graphiti_adapter.py
  letta_adapter.py
  langchain_memory.py          # LangChain memory techniques (buffer/window/summary/entity/episodic)
data/                          # generated datasets
results/                       # CSV outputs
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the repo root (gitignored):

```
OPENAI_API_KEY=sk-...
```

Some systems need external services running via Docker:

```bash
# systems/graphiti_adapter.py
docker run -d --name falkordb-benchmark -p 6379:6379 falkordb/falkordb:latest

# systems/letta_adapter.py
docker run -d --name letta-benchmark -v letta_pgdata:/var/lib/postgresql/data \
    -p 8283:8283 -e OPENAI_API_KEY="$OPENAI_API_KEY" letta/letta:latest
```

## Configuring a run

Everything that must stay constant across systems lives in `config.py`, read from env vars with sane defaults — override any of them without touching code:

| Variable | Default | Controls |
|---|---|---|
| `DATASET_VERSION` | `home_assistant_v1` | which `data/<version>.jsonl` gets loaded |
| `RETRIEVAL_K` | `5` | top-k memories retrieved per question |
| `GENERATION_MODEL` | `gpt-4o-mini` | shared answer-generation model |
| `JUDGE_MODEL` | `gpt-4o-mini` | LLM-as-judge model |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | local embedder for vector_baseline |
| `REQUEST_DELAY_SECONDS` | `6.5` | min gap between OpenAI calls (rate-limit pacing) |
| `MAX_RETRIES` | `5` | retries on a rate-limit error before giving up |
| `TEMPORAL_RECALL_INCLUDES_SUPERSEDED` | `True` | set in code; whether an invalidated old memory still counts toward recall |

Example — run the harder v2 dataset at k=3:

```bash
export DATASET_VERSION=home_assistant_v2_stress
export RETRIEVAL_K=3
```

## Generating a dataset

```bash
python -m benchmark.dataset_generator                                              # v1
DATASET_VERSION=home_assistant_v2_stress RETRIEVAL_K=3 python -m benchmark.dataset_generator_v2   # v2
```

## Running a system

```python
from benchmark.logger import run_and_log
from systems.vector_baseline import VectorBaseline   # swap for any other system

per_question_csv, summary_csv = run_and_log(VectorBaseline())
```

Writes `results/<system>_<timestamp>_per_question.csv` and `..._summary.csv`.

## Adding a new memory system

Implement `benchmark.interface.MemorySystem`: a `name` property and `add(memory)` / `retrieve(query, k)` / `reset()`. See any file in `systems/` for a reference implementation. Never generate answers inside an adapter — `benchmark/runner.py` owns the one shared generation step so every system is compared fairly.
