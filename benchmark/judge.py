"""Answer-correctness scoring: llm_judge (primary) plus offline exact_match/token_f1 helpers."""

import re
import string

try:
    # Normal package import
    from .. import config
    from .llm_client import call_openai_with_retry
except (ImportError, ValueError):
    # Allow running as a plain script/module without the parent package
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    import config
    from benchmark.llm_client import call_openai_with_retry

from openai import OpenAI

JUDGE_PROMPT = """You are grading a home-assistant AI's answer for a memory-recall benchmark. \
Compare the generated answer to the expected (reference) answer. Minor differences in \
phrasing, wording, or extra context are fine as long as the core fact is correct and \
nothing is contradicted or missing. Reply with exactly one word: "correct" or "incorrect".

Question: {question}
Expected answer: {expected}
Generated answer: {generated}

Verdict:"""


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def exact_match(expected: str, generated: str) -> bool:
    """Normalized string equality."""
    return _normalize(expected) == _normalize(generated)


def token_f1(expected: str, generated: str) -> float:
    """Token-overlap F1 between normalized expected/generated answers."""
    expected_tokens = _normalize(expected).split()
    generated_tokens = _normalize(generated).split()
    if not expected_tokens or not generated_tokens:
        return 0.0

    remaining = {}
    for tok in generated_tokens:
        remaining[tok] = remaining.get(tok, 0) + 1

    overlap = 0
    for tok in expected_tokens:
        if remaining.get(tok, 0) > 0:
            overlap += 1
            remaining[tok] -= 1

    if overlap == 0:
        return 0.0
    precision = overlap / len(generated_tokens)
    recall = overlap / len(expected_tokens)
    return 2 * precision * recall / (precision + recall)


def llm_judge(client: OpenAI, question: str, expected_answer: str, generated_answer: str) -> bool:
    """Ask config.JUDGE_MODEL whether the generated answer is semantically correct."""
    prompt = JUDGE_PROMPT.format(question=question, expected=expected_answer, generated=generated_answer)
    response = call_openai_with_retry(
        client,
        model=config.JUDGE_MODEL,
        temperature=config.TEMPERATURE,
        messages=[{"role": "user", "content": prompt}],
    )
    verdict = response.choices[0].message.content.strip().lower()
    # Check "incorrect" first since it contains "correct" as a substring.
    if "incorrect" in verdict:
        return False
    return "correct" in verdict
