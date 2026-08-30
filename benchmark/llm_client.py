"""Shared OpenAI call wrapper: rate-limit pacing + retry with backoff."""

import time

from openai import RateLimitError

try:
    # Normal package import
    from .. import config
except (ImportError, ValueError):
    # Allow running as a plain script/module without the parent package
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    import config

_last_request_time = None  # module-level: pacing shared across every caller


def call_openai_with_retry(client, **kwargs):
    """client.chat.completions.create(**kwargs), with pacing + retry."""
    global _last_request_time

    _wait_for_pacing()

    attempt = 0
    while True:
        try:
            response = client.chat.completions.create(**kwargs)
            _last_request_time = time.perf_counter()
            print("API request successful")
            return response
        except RateLimitError:
            if attempt >= config.MAX_RETRIES:
                print(f"Rate limit hit, exhausted all {config.MAX_RETRIES} retries — giving up.")
                raise
            backoff = config.RETRY_BACKOFF_BASE_SECONDS * (2 ** attempt)
            print(f"Rate limit hit, retrying in {backoff:.0f} seconds...")
            time.sleep(backoff)
            attempt += 1


def _wait_for_pacing():
    """Sleep so this call starts >= REQUEST_DELAY_SECONDS after the last successful one."""
    if _last_request_time is None:
        return
    elapsed = time.perf_counter() - _last_request_time
    remaining = config.REQUEST_DELAY_SECONDS - elapsed
    if remaining > 0:
        time.sleep(remaining)
