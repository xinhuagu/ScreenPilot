"""Shared Anthropic client with retry + backoff for rate limits.

All modules that call Claude API should use get_client() instead of
creating anthropic.Anthropic() directly.

Retry policy (same pattern as AceClaw):
  429 rate limit → exponential backoff, up to 3 retries
  Other errors   → no retry, raise immediately
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

_client = None

MAX_RETRIES = 3
BASE_DELAY = 2.0  # seconds
MAX_DELAY = 30.0


def get_client():
    """Get or create a shared Anthropic client."""
    global _client
    if _client is not None:
        return _client

    try:
        import anthropic
    except ImportError:
        raise RuntimeError("Install LLM extra: pip install gazefy[llm]")

    from gazefy.llm.credentials import get_api_key

    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("No API key found. Set ANTHROPIC_API_KEY or log in to Claude Code.")

    _client = anthropic.Anthropic(api_key=api_key)
    return _client


def call_with_retry(create_fn, retries: int = MAX_RETRIES) -> object:
    """Call an Anthropic API function with retry on 429 rate limit.

    Args:
        create_fn: A callable that makes the API request
                   (e.g. lambda: client.messages.create(...))
        retries: Max retry attempts.

    Returns:
        The API response.
    """
    import anthropic

    delay = BASE_DELAY
    for attempt in range(retries + 1):
        try:
            return create_fn()
        except anthropic.RateLimitError:
            if attempt >= retries:
                raise
            logger.warning(
                "Rate limited (429). Retry %d/%d in %.1fs",
                attempt + 1,
                retries,
                delay,
            )
            time.sleep(delay)
            delay = min(delay * 2, MAX_DELAY)
        except anthropic.APIStatusError as e:
            if e.status_code == 529:  # Overloaded
                if attempt >= retries:
                    raise
                logger.warning("API overloaded (529). Retry in %.1fs", delay)
                time.sleep(delay)
                delay = min(delay * 2, MAX_DELAY)
            else:
                raise
