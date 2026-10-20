"""Minimal TypeSafe Jev client (System One API) for behaviour classification.

Endpoint: ``POST https://api.typesafe.ai/v1/systemone``
Auth:     ``Authorization: Bearer $TYPESAFE_API_KEY``
Pricing:  jev-1.14.0 is $0.042 per 1M input tokens; output tokens are free.

The client is intentionally small and dependency-light (``requests`` only) so it
can be used from the log tools without the OpenAI/TypeSafe SDKs.
"""

from __future__ import annotations

import os
import random
import time
from typing import Callable, Optional

import requests

DEFAULT_BASE_URL = "https://api.typesafe.ai"
DEFAULT_MODEL = "jev-1.14.0"
INPUT_PRICE_PER_TOKEN = 0.042 / 1_000_000  # USD, jev-1.14.0


class JevError(RuntimeError):
    """A non-retryable (or exhausted) TypeSafe API error."""


class JevClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
        max_retries: int = 5,
        backoff: float = 1.6,
        long_context: bool = False,
        session: Optional[requests.Session] = None,
    ):
        self.api_key = api_key or os.getenv("TYPESAFE_API_KEY")
        if not self.api_key:
            raise JevError("TYPESAFE_API_KEY is not set (or pass api_key=...).")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self.long_context = long_context
        self.session = session or requests.Session()
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0

    def classify(self, state, questions: dict) -> dict:
        payload = {"model": self.model, "state": state, "questions": questions}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.long_context:
            headers["TypeSafe-Danger-Zone-Long-Context"] = "true"

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(
                    f"{self.base_url}/v1/systemone",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last_error = exc
                self._sleep(attempt)
                continue

            if response.status_code == 200:
                data = response.json()
                usage = data.get("usage") or {}
                self.input_tokens += int(usage.get("input_tokens", 0) or 0)
                self.output_tokens += int(usage.get("output_tokens", 0) or 0)
                self.calls += 1
                return data

            if response.status_code in (429, 500, 502, 503, 504):
                last_error = JevError(f"HTTP {response.status_code}: {response.text[:300]}")
                retry_after = response.headers.get("Retry-After")
                self._sleep(attempt, retry_after)
                continue

            raise JevError(f"HTTP {response.status_code}: {response.text[:500]}")

        raise JevError(f"Jev request failed after {self.max_retries} retries: {last_error}")

    def _sleep(self, attempt: int, retry_after: Optional[str] = None) -> None:
        delay = self.backoff ** attempt + random.random() * 0.25
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except ValueError:
                pass
        time.sleep(min(delay, 60.0))

    @property
    def cost_usd(self) -> float:
        """Input-only pricing: output tokens are free for Jev."""
        return self.input_tokens * INPUT_PRICE_PER_TOKEN

    def usage_summary(self) -> dict:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


class MockJevClient:
    """Offline client that delegates to a callable ``(state, questions) -> response``."""

    def __init__(self, classifier: Callable[[dict, dict], dict]):
        self.classifier = classifier
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def classify(self, state, questions: dict) -> dict:
        self.calls += 1
        response = self.classifier(state, questions)
        usage = response.get("usage") or {}
        self.input_tokens += int(usage.get("input_tokens", 0) or 0)
        self.output_tokens += int(usage.get("output_tokens", 0) or 0)
        return response

    @property
    def cost_usd(self) -> float:
        return self.input_tokens * INPUT_PRICE_PER_TOKEN

    def usage_summary(self) -> dict:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }
