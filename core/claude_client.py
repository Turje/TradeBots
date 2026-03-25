"""
Anthropic SDK wrapper with retry logic and structured logging.
Used by every agent in the TradeBots pipeline.
"""

import logging
import time
from typing import Optional

import anthropic

from config.settings import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    MAX_RETRIES,
    RETRY_BASE_DELAY,
)

# Errors that are transient and worth retrying
_RETRYABLE_ERRORS = (
    anthropic.APIConnectionError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
)

# Errors that indicate a permanent failure — fail fast, no retry
_FATAL_ERRORS = (
    anthropic.AuthenticationError,
    anthropic.BadRequestError,
)


class ClaudeClient:
    """
    Wrapper around the Anthropic SDK with:
    - Retry logic: MAX_RETRIES attempts, exponential backoff starting at RETRY_BASE_DELAY seconds
    - Structured logging: every call logs agent_name, input summary, output summary, latency
    - Simple interface for agents to call Claude
    """

    def __init__(self, agent_name: str) -> None:
        """
        agent_name: used in log messages (e.g. "scanner", "orchestrator", "strategy.technical")
        Raises EnvironmentError if ANTHROPIC_API_KEY is not set.
        """
        if not ANTHROPIC_API_KEY:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is required but not set. Add it to your .env file."
            )
        self.agent_name = agent_name
        self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self._logger = logging.getLogger(f"claude_client.{agent_name}")

    def call(
        self,
        system_prompt: str,
        user_message: str,
        model: str = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> str:
        """
        Makes one Claude API call with retry logic.
        Returns the text content of Claude's response.
        Raises after MAX_RETRIES failed attempts.
        Logs: agent_name, first 100 chars of user_message, first 100 chars of response, latency_ms.
        """
        if model is None:
            model = CLAUDE_MODEL

        last_exception: Optional[Exception] = None

        for attempt in range(MAX_RETRIES):
            start_time = time.monotonic()
            try:
                response = self._client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                    temperature=temperature,
                )
                latency_ms = int((time.monotonic() - start_time) * 1000)
                if not response.content:
                    raise ValueError(f"[{self.agent_name}] Claude returned empty content")
                response_text = response.content[0].text
                input_preview = user_message[:100]
                output_preview = response_text[:100]
                self._logger.info(
                    "Claude call | agent=%s | model=%s | latency=%dms | "
                    "input_preview=%s | output_preview=%s",
                    self.agent_name,
                    model,
                    latency_ms,
                    input_preview,
                    output_preview,
                )
                return response_text

            except _FATAL_ERRORS:
                # Fail immediately — no retry
                raise

            except _RETRYABLE_ERRORS as exc:
                last_exception = exc
                self._logger.warning(
                    "Retry %d/%d | agent=%s | error=%s",
                    attempt + 1,
                    MAX_RETRIES,
                    self.agent_name,
                    exc,
                )
                if attempt < MAX_RETRIES - 1:
                    backoff = RETRY_BASE_DELAY * (2 ** attempt)
                    time.sleep(backoff)

        if last_exception is not None:
            raise last_exception
        raise RuntimeError(
            f"[{self.agent_name}] MAX_RETRIES is 0, no API calls were made"
        )
