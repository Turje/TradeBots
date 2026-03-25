"""
Abstract base class for all strategy agents in the TradeBots pipeline.

Each strategy agent receives a list of Candidates and a read-only PipelineContext,
and returns a list of StrategySignal objects — one per candidate.
Agents MUST NOT modify candidates or context.
"""

import json
import logging
from abc import ABC, abstractmethod

from core.models import Candidate, StrategySignal
from core.context import PipelineContext


class BaseStrategyAgent(ABC):
    """Abstract base for all strategy agents."""

    name: str  # Must be overridden in subclasses as a class attribute

    VALID_ACTIONS: frozenset = frozenset({"BUY", "HOLD", "PASS"})

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Enforce that concrete subclasses define `name`
        if not getattr(cls, '__abstractmethods__', None):  # only check concrete classes
            if not hasattr(cls, 'name') or cls.name is None:
                raise TypeError(
                    f"Concrete subclass {cls.__name__} must define class attribute 'name'"
                )

    @abstractmethod
    def analyze(
        self,
        candidates: list[Candidate],
        context: PipelineContext,
    ) -> list[StrategySignal]:
        """
        Analyze candidates and return signals.
        MUST NOT modify candidates or context — return new StrategySignal objects.
        On failure: return StrategySignal with action=PASS, confidence=0.0, and error note in reasoning.
        """

    def _pass_signals(self, candidates: list[Candidate], reason: str) -> list[StrategySignal]:
        """Return PASS signals for all candidates with a reason."""
        return [
            StrategySignal(
                ticker=c.ticker,
                agent_name=self.name,
                action="PASS",
                confidence=0.0,
                reasoning=reason,
            )
            for c in candidates
        ]

    def _parse_response(
        self,
        response_text: str,
        candidates: list[Candidate],
    ) -> list[StrategySignal]:
        """Parse Claude's JSON response into StrategySignals. Falls back to PASS on parse error."""
        try:
            items = json.loads(response_text)
            if not isinstance(items, list):
                logging.getLogger(__name__).warning(
                    "%s: Claude response is not a JSON array", self.__class__.__name__
                )
                return self._pass_signals(candidates, "Parse error: response not a JSON array")

            raw_by_ticker = {c.ticker: c for c in candidates}
            signals = []

            for item in items:
                ticker = item.get("ticker")
                if ticker not in raw_by_ticker:
                    logging.getLogger(__name__).warning(
                        "%s: Claude returned unknown ticker %s, skipping",
                        self.__class__.__name__,
                        ticker,
                    )
                    continue
                action = item.get("action", "PASS")
                if action not in self.VALID_ACTIONS:
                    logging.getLogger(__name__).warning(
                        "%s: Invalid action %s for %s, using PASS",
                        self.__class__.__name__,
                        action,
                        ticker,
                    )
                    action = "PASS"
                try:
                    confidence = float(item.get("confidence", 0.0))
                    confidence = max(0.0, min(1.0, confidence))
                except (ValueError, TypeError):
                    logging.getLogger(__name__).warning(
                        "%s: Non-numeric confidence for %s, using 0.0",
                        self.__class__.__name__,
                        ticker,
                    )
                    confidence = 0.0

                signals.append(StrategySignal(
                    ticker=ticker,
                    agent_name=self.name,
                    action=action,
                    confidence=confidence,
                    reasoning=item.get("reasoning", "No reasoning provided"),
                ))

            # Add PASS for any candidates Claude omitted
            returned_tickers = {s.ticker for s in signals}
            for c in candidates:
                if c.ticker not in returned_tickers:
                    signals.append(StrategySignal(
                        ticker=c.ticker,
                        agent_name=self.name,
                        action="PASS",
                        confidence=0.0,
                        reasoning="Not analyzed by Claude",
                    ))

            return signals

        except (json.JSONDecodeError, Exception) as exc:
            logging.getLogger(__name__).warning(
                "%s: Failed to parse response: %s", self.__class__.__name__, exc
            )
            return self._pass_signals(candidates, f"Parse error: {exc}")
