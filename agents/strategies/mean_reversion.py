"""
MeanReversionAgent — strategy agent that identifies overbought/oversold
conditions using RSI extremes, Bollinger Bands, and deviation from moving averages.

Returns one StrategySignal per candidate based on mean reversion analysis.
"""

import json
import logging

from agents.strategies.base import BaseStrategyAgent
from core.claude_client import ClaudeClient
from core.context import PipelineContext
from core.models import Candidate, StrategySignal

_SYSTEM_PROMPT = (
    "You are a mean reversion expert identifying overbought and oversold conditions.\n"
    "Focus on: RSI extremes (<30 oversold, >70 overbought), distance from moving averages,\n"
    "Bollinger Band positions, and conditions where price is likely to revert to the mean."
)


class MeanReversionAgent(BaseStrategyAgent):
    """
    Strategy agent that evaluates candidates for mean reversion opportunities:
    RSI extremes, deviation from SMA20/SMA50, and Bollinger Band positions.
    """

    name = "mean_reversion"

    def __init__(self):
        self.client = ClaudeClient(agent_name="strategy.mean_reversion")
        self.logger = logging.getLogger("strategy.mean_reversion")

    def analyze(
        self,
        candidates: list[Candidate],
        context: PipelineContext,
    ) -> list[StrategySignal]:
        """
        Send candidates with mean-reversion-focused indicators to Claude and parse the response.
        Returns one StrategySignal per candidate.
        On any failure, returns PASS signals with confidence=0.0 for all candidates.
        """
        if not candidates:
            return []

        try:
            # Build a compact representation focusing on mean reversion indicators
            candidates_data = [
                {
                    "ticker": c.ticker,
                    "price": c.price,
                    "price_change_pct": c.price_change_pct,
                    "indicators": {
                        "RSI": c.indicators.get("RSI"),
                        "SMA20": c.indicators.get("SMA20"),
                        "SMA50": c.indicators.get("SMA50"),
                        "BB_upper": c.indicators.get("BB_upper"),
                        "BB_lower": c.indicators.get("BB_lower"),
                        "BB_middle": c.indicators.get("BB_middle"),
                    },
                }
                for c in candidates
            ]

            candidates_json = json.dumps(candidates_data, indent=2)
            user_message = (
                "Analyze these candidates for mean reversion opportunities.\n\n"
                f"{candidates_json}\n\n"
                "For each candidate, return a JSON array:\n"
                "[\n"
                "  {\n"
                '    "ticker": "AAPL",\n'
                '    "action": "BUY",\n'
                '    "confidence": 0.75,\n'
                '    "reasoning": "RSI at 28 signals oversold condition, price near lower Bollinger Band..."\n'
                "  },\n"
                "  ...\n"
                "]\n\n"
                "Actions: BUY, HOLD, or PASS. Confidence 0.0-1.0. Return ONLY the JSON array."
            )

            response_text = self.client.call(
                system_prompt=_SYSTEM_PROMPT,
                user_message=user_message,
            )

            return self._parse_response(response_text, candidates)

        except Exception as exc:
            self.logger.error("MeanReversionAgent.analyze failed: %s", exc)
            return self._pass_signals(candidates, f"Agent error: {exc}")

