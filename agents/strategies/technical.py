"""
TechnicalAgent — strategy agent focused on RSI, MACD crossovers,
and moving average relationships.

Returns one StrategySignal per candidate based on technical indicator analysis.
"""

import json
import logging

from agents.strategies.base import BaseStrategyAgent
from core.claude_client import ClaudeClient
from core.context import PipelineContext
from core.models import Candidate, StrategySignal


_SYSTEM_PROMPT = (
    "You are a technical analysis expert evaluating trading signals.\n"
    "Focus on: RSI levels (oversold <30, overbought >70), MACD crossovers (bullish when MACD line crosses above signal),\n"
    "moving average relationships (price above SMA50 is bullish), and trend direction."
)


class TechnicalAgent(BaseStrategyAgent):
    """
    Strategy agent that evaluates candidates using technical indicators:
    RSI, MACD line/signal crossovers, and SMA20/SMA50 relationships.
    """

    name = "technical"

    def __init__(self):
        self.client = ClaudeClient(agent_name="strategy.technical")
        self.logger = logging.getLogger("strategy.technical")

    def analyze(
        self,
        candidates: list[Candidate],
        context: PipelineContext,
    ) -> list[StrategySignal]:
        """
        Send candidates with technical indicators to Claude and parse the response.
        Returns one StrategySignal per candidate.
        On any failure, returns PASS signals with confidence=0.0 for all candidates.
        """
        if not candidates:
            return []

        try:
            # Build a compact representation focusing on technical indicators
            candidates_data = [
                {
                    "ticker": c.ticker,
                    "price": c.price,
                    "price_change_pct": c.price_change_pct,
                    "indicators": {
                        "RSI": c.indicators.get("RSI"),
                        "MACD_line": c.indicators.get("MACD_line"),
                        "MACD_signal": c.indicators.get("MACD_signal"),
                        "SMA20": c.indicators.get("SMA20"),
                        "SMA50": c.indicators.get("SMA50"),
                    },
                }
                for c in candidates
            ]

            candidates_json = json.dumps(candidates_data, indent=2)
            user_message = (
                "Analyze these candidates using technical indicators.\n\n"
                f"{candidates_json}\n\n"
                "For each candidate, return a JSON array:\n"
                "[\n"
                "  {\n"
                '    "ticker": "AAPL",\n'
                '    "action": "BUY",\n'
                '    "confidence": 0.75,\n'
                '    "reasoning": "RSI at 45 shows neutral momentum..."\n'
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
            self.logger.error("TechnicalAgent.analyze failed: %s", exc)
            return self._pass_signals(candidates, f"Agent error: {exc}")

