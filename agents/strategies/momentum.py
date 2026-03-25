"""
MomentumAgent — strategy agent focused on trend continuation and breakout
patterns over 5/20/50-day windows.

Returns one StrategySignal per candidate based on momentum analysis.
"""

import json
import logging

from agents.strategies.base import BaseStrategyAgent
from core.claude_client import ClaudeClient
from core.context import PipelineContext
from core.models import Candidate, StrategySignal

_SYSTEM_PROMPT = (
    "You are a momentum trading expert evaluating trend continuation opportunities.\n"
    "Focus on: price momentum (is it accelerating?), volume confirmation of moves,\n"
    "trend strength across timeframes (5d, 20d, 50d), and breakout patterns."
)


class MomentumAgent(BaseStrategyAgent):
    """
    Strategy agent that evaluates candidates using momentum indicators:
    price change, volume ratio, and moving average trend across timeframes.
    """

    name = "momentum"

    def __init__(self):
        self.client = ClaudeClient(agent_name="strategy.momentum")
        self.logger = logging.getLogger("strategy.momentum")

    def analyze(
        self,
        candidates: list[Candidate],
        context: PipelineContext,
    ) -> list[StrategySignal]:
        """
        Send candidates with momentum-focused data to Claude and parse the response.
        Returns one StrategySignal per candidate.
        On any failure, returns PASS signals with confidence=0.0 for all candidates.
        """
        if not candidates:
            return []

        try:
            # Build a compact representation focusing on momentum indicators
            candidates_data = [
                {
                    "ticker": c.ticker,
                    "price": c.price,
                    "price_change_pct": c.price_change_pct,
                    "volume_ratio": c.volume_ratio,
                    "indicators": {
                        "SMA20": c.indicators.get("SMA20"),
                        "SMA50": c.indicators.get("SMA50"),
                        "RSI": c.indicators.get("RSI"),
                    },
                }
                for c in candidates
            ]

            candidates_json = json.dumps(candidates_data, indent=2)
            user_message = (
                "Analyze these candidates for momentum and trend continuation.\n\n"
                f"{candidates_json}\n\n"
                "For each candidate, return a JSON array:\n"
                "[\n"
                "  {\n"
                '    "ticker": "AAPL",\n'
                '    "action": "BUY",\n'
                '    "confidence": 0.75,\n'
                '    "reasoning": "Strong uptrend with price above SMA50 and accelerating momentum..."\n'
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
            self.logger.error("MomentumAgent.analyze failed: %s", exc)
            return self._pass_signals(candidates, f"Agent error: {exc}")

