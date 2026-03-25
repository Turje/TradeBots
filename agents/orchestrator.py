"""
MetaOrchestrator — synthesizes strategy signals from multiple agents into trade decisions.

Uses Claude with weighting logic and a rolling memory of the last 10 decisions.
Pre-filters tickers where 2+ agents returned PASS before calling Claude.
"""

import json
import logging
from collections import deque
from datetime import datetime, timezone

from core.claude_client import ClaudeClient
from core.context import PipelineContext
from core.models import StrategySignal, TradeDecision
from core.portfolio import Portfolio


_SYSTEM_PROMPT = """\
You are a meta-orchestrator synthesizing signals from multiple trading strategy agents.

Weighting rules:
- 3/3 agents agree BUY with confidence > 0.7 → strong BUY (high confidence)
- 2/3 agents agree BUY with confidence > 0.6 → moderate BUY
- Split signals → HOLD (skip this ticker)
- 2+ agents return PASS → skip (do not trade)

Override rules:
- You may upgrade a weak signal to BUY ONLY IF:
  1. You provide a written rationale
  2. The dissenting agent's minimum confidence is >= 0.4
  - You CANNOT BUY on a 1/3 split where any agent has confidence=0.0 (error/pass signals)

Always consider the recent performance context provided."""


class MetaOrchestrator:
    """
    Synthesizes strategy signals from multiple agents into trade decisions.
    Uses Claude with weighting logic and rolling memory of last 10 decisions.
    """

    def __init__(self, portfolio: Portfolio):
        self.client = ClaudeClient(agent_name="orchestrator")
        self.portfolio = portfolio
        self.logger = logging.getLogger("orchestrator")
        self._decision_memory: deque = deque(maxlen=10)  # rolling last-10 decisions

    def decide(
        self,
        signals: list[StrategySignal],
        context: PipelineContext,
    ) -> list[TradeDecision]:
        """
        1. Group signals by ticker
        2. Apply pre-filter rules (before calling Claude):
           - If 2+ agents returned PASS for a ticker → immediately return HOLD, skip Claude
        3. For remaining tickers, send signals to Claude with:
           - Weighting rules in system prompt
           - Memory of last 10 decisions as context
        4. Parse Claude's response into TradeDecision objects
        5. Update _decision_memory with new decisions
        6. Return list[TradeDecision]
        On any error: log error and return [] (safe default — no trades)
        """
        try:
            if not signals:
                return []

            # Step 1: Group signals by ticker
            signals_by_ticker: dict[str, list[StrategySignal]] = {}
            for signal in signals:
                signals_by_ticker.setdefault(signal.ticker, []).append(signal)

            # Step 2: Pre-filter — tickers with 2+ PASS agents skip Claude
            signals_for_claude, pre_filtered_decisions = self._prefilter_signals(
                signals_by_ticker
            )

            # Step 3: Call Claude for remaining tickers
            claude_decisions: list[TradeDecision] = []
            if signals_for_claude:
                claude_decisions = self._call_claude(signals_for_claude)

            # Combine results
            all_decisions = pre_filtered_decisions + claude_decisions

            # Step 5: Update rolling memory with BUY decisions (the meaningful ones)
            self._update_memory(all_decisions)

            return all_decisions

        except Exception as exc:
            self.logger.error("MetaOrchestrator.decide failed: %s", exc)
            return []

    def _prefilter_signals(
        self, signals_by_ticker: dict[str, list[StrategySignal]]
    ) -> tuple[dict, list]:
        """
        Returns (signals_for_claude, pre_filtered_decisions).
        Tickers where 2+ agents returned PASS are removed from Claude's input
        and added as HOLD decisions.
        """
        signals_for_claude: dict[str, list[StrategySignal]] = {}
        pre_filtered_decisions: list[TradeDecision] = []

        for ticker, ticker_signals in signals_by_ticker.items():
            pass_count = sum(1 for s in ticker_signals if s.action == "PASS")
            if pass_count >= 2:
                self.logger.debug(
                    "Pre-filter: %s has %d PASS signals — skipping Claude, returning HOLD",
                    ticker,
                    pass_count,
                )
                pre_filtered_decisions.append(
                    TradeDecision(
                        ticker=ticker,
                        action="HOLD",
                        confidence=0.0,
                        signal_summary=f"{pass_count} agents returned PASS — pre-filtered",
                    )
                )
            else:
                signals_for_claude[ticker] = ticker_signals

        return signals_for_claude, pre_filtered_decisions

    def _call_claude(
        self, signals_for_claude: dict[str, list[StrategySignal]]
    ) -> list[TradeDecision]:
        """Send signals to Claude and parse the response into TradeDecision objects."""
        try:
            # Serialize signals grouped by ticker
            signals_payload = {
                ticker: [
                    {
                        "agent": s.agent_name,
                        "action": s.action,
                        "confidence": s.confidence,
                        "reasoning": s.reasoning,
                    }
                    for s in ticker_signals
                ]
                for ticker, ticker_signals in signals_for_claude.items()
            }

            memory_text = self._format_memory()

            user_message = (
                "Synthesize these strategy signals into trade decisions.\n\n"
                "Signals by ticker:\n"
                f"{json.dumps(signals_payload, indent=2)}\n\n"
                f"Recent decision history (last 10):\n{memory_text}\n\n"
                "For each ticker you recommend trading (BUY only; skip HOLD/PASS tickers in output):\n"
                "[\n"
                "  {\n"
                '    "ticker": "AAPL",\n'
                '    "action": "BUY",\n'
                '    "confidence": 0.75,\n'
                '    "signal_summary": "2/3 agents agree: TechnicalAgent BUY 0.8, '
                'MomentumAgent BUY 0.7, MeanReversionAgent HOLD 0.5"\n'
                "  }\n"
                "]\n\n"
                "Return ONLY the JSON array. If no tickers qualify, return []."
            )

            response_text = self.client.call(
                system_prompt=_SYSTEM_PROMPT,
                user_message=user_message,
            )

            return self._parse_claude_response(response_text)

        except Exception as exc:
            self.logger.error("MetaOrchestrator._call_claude failed: %s", exc)
            return []

    def _parse_claude_response(self, response_text: str) -> list[TradeDecision]:
        """Parse Claude's JSON array into TradeDecision objects. Returns [] on any parse error."""
        try:
            # Strip markdown code fences if present
            text = response_text.strip()
            if text.startswith("```"):
                lines = text.splitlines()
                # Remove first and last fence lines
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines).strip()

            items = json.loads(text)

            if not isinstance(items, list):
                self.logger.warning(
                    "MetaOrchestrator: Claude response is not a JSON array"
                )
                return []

            decisions: list[TradeDecision] = []
            for item in items:
                ticker = item.get("ticker")
                action = item.get("action", "HOLD")
                if action not in ("BUY", "HOLD"):
                    self.logger.warning(
                        "MetaOrchestrator: invalid action '%s' for %s — skipping",
                        action,
                        ticker,
                    )
                    continue
                try:
                    confidence = float(item.get("confidence", 0.0))
                    confidence = max(0.0, min(1.0, confidence))
                except (ValueError, TypeError):
                    confidence = 0.0

                decisions.append(
                    TradeDecision(
                        ticker=ticker,
                        action=action,
                        confidence=confidence,
                        signal_summary=item.get("signal_summary", "No summary provided"),
                    )
                )

            return decisions

        except (json.JSONDecodeError, Exception) as exc:
            self.logger.warning(
                "MetaOrchestrator: Failed to parse Claude response: %s", exc
            )
            return []

    def _format_memory(self) -> str:
        """Format last 10 decisions for inclusion in prompt."""
        if not self._decision_memory:
            return "No recent decisions."
        lines = []
        for entry in self._decision_memory:
            lines.append(
                f"- {entry['timestamp']}: {entry['ticker']} {entry['action']}"
                f" (confidence={entry['confidence']:.2f})"
            )
        return "\n".join(lines)

    def _update_memory(self, decisions: list[TradeDecision]) -> None:
        """Add decisions to rolling memory."""
        for d in decisions:
            self._decision_memory.append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat()[:10],  # date only
                    "ticker": d.ticker,
                    "action": d.action,
                    "confidence": d.confidence,
                }
            )
