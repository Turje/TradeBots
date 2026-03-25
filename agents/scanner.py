"""
ScannerBot — first stage of the TradeBots pipeline.

Fetches filtered stock and crypto candidates from MarketDataProvider, then
uses Claude to reason about which candidates deserve deeper analysis and
returns a ranked list of Candidate objects with a rationale for each.
"""

import json
import logging
from typing import Optional

from core.claude_client import ClaudeClient
from core.models import Candidate
from brokers.market_data import MarketDataProvider

DEFAULT_STOCK_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B",
    "LLY", "JPM", "V", "XOM", "UNH", "AVGO", "JNJ", "WMT", "MA", "PG", "HD", "COST"
]

_SYSTEM_PROMPT = (
    "You are a market scanner analyzing potential trading opportunities.\n"
    "Your job is to rank and filter candidates based on their technical indicators and price action.\n"
    "Focus on: momentum strength, volume confirmation, indicator alignment, and risk/reward potential."
)


class ScannerBot:
    """
    Filters stocks/crypto by price change and volume, then uses Claude to reason
    about which candidates deserve deeper analysis.

    Returns List[Candidate] set into PipelineContext.candidates.
    Error handling: returns empty list on any failure (logged).
    """

    def __init__(self):
        self.client = ClaudeClient(agent_name="scanner")
        self.market_data = MarketDataProvider()
        self.logger = logging.getLogger("scanner")

    def scan(
        self,
        stock_tickers: Optional[list] = None,
        crypto_symbols: Optional[list] = None,
    ) -> list[Candidate]:
        """
        1. Fetch filtered candidates from MarketDataProvider (stocks + crypto)
        2. If no candidates pass filters: log info and return []
        3. Send candidates to Claude with a structured prompt
        4. Parse Claude's JSON response into List[Candidate]
        5. Return candidates (sorted by Claude's priority)
        On any error: log error and return []
        """
        try:
            if stock_tickers is None:
                stock_tickers = DEFAULT_STOCK_TICKERS

            # Fetch filtered stock candidates
            stock_candidates = self.market_data.get_stock_candidates(stock_tickers)

            # Fetch filtered crypto candidates (None triggers MarketDataProvider defaults)
            crypto_candidates = self.market_data.get_crypto_candidates(crypto_symbols)

            # Tag asset type on each raw dict (non-mutating copies)
            stock_candidates = [{**c, "asset_type": "stock"} for c in stock_candidates]
            crypto_candidates = [{**c, "asset_type": "crypto"} for c in crypto_candidates]

            all_candidates = stock_candidates + crypto_candidates

            if not all_candidates:
                self.logger.info("No candidates passed market filters.")
                return []

            # Cap candidates before sending to Claude
            MAX_SCANNER_CANDIDATES = 20
            all_candidates = all_candidates[:MAX_SCANNER_CANDIDATES]

            # Build user message for Claude
            candidates_json = json.dumps(all_candidates, indent=2)
            user_message = (
                "Analyze these market candidates and rank the top ones worth investigating further.\n\n"
                f"Candidates:\n{candidates_json}\n\n"
                "For each candidate you want to keep, return a JSON array:\n"
                "[\n"
                "  {\n"
                '    "ticker": "AAPL",\n'
                '    "asset_type": "stock",\n'
                '    "priority": 1,\n'
                '    "rationale": "Strong RSI momentum with volume confirmation..."\n'
                "  },\n"
                "  ...\n"
                "]\n\n"
                "Return ONLY the JSON array. No other text. Maximum 5 candidates."
            )

            response_text = self.client.call(
                system_prompt=_SYSTEM_PROMPT,
                user_message=user_message,
            )

            return self._parse_response(response_text, all_candidates)

        except Exception as exc:
            self.logger.error("ScannerBot.scan failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_response(self, response_text: str, raw_candidates: list) -> list[Candidate]:
        """
        Parse Claude's JSON response into a sorted list of Candidate objects.

        On JSON parsing failure, falls back to converting raw market data
        dicts directly into Candidate objects with a generic rationale.
        """
        # Build a lookup from ticker -> raw dict for fast access
        raw_by_ticker = {c["ticker"]: c for c in raw_candidates}

        try:
            ranked = json.loads(response_text)
            if not isinstance(ranked, list):
                raise ValueError("Claude response is not a JSON array")

            candidates: list[Candidate] = []
            for item in ranked:
                ticker = item.get("ticker")
                raw = raw_by_ticker.get(ticker)
                if raw is None:
                    self.logger.warning(
                        "Claude returned ticker '%s' not in raw candidates — skipping", ticker
                    )
                    continue
                candidates.append(
                    Candidate(
                        ticker=ticker,
                        asset_type=item.get("asset_type", raw.get("asset_type", "stock")),
                        price=raw["price"],
                        volume_ratio=raw["volume_ratio"],
                        price_change_pct=raw["price_change_pct"],
                        indicators=raw["indicators"],
                        rationale=item.get("rationale", ""),
                    )
                )

            # Build priority map once before sort (O(n) instead of O(n*m))
            priority_map = {r.get("ticker"): r.get("priority", 999) for r in ranked}
            candidates.sort(key=lambda c: priority_map.get(c.ticker, 999))
            return candidates

        except (json.JSONDecodeError, ValueError) as exc:
            self.logger.warning(
                "Failed to parse Claude JSON response (%s) — falling back to raw candidates", exc
            )
            return self._raw_to_candidates(raw_candidates)

    def _raw_to_candidates(self, raw_candidates: list) -> list[Candidate]:
        """Convert raw market data dicts into Candidate objects using a generic rationale."""
        return [
            Candidate(
                ticker=c["ticker"],
                asset_type=c.get("asset_type", "stock"),
                price=c["price"],
                volume_ratio=c["volume_ratio"],
                price_change_pct=c["price_change_pct"],
                indicators=c["indicators"],
                rationale="Market filter passed",
            )
            for c in raw_candidates
        ]
