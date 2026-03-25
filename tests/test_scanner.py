"""
Unit tests for agents/scanner.py (ScannerBot).

All external dependencies (MarketDataProvider, ClaudeClient) are fully mocked.
No network I/O occurs.

Run with:
    cd /Users/turje87/Desktop/TradeBots && python -m pytest tests/test_scanner.py -v
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from agents.scanner import ScannerBot, DEFAULT_STOCK_TICKERS
from core.models import Candidate


# ---------------------------------------------------------------------------
# Helpers — synthetic market data dicts
# ---------------------------------------------------------------------------

def _make_raw_stock(
    ticker: str = "AAPL",
    price: float = 175.0,
    volume_ratio: float = 2.0,
    price_change_pct: float = 3.5,
    asset_type: str = "stock",
) -> dict:
    return {
        "ticker": ticker,
        "asset_type": asset_type,
        "price": price,
        "volume_ratio": volume_ratio,
        "price_change_pct": price_change_pct,
        "indicators": {
            "RSI": 65.0,
            "MACD_line": 0.5,
            "MACD_signal": 0.3,
            "SMA20": 170.0,
            "SMA50": 160.0,
        },
    }


def _make_raw_crypto(
    ticker: str = "BTC/USDT",
    price: float = 65_000.0,
    volume_ratio: float = 1.8,
    price_change_pct: float = 2.1,
) -> dict:
    return {
        "ticker": ticker,
        "asset_type": "crypto",
        "price": price,
        "volume_ratio": volume_ratio,
        "price_change_pct": price_change_pct,
        "indicators": {
            "RSI": 58.0,
            "MACD_line": 200.0,
            "MACD_signal": 150.0,
            "SMA20": 63_000.0,
            "SMA50": 60_000.0,
        },
    }


def _claude_json_response(items: list) -> str:
    """Serialise a list of ranked items the way Claude would return them."""
    return json.dumps(items)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def scanner():
    """
    Return a ScannerBot whose internal dependencies are replaced with mocks.
    Callers set return values on scanner.market_data and scanner.client as needed.
    """
    with (
        patch("agents.scanner.ClaudeClient") as mock_claude_cls,
        patch("agents.scanner.MarketDataProvider") as mock_mdp_cls,
    ):
        mock_claude = MagicMock()
        mock_mdp = MagicMock()
        mock_claude_cls.return_value = mock_claude
        mock_mdp_cls.return_value = mock_mdp

        bot = ScannerBot()
        # Expose mocks for convenient access in tests
        bot._mock_claude = mock_claude
        bot._mock_mdp = mock_mdp
        yield bot


# ---------------------------------------------------------------------------
# Test: happy path — Claude returns valid ranked JSON
# ---------------------------------------------------------------------------

class TestScanHappyPath:
    def test_returns_candidate_objects(self, scanner):
        """scan() should return a list of Candidate dataclass instances."""
        raw_stock = _make_raw_stock("AAPL")
        raw_crypto = _make_raw_crypto("BTC/USDT")

        scanner._mock_mdp.get_stock_candidates.return_value = [raw_stock]
        scanner._mock_mdp.get_crypto_candidates.return_value = [raw_crypto]

        claude_items = [
            {"ticker": "AAPL", "asset_type": "stock", "priority": 1,
             "rationale": "Strong RSI with volume"},
            {"ticker": "BTC/USDT", "asset_type": "crypto", "priority": 2,
             "rationale": "Crypto momentum"},
        ]
        scanner._mock_claude.call.return_value = _claude_json_response(claude_items)

        result = scanner.scan(stock_tickers=["AAPL"], crypto_symbols=["BTC/USDT"])

        assert len(result) == 2
        assert all(isinstance(c, Candidate) for c in result)

    def test_sorted_by_priority(self, scanner):
        """Candidates are ordered by Claude's priority field (lower = first)."""
        raw_aapl = _make_raw_stock("AAPL")
        raw_msft = _make_raw_stock("MSFT", price=420.0)

        scanner._mock_mdp.get_stock_candidates.return_value = [raw_aapl, raw_msft]
        scanner._mock_mdp.get_crypto_candidates.return_value = []

        # Claude ranks MSFT first (priority 1) and AAPL second (priority 2)
        claude_items = [
            {"ticker": "MSFT", "asset_type": "stock", "priority": 1,
             "rationale": "Better momentum"},
            {"ticker": "AAPL", "asset_type": "stock", "priority": 2,
             "rationale": "Solid but second"},
        ]
        scanner._mock_claude.call.return_value = _claude_json_response(claude_items)

        result = scanner.scan(stock_tickers=["AAPL", "MSFT"], crypto_symbols=[])

        assert result[0].ticker == "MSFT"
        assert result[1].ticker == "AAPL"

    def test_candidate_fields_populated_from_raw_data(self, scanner):
        """
        price, volume_ratio, price_change_pct, and indicators come from
        the MarketDataProvider raw dict, not from Claude's response.
        rationale comes from Claude.
        """
        raw = _make_raw_stock("AAPL", price=178.5, volume_ratio=3.2, price_change_pct=4.1)
        scanner._mock_mdp.get_stock_candidates.return_value = [raw]
        scanner._mock_mdp.get_crypto_candidates.return_value = []

        claude_items = [
            {"ticker": "AAPL", "asset_type": "stock", "priority": 1,
             "rationale": "Excellent setup"},
        ]
        scanner._mock_claude.call.return_value = _claude_json_response(claude_items)

        result = scanner.scan(stock_tickers=["AAPL"], crypto_symbols=[])

        assert len(result) == 1
        c = result[0]
        assert c.ticker == "AAPL"
        assert c.asset_type == "stock"
        assert c.price == pytest.approx(178.5)
        assert c.volume_ratio == pytest.approx(3.2)
        assert c.price_change_pct == pytest.approx(4.1)
        assert c.rationale == "Excellent setup"

    def test_indicators_dict_populated(self, scanner):
        """The indicators dict from market data is preserved on the Candidate."""
        raw = _make_raw_stock("NVDA")
        scanner._mock_mdp.get_stock_candidates.return_value = [raw]
        scanner._mock_mdp.get_crypto_candidates.return_value = []

        claude_items = [
            {"ticker": "NVDA", "asset_type": "stock", "priority": 1,
             "rationale": "GPU demand"},
        ]
        scanner._mock_claude.call.return_value = _claude_json_response(claude_items)

        result = scanner.scan(stock_tickers=["NVDA"], crypto_symbols=[])

        assert len(result) == 1
        ind = result[0].indicators
        for key in ("RSI", "MACD_line", "MACD_signal", "SMA20", "SMA50"):
            assert key in ind

    def test_default_stock_tickers_used_when_none(self, scanner):
        """When stock_tickers=None, DEFAULT_STOCK_TICKERS should be passed to MarketDataProvider."""
        scanner._mock_mdp.get_stock_candidates.return_value = []
        scanner._mock_mdp.get_crypto_candidates.return_value = []

        scanner.scan()

        call_args = scanner._mock_mdp.get_stock_candidates.call_args
        tickers_passed = call_args[0][0]
        assert tickers_passed == DEFAULT_STOCK_TICKERS

    def test_crypto_symbols_none_passed_through(self, scanner):
        """When crypto_symbols=None, None is passed to get_crypto_candidates."""
        scanner._mock_mdp.get_stock_candidates.return_value = []
        scanner._mock_mdp.get_crypto_candidates.return_value = []

        scanner.scan(crypto_symbols=None)

        call_args = scanner._mock_mdp.get_crypto_candidates.call_args
        symbols_passed = call_args[0][0]
        assert symbols_passed is None


# ---------------------------------------------------------------------------
# Test: no candidates pass market filters
# ---------------------------------------------------------------------------

class TestNoMarketCandidates:
    def test_returns_empty_list(self, scanner):
        """When no candidates pass market filters, scan() returns []."""
        scanner._mock_mdp.get_stock_candidates.return_value = []
        scanner._mock_mdp.get_crypto_candidates.return_value = []

        result = scanner.scan(stock_tickers=["AAPL"], crypto_symbols=["BTC/USDT"])

        assert result == []

    def test_claude_not_called_when_no_candidates(self, scanner):
        """Claude should NOT be called when no market candidates are found."""
        scanner._mock_mdp.get_stock_candidates.return_value = []
        scanner._mock_mdp.get_crypto_candidates.return_value = []

        scanner.scan(stock_tickers=["AAPL"], crypto_symbols=[])

        scanner._mock_claude.call.assert_not_called()


# ---------------------------------------------------------------------------
# Test: Claude JSON parse fails → fallback to raw candidates
# ---------------------------------------------------------------------------

class TestClaudeJsonParseFallback:
    def test_returns_raw_candidates_on_invalid_json(self, scanner):
        """If Claude returns non-JSON, scan() falls back to raw market data as Candidates."""
        raw_stock = _make_raw_stock("AAPL")
        scanner._mock_mdp.get_stock_candidates.return_value = [raw_stock]
        scanner._mock_mdp.get_crypto_candidates.return_value = []

        scanner._mock_claude.call.return_value = "This is not valid JSON at all."

        result = scanner.scan(stock_tickers=["AAPL"], crypto_symbols=[])

        assert len(result) == 1
        assert isinstance(result[0], Candidate)
        assert result[0].ticker == "AAPL"

    def test_fallback_rationale_is_generic(self, scanner):
        """Fallback Candidates use 'Market filter passed' as the rationale."""
        raw_stock = _make_raw_stock("TSLA")
        scanner._mock_mdp.get_stock_candidates.return_value = [raw_stock]
        scanner._mock_mdp.get_crypto_candidates.return_value = []

        scanner._mock_claude.call.return_value = "not json"

        result = scanner.scan(stock_tickers=["TSLA"], crypto_symbols=[])

        assert result[0].rationale == "Market filter passed"

    def test_fallback_preserves_price_fields(self, scanner):
        """Fallback Candidates retain price, volume_ratio, and price_change_pct from raw data."""
        raw = _make_raw_stock("META", price=520.0, volume_ratio=1.9, price_change_pct=-2.3)
        scanner._mock_mdp.get_stock_candidates.return_value = [raw]
        scanner._mock_mdp.get_crypto_candidates.return_value = []

        scanner._mock_claude.call.return_value = "{bad json"

        result = scanner.scan(stock_tickers=["META"], crypto_symbols=[])

        c = result[0]
        assert c.price == pytest.approx(520.0)
        assert c.volume_ratio == pytest.approx(1.9)
        assert c.price_change_pct == pytest.approx(-2.3)

    def test_fallback_when_claude_returns_non_array_json(self, scanner):
        """If Claude returns valid JSON that is not an array, fallback is triggered."""
        raw_stock = _make_raw_stock("AAPL")
        scanner._mock_mdp.get_stock_candidates.return_value = [raw_stock]
        scanner._mock_mdp.get_crypto_candidates.return_value = []

        # Valid JSON but not an array
        scanner._mock_claude.call.return_value = json.dumps({"error": "unexpected format"})

        result = scanner.scan(stock_tickers=["AAPL"], crypto_symbols=[])

        assert len(result) == 1
        assert result[0].rationale == "Market filter passed"


# ---------------------------------------------------------------------------
# Test: MarketDataProvider raises an exception
# ---------------------------------------------------------------------------

class TestMarketDataProviderException:
    def test_returns_empty_list_on_stock_exception(self, scanner):
        """If get_stock_candidates raises, scan() logs the error and returns []."""
        scanner._mock_mdp.get_stock_candidates.side_effect = RuntimeError("Network error")

        result = scanner.scan(stock_tickers=["AAPL"], crypto_symbols=[])

        assert result == []

    def test_returns_empty_list_on_crypto_exception(self, scanner):
        """If get_crypto_candidates raises, scan() logs the error and returns []."""
        scanner._mock_mdp.get_stock_candidates.return_value = []
        scanner._mock_mdp.get_crypto_candidates.side_effect = RuntimeError("Exchange down")

        result = scanner.scan(stock_tickers=["AAPL"], crypto_symbols=["BTC/USDT"])

        assert result == []

    def test_claude_not_called_on_market_data_exception(self, scanner):
        """Claude should not be called if market data fetch raises."""
        scanner._mock_mdp.get_stock_candidates.side_effect = RuntimeError("Timeout")

        scanner.scan(stock_tickers=["AAPL"], crypto_symbols=[])

        scanner._mock_claude.call.assert_not_called()

    def test_returns_empty_list_on_claude_exception(self, scanner):
        """If Claude.call raises, scan() logs the error and returns []."""
        raw = _make_raw_stock("AAPL")
        scanner._mock_mdp.get_stock_candidates.return_value = [raw]
        scanner._mock_mdp.get_crypto_candidates.return_value = []

        scanner._mock_claude.call.side_effect = RuntimeError("API error")

        result = scanner.scan(stock_tickers=["AAPL"], crypto_symbols=[])

        assert result == []


# ---------------------------------------------------------------------------
# Test: mixed stocks and crypto in a single scan
# ---------------------------------------------------------------------------

class TestMixedAssetTypes:
    def test_both_stocks_and_crypto_returned(self, scanner):
        """scan() correctly handles a mix of stock and crypto candidates."""
        raw_stock = _make_raw_stock("AAPL")
        raw_crypto = _make_raw_crypto("ETH/USDT")

        scanner._mock_mdp.get_stock_candidates.return_value = [raw_stock]
        scanner._mock_mdp.get_crypto_candidates.return_value = [raw_crypto]

        claude_items = [
            {"ticker": "AAPL", "asset_type": "stock", "priority": 1,
             "rationale": "Stock play"},
            {"ticker": "ETH/USDT", "asset_type": "crypto", "priority": 2,
             "rationale": "Crypto play"},
        ]
        scanner._mock_claude.call.return_value = _claude_json_response(claude_items)

        result = scanner.scan(stock_tickers=["AAPL"], crypto_symbols=["ETH/USDT"])

        tickers = [c.ticker for c in result]
        asset_types = [c.asset_type for c in result]

        assert "AAPL" in tickers
        assert "ETH/USDT" in tickers
        assert "stock" in asset_types
        assert "crypto" in asset_types

    def test_crypto_asset_type_set_correctly(self, scanner):
        """Crypto candidates should have asset_type='crypto'."""
        raw_crypto = _make_raw_crypto("SOL/USDT")
        scanner._mock_mdp.get_stock_candidates.return_value = []
        scanner._mock_mdp.get_crypto_candidates.return_value = [raw_crypto]

        claude_items = [
            {"ticker": "SOL/USDT", "asset_type": "crypto", "priority": 1,
             "rationale": "Solana momentum"},
        ]
        scanner._mock_claude.call.return_value = _claude_json_response(claude_items)

        result = scanner.scan(stock_tickers=[], crypto_symbols=["SOL/USDT"])

        assert len(result) == 1
        assert result[0].asset_type == "crypto"


# ---------------------------------------------------------------------------
# Test: Claude returns a ticker not in the raw candidates (defensive)
# ---------------------------------------------------------------------------

class TestClaudeReturnsUnknownTicker:
    def test_unknown_ticker_skipped(self, scanner):
        """If Claude references a ticker not in raw data, it is silently skipped."""
        raw = _make_raw_stock("AAPL")
        scanner._mock_mdp.get_stock_candidates.return_value = [raw]
        scanner._mock_mdp.get_crypto_candidates.return_value = []

        claude_items = [
            {"ticker": "AAPL", "asset_type": "stock", "priority": 1,
             "rationale": "Valid"},
            {"ticker": "UNKNOWN", "asset_type": "stock", "priority": 2,
             "rationale": "Should be skipped"},
        ]
        scanner._mock_claude.call.return_value = _claude_json_response(claude_items)

        result = scanner.scan(stock_tickers=["AAPL"], crypto_symbols=[])

        tickers = [c.ticker for c in result]
        assert "AAPL" in tickers
        assert "UNKNOWN" not in tickers
