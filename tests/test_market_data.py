"""
Unit tests for brokers/market_data.py (MarketDataProvider).
All network calls (yfinance, ccxt) are mocked — no real I/O occurs.

Run with:
    cd /Users/turje87/Desktop/TradeBots && python -m pytest tests/test_market_data.py -v
"""

import math
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from brokers.market_data import MarketDataProvider, _compute_indicators


# ---------------------------------------------------------------------------
# Helpers — synthetic OHLCV DataFrames
# ---------------------------------------------------------------------------

def _make_stock_df(
    n_rows: int = 21,
    base_close: float = 100.0,
    last_close: float = 103.0,
    base_volume: float = 1_000_000.0,
    last_volume: float = 2_000_000.0,
) -> pd.DataFrame:
    """
    Build a synthetic yfinance-style DataFrame with 'open', 'high', 'low',
    'close', 'volume' columns and a DatetimeIndex.
    The last row has the supplied last_close / last_volume.
    All prior rows use base_close / base_volume.
    """
    index = pd.date_range(end="2024-01-21", periods=n_rows, freq="B")
    closes = [base_close] * (n_rows - 1) + [last_close]
    volumes = [base_volume] * (n_rows - 1) + [last_volume]
    df = pd.DataFrame(
        {
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Volume": volumes,
        },
        index=index,
    )
    return df


def _make_ccxt_ohlcv(
    n_rows: int = 21,
    base_close: float = 30_000.0,
    last_close: float = 31_500.0,
    base_volume: float = 5_000.0,
    last_volume: float = 8_000.0,
) -> list:
    """Build synthetic ccxt OHLCV list: [[ts, open, high, low, close, volume], ...]"""
    import time as _time

    rows = []
    for i in range(n_rows):
        ts = int((_time.time() - (n_rows - i) * 86400) * 1000)
        close = last_close if i == n_rows - 1 else base_close
        vol = last_volume if i == n_rows - 1 else base_volume
        rows.append([ts, close, close, close, close, vol])
    return rows


# ---------------------------------------------------------------------------
# _compute_indicators unit tests
# ---------------------------------------------------------------------------

class TestComputeIndicators:
    def test_returns_all_keys(self):
        closes = pd.Series([float(i) for i in range(50, 100)])
        result = _compute_indicators(closes)
        assert set(result.keys()) == {"RSI", "MACD_line", "MACD_signal", "SMA20", "SMA50"}

    def test_sma20_correct(self):
        """SMA20 should equal the mean of the last 20 values."""
        values = list(range(1, 51))  # 50 values
        closes = pd.Series(values, dtype=float)
        result = _compute_indicators(closes)
        expected_sma20 = sum(range(31, 51)) / 20  # last 20 values: 31..50
        assert result["SMA20"] == pytest.approx(expected_sma20, rel=1e-6)

    def test_sma50_correct(self):
        """SMA50 should equal the mean of the last 50 values when there are exactly 50."""
        values = list(range(1, 51))  # 50 values
        closes = pd.Series(values, dtype=float)
        result = _compute_indicators(closes)
        expected_sma50 = sum(range(1, 51)) / 50
        assert result["SMA50"] == pytest.approx(expected_sma50, rel=1e-6)

    def test_rsi_is_float_or_none(self):
        closes = pd.Series([float(i) for i in range(100, 150)])
        result = _compute_indicators(closes)
        # RSI should be a float in range [0, 100] or None
        if result["RSI"] is not None:
            assert 0.0 <= result["RSI"] <= 100.0

    def test_nan_values_converted_to_none(self):
        """With only 2 data points, SMA50 will be NaN — should be returned as None."""
        closes = pd.Series([100.0, 101.0])
        result = _compute_indicators(closes)
        assert result["SMA50"] is None

    def test_macd_keys_present(self):
        closes = pd.Series([float(i) for i in range(50, 100)])
        result = _compute_indicators(closes)
        assert "MACD_line" in result
        assert "MACD_signal" in result


# ---------------------------------------------------------------------------
# MarketDataProvider._parse_period_days
# ---------------------------------------------------------------------------

class TestParsePeriodDays:
    def test_days(self):
        assert MarketDataProvider._parse_period_days("60d") == 60

    def test_months(self):
        assert MarketDataProvider._parse_period_days("3mo") == 90

    def test_years(self):
        assert MarketDataProvider._parse_period_days("1y") == 365

    def test_numeric_string(self):
        assert MarketDataProvider._parse_period_days("30") == 30


# ---------------------------------------------------------------------------
# MarketDataProvider.get_stock_candidates — filter logic
# ---------------------------------------------------------------------------

class TestGetStockCandidatesFiltering:
    """Test filter logic with mocked yfinance."""

    def _provider(self):
        return MarketDataProvider()

    @patch("brokers.market_data.yf.download")
    def test_candidate_passes_both_filters(self, mock_dl):
        """last_close 3% above prev_close, volume_ratio 2.0 → passes 1%/1.0 filters."""
        mock_dl.return_value = _make_stock_df(
            n_rows=21,
            base_close=100.0,
            last_close=103.0,   # +3%
            base_volume=1_000_000.0,
            last_volume=2_000_000.0,  # ratio 2.0
        )
        provider = self._provider()
        results = provider.get_stock_candidates(["AAPL"], min_price_change_pct=1.0, min_volume_ratio=1.0)
        assert len(results) == 1
        assert results[0]["ticker"] == "AAPL"

    @patch("brokers.market_data.yf.download")
    def test_candidate_filtered_out_by_price_change(self, mock_dl):
        """last_close only 0.5% above prev → filtered out by min_price_change_pct=1.0."""
        mock_dl.return_value = _make_stock_df(
            n_rows=21,
            base_close=100.0,
            last_close=100.5,   # +0.5%
            base_volume=1_000_000.0,
            last_volume=2_000_000.0,
        )
        provider = self._provider()
        results = provider.get_stock_candidates(["AAPL"], min_price_change_pct=1.0, min_volume_ratio=1.0)
        assert len(results) == 0

    @patch("brokers.market_data.yf.download")
    def test_candidate_filtered_out_by_volume_ratio(self, mock_dl):
        """Volume ratio 0.8 < min_volume_ratio=1.0 → filtered out."""
        mock_dl.return_value = _make_stock_df(
            n_rows=21,
            base_close=100.0,
            last_close=103.0,
            base_volume=1_000_000.0,
            last_volume=800_000.0,  # ratio 0.8
        )
        provider = self._provider()
        results = provider.get_stock_candidates(["AAPL"], min_price_change_pct=1.0, min_volume_ratio=1.0)
        assert len(results) == 0

    @patch("brokers.market_data.yf.download")
    def test_negative_price_change_passes_abs_filter(self, mock_dl):
        """A -3% price change should pass |price_change_pct| > 1.0."""
        mock_dl.return_value = _make_stock_df(
            n_rows=21,
            base_close=100.0,
            last_close=97.0,    # -3%
            base_volume=1_000_000.0,
            last_volume=2_000_000.0,
        )
        provider = self._provider()
        results = provider.get_stock_candidates(["TSLA"], min_price_change_pct=1.0, min_volume_ratio=1.0)
        assert len(results) == 1
        assert results[0]["price_change_pct"] == pytest.approx(-3.0, rel=1e-4)

    @patch("brokers.market_data.yf.download")
    def test_multiple_tickers_mixed_filtering(self, mock_dl):
        """One ticker passes, one fails."""
        def side_effect(ticker, **kwargs):
            if ticker == "PASS":
                return _make_stock_df(base_close=100.0, last_close=103.0,
                                      base_volume=1_000_000.0, last_volume=2_000_000.0)
            else:  # FAIL
                return _make_stock_df(base_close=100.0, last_close=100.2,
                                      base_volume=1_000_000.0, last_volume=800_000.0)
        mock_dl.side_effect = side_effect

        provider = self._provider()
        results = provider.get_stock_candidates(["PASS", "FAIL"])
        tickers = [r["ticker"] for r in results]
        assert "PASS" in tickers
        assert "FAIL" not in tickers

    @patch("brokers.market_data.yf.download")
    def test_error_per_ticker_skips_and_continues(self, mock_dl):
        """If one ticker raises an exception the rest should still be processed."""
        def side_effect(ticker, **kwargs):
            if ticker == "BROKEN":
                raise RuntimeError("Network error")
            return _make_stock_df(base_close=100.0, last_close=103.0,
                                  base_volume=1_000_000.0, last_volume=2_000_000.0)
        mock_dl.side_effect = side_effect

        provider = self._provider()
        results = provider.get_stock_candidates(["BROKEN", "GOOD"])
        tickers = [r["ticker"] for r in results]
        assert "BROKEN" not in tickers
        assert "GOOD" in tickers

    @patch("brokers.market_data.yf.download")
    def test_candidate_dict_has_required_keys(self, mock_dl):
        """Each returned candidate dict must have the documented keys."""
        mock_dl.return_value = _make_stock_df(
            base_close=100.0, last_close=103.0,
            base_volume=1_000_000.0, last_volume=2_000_000.0,
        )
        provider = self._provider()
        results = provider.get_stock_candidates(["AAPL"])
        assert len(results) == 1
        c = results[0]
        for key in ("ticker", "price", "volume_ratio", "price_change_pct", "indicators"):
            assert key in c, f"Missing key: {key}"
        for ind_key in ("RSI", "MACD_line", "MACD_signal", "SMA20", "SMA50"):
            assert ind_key in c["indicators"], f"Missing indicator: {ind_key}"

    @patch("brokers.market_data.yf.download")
    def test_volume_ratio_computed_correctly(self, mock_dl):
        """volume_ratio = last_volume / mean(prior 20 volumes)."""
        mock_dl.return_value = _make_stock_df(
            n_rows=21,
            base_close=100.0,
            last_close=103.0,
            base_volume=1_000_000.0,
            last_volume=3_000_000.0,  # ratio should be 3.0
        )
        provider = self._provider()
        results = provider.get_stock_candidates(["AAPL"])
        assert len(results) == 1
        assert results[0]["volume_ratio"] == pytest.approx(3.0, rel=1e-4)

    @patch("brokers.market_data.yf.download")
    def test_empty_df_skips_ticker(self, mock_dl):
        """If yfinance returns an empty DataFrame, the ticker is skipped."""
        mock_dl.return_value = pd.DataFrame()
        provider = self._provider()
        results = provider.get_stock_candidates(["EMPTY"])
        assert results == []


# ---------------------------------------------------------------------------
# MarketDataProvider.get_crypto_candidates — filter logic
# ---------------------------------------------------------------------------

class TestGetCryptoCandidatesFiltering:
    """Test filter logic with mocked ccxt.binance."""

    def _provider(self):
        return MarketDataProvider()

    def _mock_exchange(self, ohlcv_data):
        mock_ex = MagicMock()
        mock_ex.fetch_ohlcv.return_value = ohlcv_data
        return mock_ex

    @patch("brokers.market_data.ccxt.binance")
    def test_candidate_passes_both_filters(self, mock_binance):
        mock_binance.return_value = self._mock_exchange(
            _make_ccxt_ohlcv(base_close=30_000.0, last_close=31_500.0,
                             base_volume=5_000.0, last_volume=10_000.0)
        )
        provider = self._provider()
        results = provider.get_crypto_candidates(["BTC/USDT"], min_price_change_pct=1.0, min_volume_ratio=1.0)
        assert len(results) == 1
        assert results[0]["ticker"] == "BTC/USDT"

    @patch("brokers.market_data.ccxt.binance")
    def test_candidate_filtered_by_price_change(self, mock_binance):
        mock_binance.return_value = self._mock_exchange(
            _make_ccxt_ohlcv(base_close=30_000.0, last_close=30_150.0,  # +0.5%
                             base_volume=5_000.0, last_volume=10_000.0)
        )
        provider = self._provider()
        results = provider.get_crypto_candidates(["BTC/USDT"], min_price_change_pct=1.0, min_volume_ratio=1.0)
        assert len(results) == 0

    @patch("brokers.market_data.ccxt.binance")
    def test_candidate_filtered_by_volume_ratio(self, mock_binance):
        mock_binance.return_value = self._mock_exchange(
            _make_ccxt_ohlcv(base_close=30_000.0, last_close=31_500.0,
                             base_volume=5_000.0, last_volume=3_000.0)  # ratio 0.6
        )
        provider = self._provider()
        results = provider.get_crypto_candidates(["ETH/USDT"], min_price_change_pct=1.0, min_volume_ratio=1.0)
        assert len(results) == 0

    @patch("brokers.market_data.ccxt.binance")
    def test_default_symbols_used_when_none_passed(self, mock_binance):
        """When symbols=None, default symbols should be used."""
        mock_ex = MagicMock()
        mock_ex.fetch_ohlcv.return_value = _make_ccxt_ohlcv(
            base_close=30_000.0, last_close=31_500.0,
            base_volume=5_000.0, last_volume=10_000.0,
        )
        mock_binance.return_value = mock_ex

        provider = self._provider()
        provider.get_crypto_candidates(symbols=None, min_price_change_pct=1.0, min_volume_ratio=1.0)

        # fetch_ohlcv should have been called 4 times (4 default symbols)
        assert mock_ex.fetch_ohlcv.call_count == 4

    @patch("brokers.market_data.ccxt.binance")
    def test_error_per_symbol_skips_and_continues(self, mock_binance):
        """If one symbol raises, the rest should still be processed."""
        call_count = {"n": 0}

        def fetch_ohlcv(symbol, **kwargs):
            call_count["n"] += 1
            if symbol == "BROKEN/USDT":
                raise RuntimeError("Exchange error")
            return _make_ccxt_ohlcv(base_close=30_000.0, last_close=31_500.0,
                                    base_volume=5_000.0, last_volume=10_000.0)

        mock_ex = MagicMock()
        mock_ex.fetch_ohlcv.side_effect = fetch_ohlcv
        mock_binance.return_value = mock_ex

        provider = self._provider()
        results = provider.get_crypto_candidates(
            ["BROKEN/USDT", "BTC/USDT"],
            min_price_change_pct=1.0,
            min_volume_ratio=1.0,
        )
        tickers = [r["ticker"] for r in results]
        assert "BROKEN/USDT" not in tickers
        assert "BTC/USDT" in tickers

    @patch("brokers.market_data.ccxt.binance")
    def test_candidate_dict_has_required_keys(self, mock_binance):
        mock_binance.return_value = self._mock_exchange(
            _make_ccxt_ohlcv(base_close=30_000.0, last_close=31_500.0,
                             base_volume=5_000.0, last_volume=10_000.0)
        )
        provider = self._provider()
        results = provider.get_crypto_candidates(["BTC/USDT"])
        assert len(results) == 1
        c = results[0]
        for key in ("ticker", "price", "volume_ratio", "price_change_pct", "indicators"):
            assert key in c
        for ind_key in ("RSI", "MACD_line", "MACD_signal", "SMA20", "SMA50"):
            assert ind_key in c["indicators"]


# ---------------------------------------------------------------------------
# MarketDataProvider.get_historical_prices
# ---------------------------------------------------------------------------

class TestGetHistoricalPrices:
    @patch("brokers.market_data.yf.download")
    def test_stock_returns_dataframe(self, mock_dl):
        mock_dl.return_value = _make_stock_df(n_rows=60)
        provider = MarketDataProvider()
        df = provider.get_historical_prices("AAPL", asset_type="stock", period="60d")
        assert isinstance(df, pd.DataFrame)

    @patch("brokers.market_data.yf.download")
    def test_stock_df_has_ohlcv_columns(self, mock_dl):
        mock_dl.return_value = _make_stock_df(n_rows=60)
        provider = MarketDataProvider()
        df = provider.get_historical_prices("AAPL", asset_type="stock", period="60d")
        for col in ("open", "high", "low", "close", "volume"):
            assert col in df.columns

    @patch("brokers.market_data.ccxt.binance")
    def test_crypto_returns_dataframe(self, mock_binance):
        mock_ex = MagicMock()
        mock_ex.fetch_ohlcv.return_value = _make_ccxt_ohlcv(n_rows=60)
        mock_binance.return_value = mock_ex
        provider = MarketDataProvider()
        df = provider.get_historical_prices("BTC/USDT", asset_type="crypto", period="60d")
        assert isinstance(df, pd.DataFrame)

    @patch("brokers.market_data.ccxt.binance")
    def test_crypto_df_has_ohlcv_columns(self, mock_binance):
        mock_ex = MagicMock()
        mock_ex.fetch_ohlcv.return_value = _make_ccxt_ohlcv(n_rows=60)
        mock_binance.return_value = mock_ex
        provider = MarketDataProvider()
        df = provider.get_historical_prices("BTC/USDT", asset_type="crypto", period="60d")
        for col in ("open", "high", "low", "close", "volume"):
            assert col in df.columns

    def test_invalid_asset_type_raises(self):
        provider = MarketDataProvider()
        with pytest.raises(ValueError, match="asset_type"):
            provider.get_historical_prices("AAPL", asset_type="futures")
