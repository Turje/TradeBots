"""
MarketDataProvider — unified interface over yfinance (stocks) and ccxt (crypto).
Provides OHLCV data and basic technical indicators for the scanner pipeline.
No Claude API calls — pure data fetching and computation.
"""

import logging
import math
from typing import Optional

import pandas as pd
import ta
import yfinance as yf
import ccxt

logger = logging.getLogger(__name__)

_DEFAULT_CRYPTO_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]


def _compute_indicators(closes: pd.Series) -> dict:
    """
    Compute RSI, MACD line, MACD signal, SMA20, and SMA50 from a closing price Series.
    Returns a dict. NaN values are returned as None for clean serialisation.
    """
    def _safe(val):
        try:
            return None if math.isnan(val) else float(val)
        except (TypeError, ValueError):
            return None

    rsi = _safe(ta.momentum.RSIIndicator(close=closes, window=14).rsi().iloc[-1])
    macd_obj = ta.trend.MACD(close=closes)
    macd_line = _safe(macd_obj.macd().iloc[-1])
    macd_signal = _safe(macd_obj.macd_signal().iloc[-1])
    sma20 = _safe(closes.rolling(20).mean().iloc[-1])
    sma50 = _safe(closes.rolling(50).mean().iloc[-1])

    return {
        "RSI": rsi,
        "MACD_line": macd_line,
        "MACD_signal": macd_signal,
        "SMA20": sma20,
        "SMA50": sma50,
    }


class MarketDataProvider:
    """
    Unified interface over yfinance (S&P 500 stocks) and ccxt (Binance crypto).
    Provides OHLCV data, volume, and basic indicator inputs.
    """

    def __init__(self):
        """Initialize the provider with lazy-loaded ccxt exchange."""
        self._ccxt_binance = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_stock_candidates(
        self,
        tickers: list,
        min_price_change_pct: float = 1.0,
        min_volume_ratio: float = 1.0,
    ) -> list:
        """
        For each ticker, fetch 1d price data from yfinance.
        Filter: |price_change_pct| > min_price_change_pct AND volume_ratio > min_volume_ratio.
        Return list of dicts: {ticker, price, volume_ratio, price_change_pct, indicators}.
        On error per ticker: log warning and skip.
        """
        results = []
        for ticker in tickers:
            try:
                candidate = self._fetch_stock_candidate(ticker)
                if candidate is None:
                    continue
                if (
                    abs(candidate["price_change_pct"]) > min_price_change_pct
                    and candidate["volume_ratio"] > min_volume_ratio
                ):
                    results.append(candidate)
            except Exception as exc:
                logger.warning("Error fetching stock data for %s: %s", ticker, exc)
        return results

    def get_crypto_candidates(
        self,
        symbols: Optional[list] = None,
        min_price_change_pct: float = 1.0,
        min_volume_ratio: float = 1.0,
    ) -> list:
        """
        Fetches OHLCV for each symbol from Binance via ccxt.
        Same filter logic as get_stock_candidates.
        On error: log warning and skip.
        """
        if symbols is None:
            symbols = _DEFAULT_CRYPTO_SYMBOLS

        exchange = self._get_exchange()
        results = []
        for symbol in symbols:
            try:
                candidate = self._fetch_crypto_candidate(exchange, symbol)
                if candidate is None:
                    continue
                if (
                    abs(candidate["price_change_pct"]) > min_price_change_pct
                    and candidate["volume_ratio"] > min_volume_ratio
                ):
                    results.append(candidate)
            except Exception as exc:
                logger.warning("Error fetching crypto data for %s: %s", symbol, exc)
        return results

    def get_historical_prices(
        self,
        ticker: str,
        asset_type: str,
        period: str = "60d",
    ) -> pd.DataFrame:
        """
        Returns a DataFrame with columns: open, high, low, close, volume.
        Used by strategy agents to compute technical indicators.
        asset_type: "stock" or "crypto"
        """
        if asset_type == "stock":
            return self._get_stock_history(ticker, period)
        elif asset_type == "crypto":
            return self._get_crypto_history(ticker, period)
        else:
            raise ValueError(f"Unknown asset_type '{asset_type}'. Must be 'stock' or 'crypto'.")

    # ------------------------------------------------------------------
    # Private helpers — stocks
    # ------------------------------------------------------------------

    def _fetch_stock_candidate(self, ticker: str) -> Optional[dict]:
        """
        Download 21 days of daily data for a single stock ticker via yfinance.
        Returns a candidate dict or None if data is insufficient.
        """
        df = yf.download(ticker, period="21d", interval="1d", progress=False, auto_adjust=True)

        if df is None or len(df) < 2:
            logger.warning("Insufficient data for stock %s (got %d rows)", ticker, 0 if df is None else len(df))
            return None

        # Flatten MultiIndex columns if present (yfinance sometimes returns them)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [c.lower() for c in df.columns]

        closes = df["close"].dropna()
        volumes = df["volume"].dropna()

        if len(closes) < 2 or len(volumes) < 2:
            logger.warning("Not enough valid rows for stock %s", ticker)
            return None

        last_close = float(closes.iloc[-1])
        prev_close = float(closes.iloc[-2])
        price_change_pct = ((last_close - prev_close) / prev_close) * 100.0

        last_volume = float(volumes.iloc[-1])
        avg_volume_20d = float(volumes.iloc[-21:-1].mean()) if len(volumes) >= 21 else float(volumes.iloc[:-1].mean())
        volume_ratio = last_volume / avg_volume_20d if avg_volume_20d > 0 else 0.0

        indicators = _compute_indicators(closes)

        return {
            "ticker": ticker,
            "price": last_close,
            "volume_ratio": volume_ratio,
            "price_change_pct": price_change_pct,
            "indicators": indicators,
        }

    def _get_stock_history(self, ticker: str, period: str) -> pd.DataFrame:
        """Download stock OHLCV history and return a normalised DataFrame."""
        df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].dropna()

        if df.empty or not all(col in df.columns for col in ["open", "high", "low", "close", "volume"]):
            raise ValueError(f"No price data available for {ticker} (period={period})")

        return df

    # ------------------------------------------------------------------
    # Private helpers — crypto
    # ------------------------------------------------------------------

    def _fetch_crypto_candidate(self, exchange, symbol: str) -> Optional[dict]:
        """
        Fetch 21 days of daily OHLCV for a single crypto symbol via ccxt.
        Returns a candidate dict or None if data is insufficient.
        """
        raw = exchange.fetch_ohlcv(symbol, timeframe="1d", limit=21)

        if not raw or len(raw) < 2:
            logger.warning("Insufficient data for crypto %s", symbol)
            return None

        # ccxt returns: [timestamp, open, high, low, close, volume]
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp")

        closes = df["close"].dropna().astype(float)
        volumes = df["volume"].dropna().astype(float)

        if len(closes) < 2 or len(volumes) < 2:
            logger.warning("Not enough valid rows for crypto %s", symbol)
            return None

        last_close = float(closes.iloc[-1])
        prev_close = float(closes.iloc[-2])
        price_change_pct = ((last_close - prev_close) / prev_close) * 100.0

        last_volume = float(volumes.iloc[-1])
        avg_volume_20d = float(volumes.iloc[-21:-1].mean()) if len(volumes) >= 21 else float(volumes.iloc[:-1].mean())
        volume_ratio = last_volume / avg_volume_20d if avg_volume_20d > 0 else 0.0

        indicators = _compute_indicators(closes)

        return {
            "ticker": symbol,
            "price": last_close,
            "volume_ratio": volume_ratio,
            "price_change_pct": price_change_pct,
            "indicators": indicators,
        }

    def _get_crypto_history(self, ticker: str, period: str) -> pd.DataFrame:
        """
        Fetch crypto OHLCV history via ccxt and return a normalised DataFrame.
        period is parsed as a number of days (e.g. "60d" → 60 days).
        """
        days = self._parse_period_days(period)
        exchange = self._get_exchange()
        raw = exchange.fetch_ohlcv(ticker, timeframe="1d", limit=days)

        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp")
        df = df[["open", "high", "low", "close", "volume"]].dropna()

        if df.empty:
            raise ValueError(f"No price data available for {ticker} (period={period})")

        return df

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _get_exchange(self):
        """Get or lazily initialize the ccxt Binance exchange instance."""
        if self._ccxt_binance is None:
            self._ccxt_binance = ccxt.binance()
        return self._ccxt_binance

    @staticmethod
    def _parse_period_days(period: str) -> int:
        """Convert a period string like '60d' to an integer number of days."""
        period = period.strip().lower()
        if period.endswith("d"):
            return int(period[:-1])
        elif period.endswith("mo"):
            return int(period[:-2]) * 30
        elif period.endswith("y"):
            return int(period[:-1]) * 365
        else:
            # Fallback: try to parse as integer directly
            return int(period)
