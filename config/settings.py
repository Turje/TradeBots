"""
Configuration constants for TradeBots trading system.
All defaults can be overridden via environment variables.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# === Timing Configuration ===
SCAN_INTERVAL_MINUTES = int(os.getenv("SCAN_INTERVAL_MINUTES", "30"))
WATCHER_INTERVAL_MINUTES = int(os.getenv("WATCHER_INTERVAL_MINUTES", "5"))
MAX_HOLD_HOURS = int(os.getenv("MAX_HOLD_HOURS", "48"))

# === Position Management ===
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "10"))
MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", "0.10"))  # 10% of portfolio per position
STARTING_CAPITAL = float(os.getenv("STARTING_CAPITAL", "100000"))  # paper money

# === Risk Management ===
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.05"))  # -5%
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.15"))  # +15%

# === Strategy Configuration ===
MIN_STRATEGY_CONFIDENCE = float(os.getenv("MIN_STRATEGY_CONFIDENCE", "0.6"))
STRATEGY_AGREEMENT_THRESHOLD = int(os.getenv("STRATEGY_AGREEMENT_THRESHOLD", "2"))  # out of 3

# === LLM Configuration ===
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")  # default model
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# === Retry Configuration ===
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_BASE_DELAY = float(os.getenv("RETRY_BASE_DELAY", "1.0"))  # seconds, exponential backoff

# === Alpaca Configuration (Optional) ===
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")


def validate_required_keys() -> None:
    """Call once at application startup (not at import time) to verify required env vars are set."""
    if not ANTHROPIC_API_KEY:
        raise EnvironmentError("ANTHROPIC_API_KEY is required but not set. Add it to your .env file.")
