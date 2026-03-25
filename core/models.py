"""
Core data models (dataclasses) for the TradeBots trading system.
These types are imported by every agent in the pipeline.
No Claude API calls — pure Python data structures.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional


@dataclass
class Candidate:
    """A ticker candidate identified by the Scanner agent."""
    ticker: str
    asset_type: Literal["stock", "crypto"]
    price: float
    volume_ratio: float       # current / 20-day average
    price_change_pct: float   # 1-day percentage change
    indicators: dict          # RSI, MACD, etc.
    rationale: str            # Claude's reasoning from Scanner


@dataclass
class StrategySignal:
    """A signal produced by a single strategy agent."""
    ticker: str
    agent_name: str
    action: Literal["BUY", "HOLD", "PASS"]
    confidence: float         # 0.0 – 1.0
    reasoning: str


@dataclass
class TradeDecision:
    """The Orchestrator's synthesised decision after reviewing all strategy signals."""
    ticker: str
    action: Literal["BUY", "HOLD"]
    confidence: float
    signal_summary: str       # Orchestrator's synthesis of agent signals


@dataclass(frozen=True)
class TradeRecord:
    """An immutable record of a trade that was executed."""
    ticker: str
    side: Literal["buy", "sell"]
    qty: float
    price: float
    timestamp: datetime
    paper: bool
    decision_chain: str       # full audit trail


@dataclass
class WatcherDecision:
    """A decision produced by WatcherBot when monitoring an open position."""
    ticker: str
    action: Literal["hold", "sell", "reassess"]
    reasoning: str
    urgency: Literal["normal", "immediate"]  # immediate = execute without delay
