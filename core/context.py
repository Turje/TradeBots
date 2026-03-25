"""
PipelineContext — shared state object that flows through one full scan cycle.
Each agent reads from and writes to this context as it completes its stage.
No Claude API calls — pure Python data structures.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List

from core.models import (
    Candidate,
    StrategySignal,
    TradeDecision,
    TradeRecord,
)


@dataclass
class PipelineContext:
    """
    Carries all intermediate and final data for a single scan cycle.

    NOT thread-safe. PipelineContext follows a sequential pipeline pattern —
    only one agent writes to it at a time, coordinated by PipelineCoordinator.
    The parallel StrategyRunner stage only READS from context; agents return
    new StrategySignal objects instead of writing to context directly.

    Lifecycle:
        1. Scanner       → populates `candidates`
        2. StrategyRunner → populates `strategy_signals` (merged from all agents)
        3. Orchestrator  → populates `trade_decisions`
        4. Executor      → populates `executed_trades`
    """

    scan_id: str                              # UUID for this scan cycle
    scan_timestamp: datetime
    candidates: List[Candidate]              # set by Scanner
    strategy_signals: List[StrategySignal]   # set by StrategyRunner (after merge)
    trade_decisions: List[TradeDecision]     # set by Orchestrator
    executed_trades: List[TradeRecord]       # set by Executor
    portfolio_snapshot: dict                 # copy of portfolio state at scan start

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def new(cls, portfolio_snapshot: dict) -> "PipelineContext":
        """
        Create a fresh PipelineContext for a new scan cycle.

        Args:
            portfolio_snapshot: A dict snapshot of the portfolio at scan start
                                (obtained via Portfolio.get_snapshot()).

        Returns:
            A new PipelineContext with a unique scan_id, current UTC timestamp,
            and all list fields initialised to empty lists.
        """
        return cls(
            scan_id=str(uuid.uuid4()),
            scan_timestamp=datetime.now(timezone.utc),
            candidates=[],
            strategy_signals=[],
            trade_decisions=[],
            executed_trades=[],
            portfolio_snapshot=portfolio_snapshot,
        )
