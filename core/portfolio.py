"""
Portfolio — thread-safe portfolio state management.
WatcherBot and ExecutorAgent write to the portfolio concurrently, so all
mutating methods are protected by a threading.Lock.
No Claude API calls — pure Python data structures.
"""

import threading
from dataclasses import dataclass
from datetime import datetime, timezone

from core.models import TradeRecord
from config.settings import MAX_POSITIONS, MAX_POSITION_PCT


@dataclass
class Position:
    """Represents a single open position in the portfolio."""
    ticker: str
    qty: float
    avg_cost: float
    opened_at: datetime


class Portfolio:
    """
    Thread-safe portfolio state.

    Attributes:
        cash:          Available cash balance.
        positions:     Dict mapping ticker -> Position for all open positions.
        trade_history: Chronological list of all executed TradeRecords.
    """

    def __init__(self, starting_capital: float) -> None:
        self.cash: float = starting_capital
        self.positions: dict[str, Position] = {}
        self.trade_history: list[TradeRecord] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Read helpers (still lock-protected for consistency)
    # ------------------------------------------------------------------

    def get_total_value(self) -> float:
        """
        Return total portfolio value: cash + sum of (qty * avg_cost) for all
        open positions.

        Note: avg_cost is used as a price proxy so no live price feed is
        required at snapshot time.
        """
        with self._lock:
            return self._total_value_unlocked()

    def _total_value_unlocked(self) -> float:
        """Must be called while the lock is already held."""
        positions_value = sum(
            pos.qty * pos.avg_cost for pos in self.positions.values()
        )
        return self.cash + positions_value

    def get_snapshot(self) -> dict:
        """
        Return a thread-safe point-in-time snapshot of the portfolio.

        Returns a dict with:
            - cash (float)
            - positions (dict[str, dict]) — each position serialised to a dict
            - total_value (float)
        """
        with self._lock:
            return {
                "cash": self.cash,
                "positions": {
                    ticker: {
                        "ticker": pos.ticker,
                        "qty": pos.qty,
                        "avg_cost": pos.avg_cost,
                        "opened_at": pos.opened_at.isoformat(),
                    }
                    for ticker, pos in self.positions.items()
                },
                "total_value": self._total_value_unlocked(),
            }

    # ------------------------------------------------------------------
    # Validation (lock-free helpers + public lock-acquiring wrapper)
    # ------------------------------------------------------------------

    def _can_buy_unlocked(self, ticker: str, price: float, qty: float) -> bool:
        """
        Core buy-check logic. Must be called while the lock is already held.

        Note: total portfolio value uses avg_cost as a price proxy (no live
        price feed at this layer). The position being checked uses current
        price. This is a known approximation in the paper-trading context.
        """
        total_value = self._total_value_unlocked()

        # --- Rule 1: max open positions ---
        at_capacity = len(self.positions) >= MAX_POSITIONS
        already_in = ticker in self.positions
        if at_capacity and not already_in:
            return False

        # --- Rule 2: max position size as % of portfolio ---
        existing_value = 0.0
        if already_in:
            # Use current price for the ticker being checked (not avg_cost),
            # because the caller supplies the live price at decision time.
            existing_value = self.positions[ticker].qty * price
        new_trade_cost = price * qty
        projected_exposure = existing_value + new_trade_cost

        if total_value <= 0:
            return False

        if projected_exposure / total_value >= MAX_POSITION_PCT:
            return False

        return True

    def can_buy(self, ticker: str, price: float, qty: float) -> bool:
        """
        Return True only if BOTH of the following conditions are met:

        1. The number of open positions is below MAX_POSITIONS (or the ticker
           already has an open position, i.e. we are adding to it).
        2. The cost of the new position (price * qty) does not cause the total
           exposure in that ticker to exceed MAX_POSITION_PCT of total portfolio
           value.

        Note: total portfolio value uses avg_cost as a price proxy (no live
        price feed at this layer). The position being checked uses current
        price. This is a known approximation in the paper-trading context.

        Args:
            ticker: The asset ticker symbol.
            price:  The expected execution price per unit.
            qty:    The number of units to buy.

        Returns:
            True if the trade is permitted by position-management rules.
        """
        with self._lock:
            return self._can_buy_unlocked(ticker, price, qty)

    # ------------------------------------------------------------------
    # Mutations (lock-free helpers + public lock-acquiring wrappers)
    # ------------------------------------------------------------------

    def _open_position_unlocked(self, trade: TradeRecord) -> None:
        """
        Core open-position logic. Must be called while the lock is already held.

        Raises:
            ValueError: If trade.side is not "buy" or cash is insufficient.
        """
        if trade.side != "buy":
            raise ValueError(
                f"open_position expects a 'buy' trade, got '{trade.side}'"
            )

        cost = trade.price * trade.qty

        if cost > self.cash:
            raise ValueError(
                f"Insufficient cash: need {cost:.2f}, have {self.cash:.2f}"
            )

        if trade.ticker in self.positions:
            existing = self.positions[trade.ticker]
            total_qty = existing.qty + trade.qty
            avg_cost = (
                (existing.qty * existing.avg_cost) + (trade.qty * trade.price)
            ) / total_qty
            self.positions[trade.ticker] = Position(
                ticker=trade.ticker,
                qty=total_qty,
                avg_cost=avg_cost,
                opened_at=existing.opened_at,  # keep original open time
            )
        else:
            self.positions[trade.ticker] = Position(
                ticker=trade.ticker,
                qty=trade.qty,
                avg_cost=trade.price,
                opened_at=trade.timestamp,
            )

        self.cash -= cost
        self.trade_history.append(trade)

    def open_position(self, trade: TradeRecord) -> None:
        """
        Record a buy trade and update portfolio state.

        - Deducts trade cost from cash.
        - If the ticker already has an open position the qty and avg_cost are
          updated using a weighted-average cost calculation (position averaging).
        - Appends the TradeRecord to trade_history.

        Args:
            trade: A TradeRecord with side == "buy".

        Raises:
            ValueError: If trade.side is not "buy" or cash is insufficient.
        """
        with self._lock:
            self._open_position_unlocked(trade)

    def validate_and_open(self, trade: TradeRecord) -> bool:
        """
        Atomically validate and open a buy position in a single lock acquisition.

        This prevents the TOCTOU (time-of-check/time-of-use) race condition
        that would exist if can_buy() and open_position() were called separately:
        a concurrent thread could drain cash between the two calls.

        Args:
            trade: A TradeRecord with side == "buy".

        Returns:
            True  if the trade passed the can_buy check and was opened.
            False if the trade was rejected by can_buy (no mutation occurs).

        Raises:
            ValueError: If trade.side is not "buy" or cash is insufficient
                        (the latter can happen even after a passing can_buy
                        check in extreme edge cases — callers should handle it).
        """
        with self._lock:
            if not self._can_buy_unlocked(trade.ticker, trade.price, trade.qty):
                return False
            self._open_position_unlocked(trade)
            return True

    def close_position(self, ticker: str, price: float, paper: bool = True) -> TradeRecord:
        """
        Close the entire open position for a ticker and update portfolio state.

        - Removes the position from `positions`.
        - Credits proceeds to cash.
        - Creates, records, and returns a sell TradeRecord.

        Args:
            ticker: The asset ticker to close.
            price:  The execution price for the sell.
            paper:  Whether this is a paper trade (default True). Pass False
                    for live trading to preserve the correct audit trail.

        Returns:
            The TradeRecord created for this sell.

        Raises:
            KeyError: If there is no open position for the given ticker.
        """
        with self._lock:
            if ticker not in self.positions:
                raise KeyError(f"No open position for ticker '{ticker}'")

            position = self.positions.pop(ticker)
            proceeds = position.qty * price
            self.cash += proceeds

            sell_record = TradeRecord(
                ticker=ticker,
                side="sell",
                qty=position.qty,
                price=price,
                timestamp=datetime.now(timezone.utc),
                paper=paper,
                decision_chain=f"close_position: sold {position.qty} {ticker} @ {price}",
            )
            self.trade_history.append(sell_record)
            return sell_record
