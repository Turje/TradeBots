"""
Unit tests for core/models.py, core/context.py, and core/portfolio.py.

Run with:
    cd /Users/turje87/Desktop/TradeBots && python -m pytest tests/test_models.py -v
"""

import time
import threading
import uuid
from datetime import datetime, timezone

import pytest

from core.models import (
    Candidate,
    StrategySignal,
    TradeDecision,
    TradeRecord,
    WatcherDecision,
)
from core.context import PipelineContext
from core.portfolio import Portfolio, Position
from config.settings import MAX_POSITIONS, MAX_POSITION_PCT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_buy_record(
    ticker: str = "AAPL",
    qty: float = 10.0,
    price: float = 100.0,
    decision_chain: str = "test",
) -> TradeRecord:
    return TradeRecord(
        ticker=ticker,
        side="buy",
        qty=qty,
        price=price,
        timestamp=datetime.now(timezone.utc),
        paper=True,
        decision_chain=decision_chain,
    )


def _portfolio_with_capital(capital: float = 100_000.0) -> Portfolio:
    return Portfolio(starting_capital=capital)


# ---------------------------------------------------------------------------
# TradeRecord / StrategySignal / TradeDecision / WatcherDecision / Candidate
# ---------------------------------------------------------------------------

class TestDataclassInstantiation:
    def test_candidate(self):
        c = Candidate(
            ticker="TSLA",
            asset_type="stock",
            price=250.0,
            volume_ratio=1.5,
            price_change_pct=3.2,
            indicators={"RSI": 65, "MACD": 0.5},
            rationale="High volume breakout",
        )
        assert c.ticker == "TSLA"
        assert c.asset_type == "stock"
        assert c.indicators["RSI"] == 65

    def test_strategy_signal(self):
        s = StrategySignal(
            ticker="TSLA",
            agent_name="MomentumAgent",
            action="BUY",
            confidence=0.85,
            reasoning="Strong momentum",
        )
        assert s.action == "BUY"
        assert 0.0 <= s.confidence <= 1.0

    def test_trade_decision(self):
        td = TradeDecision(
            ticker="TSLA",
            action="BUY",
            confidence=0.9,
            signal_summary="3/3 agents agree",
        )
        assert td.action == "BUY"

    def test_trade_record(self):
        tr = _make_buy_record()
        assert tr.side == "buy"
        assert tr.paper is True

    def test_trade_record_is_immutable(self):
        """TradeRecord must be frozen — attribute assignment must raise."""
        tr = _make_buy_record()
        with pytest.raises((AttributeError, TypeError)):
            tr.price = 999.0  # type: ignore[misc]

    def test_watcher_decision(self):
        wd = WatcherDecision(
            ticker="TSLA",
            action="sell",
            reasoning="Stop-loss triggered",
            urgency="immediate",
        )
        assert wd.urgency == "immediate"


# ---------------------------------------------------------------------------
# PipelineContext
# ---------------------------------------------------------------------------

class TestPipelineContext:
    def test_new_creates_valid_uuid(self):
        ctx = PipelineContext.new(portfolio_snapshot={})
        # Must be a valid UUID string
        parsed = uuid.UUID(ctx.scan_id)
        assert str(parsed) == ctx.scan_id

    def test_new_creates_unique_scan_ids(self):
        ctx1 = PipelineContext.new(portfolio_snapshot={})
        ctx2 = PipelineContext.new(portfolio_snapshot={})
        assert ctx1.scan_id != ctx2.scan_id

    def test_new_sets_timestamp(self):
        before = datetime.now(timezone.utc)
        ctx = PipelineContext.new(portfolio_snapshot={})
        after = datetime.now(timezone.utc)
        assert before <= ctx.scan_timestamp <= after

    def test_new_empty_lists(self):
        ctx = PipelineContext.new(portfolio_snapshot={"cash": 100_000})
        assert ctx.candidates == []
        assert ctx.strategy_signals == []
        assert ctx.trade_decisions == []
        assert ctx.executed_trades == []

    def test_new_stores_portfolio_snapshot(self):
        snapshot = {"cash": 50_000, "positions": {}, "total_value": 50_000}
        ctx = PipelineContext.new(portfolio_snapshot=snapshot)
        assert ctx.portfolio_snapshot == snapshot

    def test_lists_are_independent_between_instances(self):
        ctx1 = PipelineContext.new(portfolio_snapshot={})
        ctx2 = PipelineContext.new(portfolio_snapshot={})
        ctx1.candidates.append(
            Candidate("X", "stock", 1.0, 1.0, 0.0, {}, "test")
        )
        assert len(ctx2.candidates) == 0


# ---------------------------------------------------------------------------
# Portfolio.can_buy()
# ---------------------------------------------------------------------------

class TestPortfolioCanBuy:
    def test_allows_buy_within_limits(self):
        p = _portfolio_with_capital(100_000)
        # 5 % of 100_000 = 5_000 — within 10 % MAX_POSITION_PCT
        assert p.can_buy("AAPL", price=100.0, qty=50.0) is True

    def test_rejects_when_position_exceeds_max_pct(self):
        p = _portfolio_with_capital(100_000)
        # 12 % of 100_000 = 12_000 — exceeds 10 % MAX_POSITION_PCT
        assert p.can_buy("AAPL", price=100.0, qty=120.0) is False

    def test_rejects_when_max_positions_reached(self):
        p = _portfolio_with_capital(10_000_000)  # large capital so pct limit won't bite
        # Fill up to MAX_POSITIONS
        for i in range(MAX_POSITIONS):
            trade = _make_buy_record(ticker=f"T{i}", qty=1.0, price=1.0)
            p.open_position(trade)
        # Adding a brand-new ticker should be rejected
        assert p.can_buy("BRAND_NEW", price=1.0, qty=1.0) is False

    def test_allows_adding_to_existing_position_when_at_capacity(self):
        """When at MAX_POSITIONS, adding to an already-open position should be
        allowed (provided the size limit is respected)."""
        p = _portfolio_with_capital(10_000_000)
        for i in range(MAX_POSITIONS):
            trade = _make_buy_record(ticker=f"T{i}", qty=1.0, price=1.0)
            p.open_position(trade)
        # Adding to T0 (already open) — small qty so pct limit won't bite
        assert p.can_buy("T0", price=1.0, qty=1.0) is True

    def test_rejects_when_total_value_is_zero(self):
        p = Portfolio(starting_capital=0.0)
        assert p.can_buy("AAPL", price=100.0, qty=1.0) is False

    def test_boundary_exactly_at_max_pct_is_rejected(self):
        """Exactly at the limit should be rejected (>= MAX_POSITION_PCT)."""
        p = _portfolio_with_capital(100_000)
        # MAX_POSITION_PCT = 0.10 → cost of 10_000 is exactly 10 % of 100_000 → rejected
        exact_boundary_qty = (MAX_POSITION_PCT * 100_000) / 100.0  # 100 units @ $100
        assert p.can_buy("AAPL", price=100.0, qty=exact_boundary_qty) is False

    def test_one_unit_below_boundary_is_allowed(self):
        """One unit below the limit must be allowed."""
        p = _portfolio_with_capital(100_000)
        # 99 units @ $100 = $9_900 = 9.9 % < 10 % → allowed
        just_under_qty = (MAX_POSITION_PCT * 100_000) / 100.0 - 1.0
        assert p.can_buy("AAPL", price=100.0, qty=just_under_qty) is True


# ---------------------------------------------------------------------------
# Portfolio.open_position() and close_position()
# ---------------------------------------------------------------------------

class TestPortfolioOpenClose:
    def test_open_position_deducts_cash(self):
        p = _portfolio_with_capital(100_000)
        trade = _make_buy_record(qty=10.0, price=200.0)  # cost = 2_000
        p.open_position(trade)
        assert p.cash == pytest.approx(98_000.0)

    def test_open_position_creates_position(self):
        p = _portfolio_with_capital(100_000)
        trade = _make_buy_record(ticker="AAPL", qty=5.0, price=150.0)
        p.open_position(trade)
        assert "AAPL" in p.positions
        pos = p.positions["AAPL"]
        assert pos.qty == pytest.approx(5.0)
        assert pos.avg_cost == pytest.approx(150.0)

    def test_open_position_appends_to_history(self):
        p = _portfolio_with_capital(100_000)
        trade = _make_buy_record()
        p.open_position(trade)
        assert len(p.trade_history) == 1
        assert p.trade_history[0] is trade

    def test_open_position_averaging(self):
        """Buying the same ticker twice averages the cost."""
        p = _portfolio_with_capital(100_000)
        t1 = _make_buy_record(ticker="AAPL", qty=10.0, price=100.0)
        t2 = _make_buy_record(ticker="AAPL", qty=10.0, price=200.0)
        p.open_position(t1)
        p.open_position(t2)
        pos = p.positions["AAPL"]
        assert pos.qty == pytest.approx(20.0)
        assert pos.avg_cost == pytest.approx(150.0)

    def test_open_position_raises_on_sell_record(self):
        p = _portfolio_with_capital(100_000)
        sell = TradeRecord(
            ticker="AAPL", side="sell", qty=1.0, price=100.0,
            timestamp=datetime.now(timezone.utc), paper=True, decision_chain="bad",
        )
        with pytest.raises(ValueError, match="'buy'"):
            p.open_position(sell)

    def test_open_position_raises_on_insufficient_cash(self):
        p = _portfolio_with_capital(500)
        trade = _make_buy_record(qty=10.0, price=100.0)  # cost = 1_000 > 500
        with pytest.raises(ValueError, match="Insufficient cash"):
            p.open_position(trade)

    def test_close_position_credits_cash(self):
        p = _portfolio_with_capital(100_000)
        trade = _make_buy_record(ticker="MSFT", qty=10.0, price=300.0)
        p.open_position(trade)
        cash_after_buy = p.cash
        p.close_position("MSFT", price=350.0)
        assert p.cash == pytest.approx(cash_after_buy + 10.0 * 350.0)

    def test_close_position_removes_from_positions(self):
        p = _portfolio_with_capital(100_000)
        p.open_position(_make_buy_record(ticker="GOOG", qty=2.0, price=2000.0))
        p.close_position("GOOG", price=2100.0)
        assert "GOOG" not in p.positions

    def test_close_position_returns_sell_record(self):
        p = _portfolio_with_capital(100_000)
        p.open_position(_make_buy_record(ticker="AMZN", qty=5.0, price=180.0))
        sell = p.close_position("AMZN", price=200.0)
        assert sell.side == "sell"
        assert sell.ticker == "AMZN"
        assert sell.qty == pytest.approx(5.0)
        assert sell.price == pytest.approx(200.0)

    def test_close_position_appends_sell_to_history(self):
        p = _portfolio_with_capital(100_000)
        p.open_position(_make_buy_record(ticker="NVDA", qty=3.0, price=500.0))
        p.close_position("NVDA", price=600.0)
        assert len(p.trade_history) == 2
        assert p.trade_history[-1].side == "sell"

    def test_close_position_raises_when_no_position(self):
        p = _portfolio_with_capital(100_000)
        with pytest.raises(KeyError, match="FAKE"):
            p.close_position("FAKE", price=100.0)

    def test_close_position_default_paper_is_true(self):
        """close_position without paper arg must produce a paper=True record."""
        p = _portfolio_with_capital(100_000)
        p.open_position(_make_buy_record(ticker="AAPL", qty=1.0, price=100.0))
        sell = p.close_position("AAPL", price=110.0)
        assert sell.paper is True

    def test_close_position_paper_false_for_live_trading(self):
        """close_position with paper=False must produce a paper=False record."""
        p = _portfolio_with_capital(100_000)
        p.open_position(_make_buy_record(ticker="AAPL", qty=1.0, price=100.0))
        sell = p.close_position("AAPL", price=110.0, paper=False)
        assert sell.paper is False


# ---------------------------------------------------------------------------
# Portfolio.validate_and_open()
# ---------------------------------------------------------------------------

class TestPortfolioValidateAndOpen:
    def test_returns_true_and_opens_valid_trade(self):
        p = _portfolio_with_capital(100_000)
        trade = _make_buy_record(ticker="AAPL", qty=5.0, price=100.0)
        result = p.validate_and_open(trade)
        assert result is True
        assert "AAPL" in p.positions
        assert p.cash == pytest.approx(99_500.0)

    def test_returns_false_and_does_not_mutate_when_rejected(self):
        p = _portfolio_with_capital(100_000)
        # qty=120 @ price=100 = 12% > MAX_POSITION_PCT (10%)
        trade = _make_buy_record(ticker="AAPL", qty=120.0, price=100.0)
        result = p.validate_and_open(trade)
        assert result is False
        assert "AAPL" not in p.positions
        assert p.cash == pytest.approx(100_000.0)


# ---------------------------------------------------------------------------
# Portfolio.get_snapshot()
# ---------------------------------------------------------------------------

class TestPortfolioGetSnapshot:
    def test_snapshot_structure_empty(self):
        p = _portfolio_with_capital(50_000)
        snap = p.get_snapshot()
        assert "cash" in snap
        assert "positions" in snap
        assert "total_value" in snap

    def test_snapshot_cash_equals_starting_capital_when_empty(self):
        p = _portfolio_with_capital(75_000)
        snap = p.get_snapshot()
        assert snap["cash"] == pytest.approx(75_000.0)

    def test_snapshot_total_value_empty(self):
        p = _portfolio_with_capital(80_000)
        snap = p.get_snapshot()
        assert snap["total_value"] == pytest.approx(80_000.0)

    def test_snapshot_reflects_open_positions(self):
        p = _portfolio_with_capital(100_000)
        p.open_position(_make_buy_record(ticker="SPY", qty=10.0, price=400.0))
        snap = p.get_snapshot()
        assert "SPY" in snap["positions"]
        pos = snap["positions"]["SPY"]
        assert pos["qty"] == pytest.approx(10.0)
        assert pos["avg_cost"] == pytest.approx(400.0)

    def test_snapshot_total_value_with_position(self):
        p = _portfolio_with_capital(100_000)
        p.open_position(_make_buy_record(ticker="QQQ", qty=5.0, price=400.0))
        # cash = 100_000 - 2_000 = 98_000; position value = 5 * 400 = 2_000; total = 100_000
        snap = p.get_snapshot()
        assert snap["total_value"] == pytest.approx(100_000.0)

    def test_snapshot_positions_are_dicts_not_dataclasses(self):
        p = _portfolio_with_capital(100_000)
        p.open_position(_make_buy_record(ticker="BTC", qty=1.0, price=50_000.0))
        snap = p.get_snapshot()
        pos = snap["positions"]["BTC"]
        assert isinstance(pos, dict)
        assert "ticker" in pos
        assert "opened_at" in pos  # serialised as ISO string

    def test_snapshot_is_independent_copy(self):
        """Mutating the snapshot dict must not affect portfolio state."""
        p = _portfolio_with_capital(100_000)
        snap = p.get_snapshot()
        snap["cash"] = 0
        snap["positions"]["FAKE"] = {}
        # Original portfolio is unchanged
        assert p.cash == pytest.approx(100_000.0)
        assert "FAKE" not in p.positions


# ---------------------------------------------------------------------------
# Thread-safety smoke tests
# ---------------------------------------------------------------------------

class TestPortfolioThreadSafety:
    def test_concurrent_open_and_snapshot(self):
        """Multiple threads opening positions simultaneously must not corrupt state."""
        p = _portfolio_with_capital(10_000_000)
        errors = []

        def buy(ticker: str) -> None:
            try:
                trade = _make_buy_record(ticker=ticker, qty=1.0, price=100.0)
                p.open_position(trade)
            except Exception as exc:
                errors.append(exc)

        tickers = [f"T{i}" for i in range(min(MAX_POSITIONS, 10))]
        threads = [threading.Thread(target=buy, args=(t,)) for t in tickers]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert errors == [], f"Thread errors: {errors}"
        assert len(p.positions) == len(tickers)
        # Cash must equal starting_capital - (num_positions * 100)
        expected_cash = 10_000_000 - len(tickers) * 100.0
        assert p.cash == pytest.approx(expected_cash)

    def test_concurrent_same_ticker_position_averaging(self):
        """5 threads buying the same ticker simultaneously must not lose qty or cash."""
        num_threads = 5
        per_thread_qty = 2.0
        price = 100.0
        starting_capital = 10_000_000.0

        p = Portfolio(starting_capital=starting_capital)
        errors = []

        def buy_spy() -> None:
            try:
                trade = _make_buy_record(ticker="SPY", qty=per_thread_qty, price=price)
                p.open_position(trade)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=buy_spy) for _ in range(num_threads)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert errors == [], f"Thread errors: {errors}"

        expected_total_qty = num_threads * per_thread_qty
        assert p.positions["SPY"].qty == pytest.approx(expected_total_qty)

        expected_cash = starting_capital - (num_threads * per_thread_qty * price)
        assert p.cash == pytest.approx(expected_cash)
