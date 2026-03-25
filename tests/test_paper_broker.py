"""
Unit tests for brokers/paper.py (PaperBroker).

Run with:
    cd /Users/turje87/Desktop/TradeBots && python -m pytest tests/test_paper_broker.py -v
"""

import pytest
from datetime import datetime, timezone

from core.models import TradeRecord
from core.portfolio import Portfolio
from brokers.paper import PaperBroker
from config.settings import MAX_POSITIONS, MAX_POSITION_PCT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_portfolio(capital: float = 100_000.0) -> Portfolio:
    return Portfolio(starting_capital=capital)


def _make_broker(capital: float = 100_000.0) -> PaperBroker:
    return PaperBroker(portfolio=_make_portfolio(capital))


# ---------------------------------------------------------------------------
# PaperBroker.buy()
# ---------------------------------------------------------------------------

class TestPaperBrokerBuy:
    def test_buy_returns_trade_record(self):
        broker = _make_broker()
        record = broker.buy("AAPL", price=100.0, qty=5.0)
        assert isinstance(record, TradeRecord)

    def test_buy_record_fields(self):
        broker = _make_broker()
        record = broker.buy("TSLA", price=250.0, qty=2.0, decision_chain="scanner→orchestrator")
        assert record.ticker == "TSLA"
        assert record.side == "buy"
        assert record.qty == pytest.approx(2.0)
        assert record.price == pytest.approx(250.0)
        assert record.paper is True
        assert record.decision_chain == "scanner→orchestrator"

    def test_buy_deducts_cash_from_portfolio(self):
        portfolio = _make_portfolio(100_000.0)
        broker = PaperBroker(portfolio)
        broker.buy("MSFT", price=200.0, qty=10.0)  # cost = 2_000
        assert portfolio.cash == pytest.approx(98_000.0)

    def test_buy_creates_position_in_portfolio(self):
        portfolio = _make_portfolio(100_000.0)
        broker = PaperBroker(portfolio)
        broker.buy("NVDA", price=500.0, qty=3.0)
        assert "NVDA" in portfolio.positions
        assert portfolio.positions["NVDA"].qty == pytest.approx(3.0)

    def test_buy_record_is_frozen(self):
        broker = _make_broker()
        record = broker.buy("AAPL", price=100.0, qty=1.0)
        with pytest.raises((AttributeError, TypeError)):
            record.price = 999.0  # type: ignore[misc]

    def test_buy_appends_to_trade_history(self):
        portfolio = _make_portfolio()
        broker = PaperBroker(portfolio)
        broker.buy("SPY", price=400.0, qty=1.0)
        assert len(portfolio.trade_history) == 1
        assert portfolio.trade_history[0].side == "buy"

    def test_buy_uses_paper_true(self):
        broker = _make_broker()
        record = broker.buy("AAPL", price=100.0, qty=1.0)
        assert record.paper is True

    def test_buy_timestamp_is_recent(self):
        before = datetime.now(timezone.utc)
        broker = _make_broker()
        record = broker.buy("GOOG", price=150.0, qty=1.0)
        after = datetime.now(timezone.utc)
        assert before <= record.timestamp <= after


# ---------------------------------------------------------------------------
# PaperBroker.buy() — rejected trades
# ---------------------------------------------------------------------------

class TestPaperBrokerBuyRejected:
    def test_buy_raises_when_position_exceeds_max_pct(self):
        """qty * price > MAX_POSITION_PCT of portfolio should raise ValueError."""
        broker = _make_broker(capital=100_000.0)
        # Calculate qty that exceeds MAX_POSITION_PCT limit (20% above)
        over_limit_qty = (MAX_POSITION_PCT * 100_000.0 / 100.0) * 1.2 + 1
        with pytest.raises(ValueError, match="Trade rejected"):
            broker.buy("AAPL", price=100.0, qty=over_limit_qty)

    def test_buy_raises_when_max_positions_reached(self):
        """When MAX_POSITIONS is full, opening a new ticker should raise ValueError."""
        portfolio = _make_portfolio(capital=10_000_000.0)
        broker = PaperBroker(portfolio)
        # Fill all slots with unique tickers
        for i in range(MAX_POSITIONS):
            broker.buy(f"T{i}", price=1.0, qty=1.0)
        # One more brand-new ticker should be rejected
        with pytest.raises(ValueError, match="Trade rejected"):
            broker.buy("BRAND_NEW", price=1.0, qty=1.0)

    def test_buy_raises_on_insufficient_cash(self):
        """Portfolio with tiny capital should raise ValueError for large trades."""
        broker = _make_broker(capital=500.0)
        with pytest.raises(ValueError):
            broker.buy("AAPL", price=100.0, qty=100.0)  # cost = 10,000 > 500 cash

    def test_buy_does_not_mutate_portfolio_when_rejected(self):
        """If a buy is rejected, portfolio state must be unchanged."""
        portfolio = _make_portfolio(100_000.0)
        broker = PaperBroker(portfolio)
        try:
            broker.buy("AAPL", price=100.0, qty=120.0)  # exceeds limit
        except ValueError:
            pass
        assert "AAPL" not in portfolio.positions
        assert portfolio.cash == pytest.approx(100_000.0)
        assert len(portfolio.trade_history) == 0


# ---------------------------------------------------------------------------
# PaperBroker.sell()
# ---------------------------------------------------------------------------

class TestPaperBrokerSell:
    def test_sell_returns_trade_record(self):
        portfolio = _make_portfolio()
        broker = PaperBroker(portfolio)
        broker.buy("AAPL", price=100.0, qty=5.0)
        record = broker.sell("AAPL", price=120.0)
        assert isinstance(record, TradeRecord)

    def test_sell_record_fields(self):
        portfolio = _make_portfolio()
        broker = PaperBroker(portfolio)
        broker.buy("TSLA", price=250.0, qty=3.0)
        record = broker.sell("TSLA", price=300.0)
        assert record.ticker == "TSLA"
        assert record.side == "sell"
        assert record.qty == pytest.approx(3.0)
        assert record.price == pytest.approx(300.0)
        assert record.paper is True

    def test_sell_credits_cash(self):
        portfolio = _make_portfolio(100_000.0)
        broker = PaperBroker(portfolio)
        broker.buy("MSFT", price=200.0, qty=10.0)  # spend 2_000
        cash_after_buy = portfolio.cash
        broker.sell("MSFT", price=250.0)  # credit 10 * 250 = 2_500
        assert portfolio.cash == pytest.approx(cash_after_buy + 2_500.0)

    def test_sell_removes_position(self):
        portfolio = _make_portfolio()
        broker = PaperBroker(portfolio)
        broker.buy("NVDA", price=500.0, qty=2.0)
        broker.sell("NVDA", price=600.0)
        assert "NVDA" not in portfolio.positions

    def test_sell_paper_false_flag(self):
        portfolio = _make_portfolio()
        broker = PaperBroker(portfolio)
        broker.buy("AAPL", price=100.0, qty=1.0)
        record = broker.sell("AAPL", price=110.0, paper=False)
        assert record.paper is False

    def test_sell_appends_to_trade_history(self):
        portfolio = _make_portfolio()
        broker = PaperBroker(portfolio)
        broker.buy("SPY", price=400.0, qty=1.0)
        broker.sell("SPY", price=420.0)
        assert len(portfolio.trade_history) == 2
        assert portfolio.trade_history[-1].side == "sell"

    def test_sell_raises_when_no_position(self):
        broker = _make_broker()
        with pytest.raises(ValueError, match="no open position"):
            broker.sell("FAKE", price=100.0)

    def test_sell_raises_on_unknown_ticker(self):
        portfolio = _make_portfolio()
        broker = PaperBroker(portfolio)
        broker.buy("AAPL", price=100.0, qty=1.0)
        # Selling a different ticker that was never bought
        with pytest.raises(ValueError):
            broker.sell("GOOG", price=150.0)


# ---------------------------------------------------------------------------
# PaperBroker.get_portfolio_value()
# ---------------------------------------------------------------------------

class TestPaperBrokerGetPortfolioValue:
    def test_returns_dict(self):
        broker = _make_broker(50_000.0)
        result = broker.get_portfolio_value()
        assert isinstance(result, dict)

    def test_snapshot_structure(self):
        broker = _make_broker(75_000.0)
        snap = broker.get_portfolio_value()
        assert "cash" in snap
        assert "positions" in snap
        assert "total_value" in snap

    def test_initial_snapshot_values(self):
        broker = _make_broker(80_000.0)
        snap = broker.get_portfolio_value()
        assert snap["cash"] == pytest.approx(80_000.0)
        assert snap["total_value"] == pytest.approx(80_000.0)
        assert snap["positions"] == {}

    def test_snapshot_after_buy(self):
        portfolio = _make_portfolio(100_000.0)
        broker = PaperBroker(portfolio)
        broker.buy("AAPL", price=100.0, qty=5.0)  # cost = 500
        snap = broker.get_portfolio_value()
        assert snap["cash"] == pytest.approx(99_500.0)
        assert "AAPL" in snap["positions"]

    def test_snapshot_after_buy_and_sell(self):
        portfolio = _make_portfolio(100_000.0)
        broker = PaperBroker(portfolio)
        broker.buy("AAPL", price=100.0, qty=5.0)
        broker.sell("AAPL", price=120.0)  # 5 * 120 = 600 returned
        snap = broker.get_portfolio_value()
        assert snap["positions"] == {}
        assert snap["cash"] == pytest.approx(100_100.0)  # 100_000 - 500 + 600

    def test_snapshot_is_independent_copy(self):
        """Mutating the returned dict must not affect portfolio state."""
        portfolio = _make_portfolio(100_000.0)
        broker = PaperBroker(portfolio)
        snap = broker.get_portfolio_value()
        snap["cash"] = 0
        snap["positions"]["FAKE"] = {}
        # Original portfolio unchanged
        assert portfolio.cash == pytest.approx(100_000.0)
        assert "FAKE" not in portfolio.positions
