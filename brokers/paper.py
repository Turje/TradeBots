"""
PaperBroker — pure in-memory paper trading broker.
Simulates market orders with no slippage.
Used as fallback when Alpaca is not configured.
No Claude API calls — pure portfolio delegation.
"""

import logging
from datetime import datetime, timezone

from core.models import TradeRecord
from core.portfolio import Portfolio

logger = logging.getLogger(__name__)


class PaperBroker:
    """
    Pure in-memory paper trading broker.
    Simulates market orders with no slippage.
    Used as fallback when Alpaca is not configured.
    """

    def __init__(self, portfolio: Portfolio) -> None:
        self.portfolio = portfolio

    def buy(
        self,
        ticker: str,
        price: float,
        qty: float,
        decision_chain: str = "",
    ) -> TradeRecord:
        """
        Calls portfolio.validate_and_open() with a TradeRecord.
        Returns the TradeRecord on success.
        Raises ValueError if the trade is rejected (position limits, insufficient cash).
        """
        trade = TradeRecord(
            ticker=ticker,
            side="buy",
            qty=qty,
            price=price,
            timestamp=datetime.now(timezone.utc),
            paper=True,
            decision_chain=decision_chain,
        )

        accepted = self.portfolio.validate_and_open(trade)
        if not accepted:
            raise ValueError(
                f"Trade rejected for {ticker}: position limits or insufficient cash "
                f"(price={price}, qty={qty})"
            )

        logger.info("PaperBroker BUY  %s qty=%.4f @ %.4f", ticker, qty, price)
        return trade

    def sell(
        self,
        ticker: str,
        price: float,
        paper: bool = True,
    ) -> TradeRecord:
        """
        Calls portfolio.close_position(ticker, price, paper=paper).
        Returns the sell TradeRecord.
        Raises ValueError if ticker not in portfolio.
        """
        try:
            sell_record = self.portfolio.close_position(ticker, price, paper=paper)
        except KeyError as exc:
            raise ValueError(
                f"Cannot sell {ticker}: no open position in portfolio"
            ) from exc

        logger.info("PaperBroker SELL %s qty=%.4f @ %.4f", ticker, sell_record.qty, price)
        return sell_record

    def get_portfolio_value(self) -> dict:
        """Returns portfolio.get_snapshot()"""
        return self.portfolio.get_snapshot()
