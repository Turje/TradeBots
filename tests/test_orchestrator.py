"""
Unit tests for agents/orchestrator.py (MetaOrchestrator).

All Claude API calls are mocked — no network I/O occurs.

Run with:
    cd /Users/turje87/Desktop/TradeBots && python -m pytest tests/test_orchestrator.py -v
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from core.models import StrategySignal, TradeDecision
from core.context import PipelineContext
from core.portfolio import Portfolio


# ---------------------------------------------------------------------------
# Helpers — synthetic test data
# ---------------------------------------------------------------------------

def _make_context() -> PipelineContext:
    return PipelineContext.new(portfolio_snapshot={"cash": 100_000.0, "positions": {}})


def _make_portfolio() -> Portfolio:
    return Portfolio(starting_capital=100_000.0)


def _make_signal(
    ticker: str,
    agent_name: str,
    action: str = "BUY",
    confidence: float = 0.75,
    reasoning: str = "Strong signal",
) -> StrategySignal:
    return StrategySignal(
        ticker=ticker,
        agent_name=agent_name,
        action=action,
        confidence=confidence,
        reasoning=reasoning,
    )


def _claude_decision_response(items: list) -> str:
    return json.dumps(items)


def _valid_claude_response(ticker: str = "AAPL") -> str:
    return _claude_decision_response([
        {
            "ticker": ticker,
            "action": "BUY",
            "confidence": 0.75,
            "signal_summary": "2/3 agents agree BUY",
        }
    ])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def orchestrator():
    with patch("agents.orchestrator.ClaudeClient") as mock_cls:
        mock_claude = MagicMock()
        mock_cls.return_value = mock_claude
        from agents.orchestrator import MetaOrchestrator
        orch = MetaOrchestrator(portfolio=_make_portfolio())
        orch._mock_claude = mock_claude
        yield orch


# ---------------------------------------------------------------------------
# Test: Happy path — Claude returns BUY decisions
# ---------------------------------------------------------------------------

class TestMetaOrchestratorHappyPath:
    def test_returns_trade_decision_objects(self, orchestrator):
        """decide() should return a list of TradeDecision instances."""
        signals = [
            _make_signal("AAPL", "technical", "BUY", 0.8),
            _make_signal("AAPL", "momentum", "BUY", 0.7),
            _make_signal("AAPL", "mean_reversion", "HOLD", 0.5),
        ]
        context = _make_context()

        orchestrator._mock_claude.call.return_value = _valid_claude_response("AAPL")

        result = orchestrator.decide(signals, context)

        assert len(result) == 1
        assert isinstance(result[0], TradeDecision)

    def test_decision_fields_populated(self, orchestrator):
        """TradeDecision fields should match Claude's response."""
        signals = [
            _make_signal("AAPL", "technical", "BUY", 0.8),
            _make_signal("AAPL", "momentum", "BUY", 0.7),
            _make_signal("AAPL", "mean_reversion", "HOLD", 0.5),
        ]
        context = _make_context()

        orchestrator._mock_claude.call.return_value = _claude_decision_response([
            {
                "ticker": "AAPL",
                "action": "BUY",
                "confidence": 0.8,
                "signal_summary": "Strong consensus",
            }
        ])

        result = orchestrator.decide(signals, context)

        decision = result[0]
        assert decision.ticker == "AAPL"
        assert decision.action == "BUY"
        assert decision.confidence == pytest.approx(0.8)
        assert decision.signal_summary == "Strong consensus"

    def test_multiple_tickers_all_forwarded_to_claude(self, orchestrator):
        """Signals for multiple tickers should all be sent to Claude."""
        signals = [
            _make_signal("AAPL", "technical", "BUY", 0.8),
            _make_signal("MSFT", "technical", "BUY", 0.7),
        ]
        context = _make_context()

        orchestrator._mock_claude.call.return_value = _claude_decision_response([
            {"ticker": "AAPL", "action": "BUY", "confidence": 0.8, "signal_summary": "OK"},
            {"ticker": "MSFT", "action": "BUY", "confidence": 0.7, "signal_summary": "OK"},
        ])

        result = orchestrator.decide(signals, context)

        assert len(result) == 2
        tickers = {d.ticker for d in result}
        assert tickers == {"AAPL", "MSFT"}

    def test_empty_signals_returns_empty_list(self, orchestrator):
        """decide() with empty signals returns [] without calling Claude."""
        context = _make_context()

        result = orchestrator.decide([], context)

        assert result == []
        orchestrator._mock_claude.call.assert_not_called()


# ---------------------------------------------------------------------------
# Test: Pre-filter logic — 2+ PASS agents skip Claude
# ---------------------------------------------------------------------------

class TestMetaOrchestratorPrefilter:
    def test_two_pass_signals_yield_hold_without_claude(self, orchestrator):
        """Tickers with 2+ PASS signals return HOLD immediately, skip Claude."""
        signals = [
            _make_signal("AAPL", "technical", "PASS", 0.0),
            _make_signal("AAPL", "momentum", "PASS", 0.0),
            _make_signal("AAPL", "mean_reversion", "BUY", 0.7),
        ]
        context = _make_context()

        result = orchestrator.decide(signals, context)

        orchestrator._mock_claude.call.assert_not_called()
        assert len(result) == 1
        assert result[0].action == "HOLD"
        assert result[0].ticker == "AAPL"

    def test_three_pass_signals_yield_hold(self, orchestrator):
        """3 PASS signals still gets pre-filtered as HOLD."""
        signals = [
            _make_signal("AAPL", "technical", "PASS", 0.0),
            _make_signal("AAPL", "momentum", "PASS", 0.0),
            _make_signal("AAPL", "mean_reversion", "PASS", 0.0),
        ]
        context = _make_context()

        result = orchestrator.decide(signals, context)

        orchestrator._mock_claude.call.assert_not_called()
        assert result[0].action == "HOLD"

    def test_one_pass_signal_not_filtered(self, orchestrator):
        """Only 1 PASS signal: ticker still goes to Claude (not filtered)."""
        signals = [
            _make_signal("AAPL", "technical", "PASS", 0.0),
            _make_signal("AAPL", "momentum", "BUY", 0.7),
            _make_signal("AAPL", "mean_reversion", "BUY", 0.8),
        ]
        context = _make_context()

        orchestrator._mock_claude.call.return_value = _valid_claude_response("AAPL")

        orchestrator.decide(signals, context)

        orchestrator._mock_claude.call.assert_called_once()

    def test_mixed_tickers_partially_filtered(self, orchestrator):
        """One ticker filtered (2+ PASS), another goes to Claude."""
        signals = [
            # AAPL — 2 PASS → pre-filtered
            _make_signal("AAPL", "technical", "PASS", 0.0),
            _make_signal("AAPL", "momentum", "PASS", 0.0),
            _make_signal("AAPL", "mean_reversion", "BUY", 0.7),
            # MSFT — 0 PASS → goes to Claude
            _make_signal("MSFT", "technical", "BUY", 0.8),
            _make_signal("MSFT", "momentum", "BUY", 0.7),
            _make_signal("MSFT", "mean_reversion", "HOLD", 0.5),
        ]
        context = _make_context()

        orchestrator._mock_claude.call.return_value = _claude_decision_response([
            {"ticker": "MSFT", "action": "BUY", "confidence": 0.75, "signal_summary": "OK"},
        ])

        result = orchestrator.decide(signals, context)

        # AAPL → HOLD (pre-filtered), MSFT → BUY (from Claude)
        assert len(result) == 2
        aapl_decision = next(d for d in result if d.ticker == "AAPL")
        msft_decision = next(d for d in result if d.ticker == "MSFT")
        assert aapl_decision.action == "HOLD"
        assert msft_decision.action == "BUY"


# ---------------------------------------------------------------------------
# Test: Rolling memory updated after decisions
# ---------------------------------------------------------------------------

class TestMetaOrchestratorMemory:
    def test_memory_updated_after_buy_decision(self, orchestrator):
        """After a BUY decision, _decision_memory should have one entry."""
        signals = [_make_signal("AAPL", "technical", "BUY", 0.8)]
        context = _make_context()

        orchestrator._mock_claude.call.return_value = _valid_claude_response("AAPL")

        orchestrator.decide(signals, context)

        assert len(orchestrator._decision_memory) == 1
        entry = orchestrator._decision_memory[0]
        assert entry["ticker"] == "AAPL"
        assert entry["action"] == "BUY"

    def test_memory_capped_at_ten(self, orchestrator):
        """Rolling memory should not exceed 10 entries."""
        context = _make_context()

        for i in range(15):
            ticker = f"T{i:03d}"
            signals = [_make_signal(ticker, "technical", "BUY", 0.7)]
            orchestrator._mock_claude.call.return_value = _valid_claude_response(ticker)
            orchestrator.decide(signals, context)

        assert len(orchestrator._decision_memory) == 10

    def test_hold_decisions_also_stored(self, orchestrator):
        """Pre-filtered HOLD decisions are stored in memory too."""
        signals = [
            _make_signal("AAPL", "technical", "PASS", 0.0),
            _make_signal("AAPL", "momentum", "PASS", 0.0),
            _make_signal("AAPL", "mean_reversion", "HOLD", 0.5),
        ]
        context = _make_context()

        orchestrator.decide(signals, context)

        assert len(orchestrator._decision_memory) == 1
        assert orchestrator._decision_memory[0]["ticker"] == "AAPL"
        assert orchestrator._decision_memory[0]["action"] == "HOLD"


# ---------------------------------------------------------------------------
# Test: Claude exception handling
# ---------------------------------------------------------------------------

class TestMetaOrchestratorClaudeException:
    def test_returns_empty_on_claude_exception(self, orchestrator):
        """If ClaudeClient.call raises, decide() returns []."""
        signals = [_make_signal("AAPL", "technical", "BUY", 0.8)]
        context = _make_context()

        orchestrator._mock_claude.call.side_effect = RuntimeError("API error")

        result = orchestrator.decide(signals, context)

        assert result == []

    def test_still_returns_prefiltered_on_claude_exception(self, orchestrator):
        """Pre-filtered HOLD decisions are returned even if Claude errors for other tickers."""
        signals = [
            # AAPL — 2 PASS → pre-filtered (should still be returned)
            _make_signal("AAPL", "technical", "PASS", 0.0),
            _make_signal("AAPL", "momentum", "PASS", 0.0),
            _make_signal("AAPL", "mean_reversion", "BUY", 0.7),
            # MSFT — goes to Claude, which will error
            _make_signal("MSFT", "technical", "BUY", 0.8),
            _make_signal("MSFT", "momentum", "BUY", 0.7),
        ]
        context = _make_context()

        orchestrator._mock_claude.call.side_effect = RuntimeError("Network error")

        result = orchestrator.decide(signals, context)

        # AAPL HOLD is still returned; MSFT from Claude fails silently
        assert any(d.ticker == "AAPL" and d.action == "HOLD" for d in result)


# ---------------------------------------------------------------------------
# Test: JSON parse failure
# ---------------------------------------------------------------------------

class TestMetaOrchestratorJsonParseFailure:
    def test_returns_empty_on_invalid_json(self, orchestrator):
        """If Claude returns non-JSON, decide() returns no claude-derived decisions."""
        signals = [_make_signal("AAPL", "technical", "BUY", 0.8)]
        context = _make_context()

        orchestrator._mock_claude.call.return_value = "This is not JSON."

        result = orchestrator.decide(signals, context)

        assert result == []

    def test_returns_empty_on_non_array_json(self, orchestrator):
        """If Claude returns a JSON object (not array), no decisions returned."""
        signals = [_make_signal("AAPL", "technical", "BUY", 0.8)]
        context = _make_context()

        orchestrator._mock_claude.call.return_value = json.dumps({"error": "bad format"})

        result = orchestrator.decide(signals, context)

        assert result == []

    def test_strips_markdown_fences(self, orchestrator):
        """Claude response wrapped in ```json ... ``` fences should parse correctly."""
        signals = [_make_signal("AAPL", "technical", "BUY", 0.8)]
        context = _make_context()

        fenced = (
            "```json\n"
            '[{"ticker": "AAPL", "action": "BUY", "confidence": 0.8, "signal_summary": "OK"}]\n'
            "```"
        )
        orchestrator._mock_claude.call.return_value = fenced

        result = orchestrator.decide(signals, context)

        assert len(result) == 1
        assert result[0].ticker == "AAPL"
        assert result[0].action == "BUY"

    def test_invalid_action_skipped(self, orchestrator):
        """Claude returning invalid action values are skipped."""
        signals = [_make_signal("AAPL", "technical", "BUY", 0.8)]
        context = _make_context()

        orchestrator._mock_claude.call.return_value = _claude_decision_response([
            {"ticker": "AAPL", "action": "SELL", "confidence": 0.9, "signal_summary": "Invalid"},
        ])

        result = orchestrator.decide(signals, context)

        assert result == []

    def test_confidence_clamped_to_one(self, orchestrator):
        """Confidence values above 1.0 from Claude are clamped to 1.0."""
        signals = [_make_signal("AAPL", "technical", "BUY", 0.8)]
        context = _make_context()

        orchestrator._mock_claude.call.return_value = _claude_decision_response([
            {"ticker": "AAPL", "action": "BUY", "confidence": 1.5, "signal_summary": "Over-confident"},
        ])

        result = orchestrator.decide(signals, context)

        assert result[0].confidence == pytest.approx(1.0)

    def test_confidence_clamped_to_zero(self, orchestrator):
        """Confidence values below 0.0 from Claude are clamped to 0.0."""
        signals = [_make_signal("AAPL", "technical", "BUY", 0.8)]
        context = _make_context()

        orchestrator._mock_claude.call.return_value = _claude_decision_response([
            {"ticker": "AAPL", "action": "BUY", "confidence": -0.3, "signal_summary": "Negative"},
        ])

        result = orchestrator.decide(signals, context)

        assert result[0].confidence == pytest.approx(0.0)
