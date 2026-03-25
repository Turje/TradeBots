"""
Unit tests for agents/strategies/ and pipeline/strategy_runner.py.

All Claude API calls are mocked — no network I/O occurs.

Run with:
    cd /Users/turje87/Desktop/TradeBots && python -m pytest tests/test_strategies.py -v
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from core.models import Candidate, StrategySignal
from core.context import PipelineContext


# ---------------------------------------------------------------------------
# Helpers — synthetic test data
# ---------------------------------------------------------------------------

def _make_candidate(
    ticker: str = "AAPL",
    price: float = 175.0,
    volume_ratio: float = 2.0,
    price_change_pct: float = 3.5,
    asset_type: str = "stock",
) -> Candidate:
    return Candidate(
        ticker=ticker,
        asset_type=asset_type,
        price=price,
        volume_ratio=volume_ratio,
        price_change_pct=price_change_pct,
        indicators={
            "RSI": 55.0,
            "MACD_line": 0.5,
            "MACD_signal": 0.3,
            "SMA20": 170.0,
            "SMA50": 160.0,
            "BB_upper": 185.0,
            "BB_lower": 155.0,
            "BB_middle": 170.0,
        },
        rationale="Passed market filter",
    )


def _make_context() -> PipelineContext:
    return PipelineContext.new(portfolio_snapshot={"cash": 100_000.0, "positions": {}})


def _claude_signal_response(items: list) -> str:
    """Serialize a list of signal dicts the way Claude would return them."""
    return json.dumps(items)


def _valid_claude_response(ticker: str = "AAPL") -> str:
    return _claude_signal_response([
        {
            "ticker": ticker,
            "action": "BUY",
            "confidence": 0.75,
            "reasoning": "Strong indicators",
        }
    ])


# ---------------------------------------------------------------------------
# Fixtures — one per agent type
# ---------------------------------------------------------------------------

@pytest.fixture
def technical_agent():
    with patch("agents.strategies.technical.ClaudeClient") as mock_cls:
        mock_claude = MagicMock()
        mock_cls.return_value = mock_claude
        from agents.strategies.technical import TechnicalAgent
        agent = TechnicalAgent()
        agent._mock_claude = mock_claude
        yield agent


@pytest.fixture
def momentum_agent():
    with patch("agents.strategies.momentum.ClaudeClient") as mock_cls:
        mock_claude = MagicMock()
        mock_cls.return_value = mock_claude
        from agents.strategies.momentum import MomentumAgent
        agent = MomentumAgent()
        agent._mock_claude = mock_claude
        yield agent


@pytest.fixture
def mean_reversion_agent():
    with patch("agents.strategies.mean_reversion.ClaudeClient") as mock_cls:
        mock_claude = MagicMock()
        mock_cls.return_value = mock_claude
        from agents.strategies.mean_reversion import MeanReversionAgent
        agent = MeanReversionAgent()
        agent._mock_claude = mock_claude
        yield agent


# ---------------------------------------------------------------------------
# Test: TechnicalAgent happy path
# ---------------------------------------------------------------------------

class TestTechnicalAgentHappyPath:
    def test_returns_strategy_signal_objects(self, technical_agent):
        """analyze() should return a list of StrategySignal instances."""
        candidate = _make_candidate("AAPL")
        context = _make_context()

        technical_agent._mock_claude.call.return_value = _valid_claude_response("AAPL")

        result = technical_agent.analyze([candidate], context)

        assert len(result) == 1
        assert isinstance(result[0], StrategySignal)

    def test_signal_fields_populated_correctly(self, technical_agent):
        """StrategySignal fields should match Claude's response."""
        candidate = _make_candidate("AAPL")
        context = _make_context()

        technical_agent._mock_claude.call.return_value = _claude_signal_response([
            {"ticker": "AAPL", "action": "BUY", "confidence": 0.8, "reasoning": "RSI bullish"}
        ])

        result = technical_agent.analyze([candidate], context)

        signal = result[0]
        assert signal.ticker == "AAPL"
        assert signal.agent_name == "technical"
        assert signal.action == "BUY"
        assert signal.confidence == pytest.approx(0.8)
        assert signal.reasoning == "RSI bullish"

    def test_multiple_candidates(self, technical_agent):
        """analyze() handles multiple candidates and returns one signal each."""
        candidates = [_make_candidate("AAPL"), _make_candidate("MSFT")]
        context = _make_context()

        technical_agent._mock_claude.call.return_value = _claude_signal_response([
            {"ticker": "AAPL", "action": "BUY", "confidence": 0.7, "reasoning": "Good RSI"},
            {"ticker": "MSFT", "action": "HOLD", "confidence": 0.5, "reasoning": "Neutral"},
        ])

        result = technical_agent.analyze(candidates, context)

        assert len(result) == 2
        tickers = {s.ticker for s in result}
        assert tickers == {"AAPL", "MSFT"}

    def test_empty_candidates_returns_empty_list(self, technical_agent):
        """analyze() with empty candidates should return [] without calling Claude."""
        context = _make_context()

        result = technical_agent.analyze([], context)

        assert result == []
        technical_agent._mock_claude.call.assert_not_called()

    def test_agent_name_is_technical(self, technical_agent):
        """TechnicalAgent.name should be 'technical'."""
        assert technical_agent.name == "technical"

    def test_confidence_clamped_to_one(self, technical_agent):
        """Confidence values above 1.0 from Claude are clamped to 1.0."""
        candidate = _make_candidate("AAPL")
        context = _make_context()

        technical_agent._mock_claude.call.return_value = _claude_signal_response([
            {"ticker": "AAPL", "action": "BUY", "confidence": 1.5, "reasoning": "Over-confident"}
        ])

        result = technical_agent.analyze([candidate], context)
        assert result[0].confidence == pytest.approx(1.0)

    def test_invalid_action_defaults_to_pass(self, technical_agent):
        """Invalid action strings should default to PASS."""
        candidate = _make_candidate("AAPL")
        context = _make_context()

        technical_agent._mock_claude.call.return_value = _claude_signal_response([
            {"ticker": "AAPL", "action": "SELL", "confidence": 0.9, "reasoning": "Invalid"}
        ])

        result = technical_agent.analyze([candidate], context)
        assert result[0].action == "PASS"


# ---------------------------------------------------------------------------
# Test: TechnicalAgent — PASS on Claude exception
# ---------------------------------------------------------------------------

class TestTechnicalAgentClaudeException:
    def test_returns_pass_on_claude_exception(self, technical_agent):
        """If ClaudeClient.call raises, analyze() returns PASS signals for all candidates."""
        candidates = [_make_candidate("AAPL"), _make_candidate("MSFT")]
        context = _make_context()

        technical_agent._mock_claude.call.side_effect = RuntimeError("API error")

        result = technical_agent.analyze(candidates, context)

        assert len(result) == 2
        for signal in result:
            assert signal.action == "PASS"
            assert signal.confidence == pytest.approx(0.0)

    def test_pass_signals_have_error_reasoning(self, technical_agent):
        """PASS signals from exceptions should note the error in reasoning."""
        candidate = _make_candidate("AAPL")
        context = _make_context()

        technical_agent._mock_claude.call.side_effect = RuntimeError("Timeout")

        result = technical_agent.analyze([candidate], context)

        assert "Timeout" in result[0].reasoning or "error" in result[0].reasoning.lower()


# ---------------------------------------------------------------------------
# Test: TechnicalAgent — PASS on JSON parse failure
# ---------------------------------------------------------------------------

class TestTechnicalAgentJsonParseFailure:
    def test_returns_pass_on_invalid_json(self, technical_agent):
        """If Claude returns non-JSON, analyze() returns PASS signals."""
        candidate = _make_candidate("AAPL")
        context = _make_context()

        technical_agent._mock_claude.call.return_value = "This is not valid JSON."

        result = technical_agent.analyze([candidate], context)

        assert len(result) == 1
        assert result[0].action == "PASS"
        assert result[0].confidence == pytest.approx(0.0)

    def test_returns_pass_on_non_array_json(self, technical_agent):
        """If Claude returns a JSON object (not array), analyze() returns PASS signals."""
        candidate = _make_candidate("AAPL")
        context = _make_context()

        technical_agent._mock_claude.call.return_value = json.dumps({"error": "bad format"})

        result = technical_agent.analyze([candidate], context)

        assert result[0].action == "PASS"
        assert result[0].confidence == pytest.approx(0.0)

    def test_pass_signals_include_all_candidates_on_parse_error(self, technical_agent):
        """PASS fallback covers all candidates, not just first."""
        candidates = [_make_candidate("AAPL"), _make_candidate("NVDA")]
        context = _make_context()

        technical_agent._mock_claude.call.return_value = "{ broken json"

        result = technical_agent.analyze(candidates, context)

        assert len(result) == 2
        tickers = {s.ticker for s in result}
        assert tickers == {"AAPL", "NVDA"}


# ---------------------------------------------------------------------------
# Test: MomentumAgent
# ---------------------------------------------------------------------------

class TestMomentumAgentHappyPath:
    def test_returns_strategy_signal_objects(self, momentum_agent):
        candidate = _make_candidate("TSLA")
        context = _make_context()

        momentum_agent._mock_claude.call.return_value = _valid_claude_response("TSLA")

        result = momentum_agent.analyze([candidate], context)

        assert len(result) == 1
        assert isinstance(result[0], StrategySignal)

    def test_agent_name_is_momentum(self, momentum_agent):
        assert momentum_agent.name == "momentum"

    def test_signal_agent_name_field(self, momentum_agent):
        candidate = _make_candidate("TSLA")
        context = _make_context()

        momentum_agent._mock_claude.call.return_value = _valid_claude_response("TSLA")

        result = momentum_agent.analyze([candidate], context)

        assert result[0].agent_name == "momentum"

    def test_empty_candidates_returns_empty_list(self, momentum_agent):
        context = _make_context()
        result = momentum_agent.analyze([], context)
        assert result == []
        momentum_agent._mock_claude.call.assert_not_called()


class TestMomentumAgentClaudeException:
    def test_returns_pass_on_claude_exception(self, momentum_agent):
        candidates = [_make_candidate("TSLA")]
        context = _make_context()

        momentum_agent._mock_claude.call.side_effect = RuntimeError("Rate limit")

        result = momentum_agent.analyze(candidates, context)

        assert len(result) == 1
        assert result[0].action == "PASS"
        assert result[0].confidence == pytest.approx(0.0)


class TestMomentumAgentJsonParseFailure:
    def test_returns_pass_on_invalid_json(self, momentum_agent):
        candidate = _make_candidate("TSLA")
        context = _make_context()

        momentum_agent._mock_claude.call.return_value = "not json at all"

        result = momentum_agent.analyze([candidate], context)

        assert result[0].action == "PASS"
        assert result[0].confidence == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test: MeanReversionAgent
# ---------------------------------------------------------------------------

class TestMeanReversionAgentHappyPath:
    def test_returns_strategy_signal_objects(self, mean_reversion_agent):
        candidate = _make_candidate("META")
        context = _make_context()

        mean_reversion_agent._mock_claude.call.return_value = _valid_claude_response("META")

        result = mean_reversion_agent.analyze([candidate], context)

        assert len(result) == 1
        assert isinstance(result[0], StrategySignal)

    def test_agent_name_is_mean_reversion(self, mean_reversion_agent):
        assert mean_reversion_agent.name == "mean_reversion"

    def test_signal_agent_name_field(self, mean_reversion_agent):
        candidate = _make_candidate("META")
        context = _make_context()

        mean_reversion_agent._mock_claude.call.return_value = _valid_claude_response("META")

        result = mean_reversion_agent.analyze([candidate], context)

        assert result[0].agent_name == "mean_reversion"

    def test_hold_action_accepted(self, mean_reversion_agent):
        candidate = _make_candidate("META")
        context = _make_context()

        mean_reversion_agent._mock_claude.call.return_value = _claude_signal_response([
            {"ticker": "META", "action": "HOLD", "confidence": 0.4, "reasoning": "Neutral RSI"}
        ])

        result = mean_reversion_agent.analyze([candidate], context)
        assert result[0].action == "HOLD"

    def test_empty_candidates_returns_empty_list(self, mean_reversion_agent):
        context = _make_context()
        result = mean_reversion_agent.analyze([], context)
        assert result == []
        mean_reversion_agent._mock_claude.call.assert_not_called()


class TestMeanReversionAgentClaudeException:
    def test_returns_pass_on_claude_exception(self, mean_reversion_agent):
        candidates = [_make_candidate("META")]
        context = _make_context()

        mean_reversion_agent._mock_claude.call.side_effect = RuntimeError("Connection error")

        result = mean_reversion_agent.analyze(candidates, context)

        assert len(result) == 1
        assert result[0].action == "PASS"
        assert result[0].confidence == pytest.approx(0.0)


class TestMeanReversionAgentJsonParseFailure:
    def test_returns_pass_on_invalid_json(self, mean_reversion_agent):
        candidate = _make_candidate("META")
        context = _make_context()

        mean_reversion_agent._mock_claude.call.return_value = "{bad json"

        result = mean_reversion_agent.analyze([candidate], context)

        assert result[0].action == "PASS"
        assert result[0].confidence == pytest.approx(0.0)

    def test_returns_pass_on_non_array_json(self, mean_reversion_agent):
        candidate = _make_candidate("META")
        context = _make_context()

        mean_reversion_agent._mock_claude.call.return_value = json.dumps({"bad": "format"})

        result = mean_reversion_agent.analyze([candidate], context)

        assert result[0].action == "PASS"
        assert result[0].confidence == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test: StrategyRunner — parallel execution and result merging
# ---------------------------------------------------------------------------

class TestStrategyRunnerParallelExecution:

    def _make_runner(self):
        """Create a StrategyRunner with all three agents mocked."""
        with (
            patch("agents.strategies.technical.ClaudeClient"),
            patch("agents.strategies.momentum.ClaudeClient"),
            patch("agents.strategies.mean_reversion.ClaudeClient"),
        ):
            from pipeline.strategy_runner import StrategyRunner
            runner = StrategyRunner()
        return runner

    def test_runs_all_three_agents(self):
        """StrategyRunner should produce signals from all 3 agents."""
        with (
            patch("agents.strategies.technical.ClaudeClient"),
            patch("agents.strategies.momentum.ClaudeClient"),
            patch("agents.strategies.mean_reversion.ClaudeClient"),
        ):
            from pipeline.strategy_runner import StrategyRunner
            runner = StrategyRunner()

        candidate = _make_candidate("AAPL")
        context = _make_context()

        # Patch each agent's analyze method
        for agent in runner.agents:
            agent.analyze = MagicMock(return_value=[
                StrategySignal(
                    ticker="AAPL",
                    agent_name=agent.name,
                    action="BUY",
                    confidence=0.7,
                    reasoning="Mocked signal",
                )
            ])

        result = runner.run([candidate], context)

        # Should have 3 signals — one per agent
        assert len(result) == 3
        agent_names = {s.agent_name for s in result}
        assert "technical" in agent_names
        assert "momentum" in agent_names
        assert "mean_reversion" in agent_names

    def test_merges_results_from_all_agents(self):
        """All signals from all agents are merged into a single flat list."""
        with (
            patch("agents.strategies.technical.ClaudeClient"),
            patch("agents.strategies.momentum.ClaudeClient"),
            patch("agents.strategies.mean_reversion.ClaudeClient"),
        ):
            from pipeline.strategy_runner import StrategyRunner
            runner = StrategyRunner()

        candidates = [_make_candidate("AAPL"), _make_candidate("MSFT")]
        context = _make_context()

        for agent in runner.agents:
            agent.analyze = MagicMock(return_value=[
                StrategySignal(
                    ticker=c.ticker,
                    agent_name=agent.name,
                    action="HOLD",
                    confidence=0.5,
                    reasoning="Mocked",
                )
                for c in candidates
            ])

        result = runner.run(candidates, context)

        # 3 agents × 2 candidates = 6 signals
        assert len(result) == 6
        assert all(isinstance(s, StrategySignal) for s in result)

    def test_returns_empty_on_no_candidates(self):
        """StrategyRunner.run() with empty candidates list returns [] without calling agents."""
        with (
            patch("agents.strategies.technical.ClaudeClient"),
            patch("agents.strategies.momentum.ClaudeClient"),
            patch("agents.strategies.mean_reversion.ClaudeClient"),
        ):
            from pipeline.strategy_runner import StrategyRunner
            runner = StrategyRunner()

        context = _make_context()

        for agent in runner.agents:
            agent.analyze = MagicMock()

        result = runner.run([], context)

        assert result == []
        for agent in runner.agents:
            agent.analyze.assert_not_called()

    def test_each_agent_called_with_correct_candidates(self):
        """Each agent should receive the same candidate list."""
        with (
            patch("agents.strategies.technical.ClaudeClient"),
            patch("agents.strategies.momentum.ClaudeClient"),
            patch("agents.strategies.mean_reversion.ClaudeClient"),
        ):
            from pipeline.strategy_runner import StrategyRunner
            runner = StrategyRunner()

        candidates = [_make_candidate("AAPL"), _make_candidate("NVDA")]
        context = _make_context()

        received_candidates = {}
        for agent in runner.agents:
            name = agent.name

            def _capture(cands, ctx, _name=name):
                received_candidates[_name] = cands
                return []

            agent.analyze = MagicMock(side_effect=_capture)

        runner.run(candidates, context)

        for agent in runner.agents:
            assert agent.name in received_candidates
            tickers = [c.ticker for c in received_candidates[agent.name]]
            assert "AAPL" in tickers
            assert "NVDA" in tickers


# ---------------------------------------------------------------------------
# Test: StrategyRunner — agent exception handling
# ---------------------------------------------------------------------------

class TestStrategyRunnerExceptionHandling:

    def test_runner_handles_agent_exception_gracefully(self):
        """If an agent raises inside ThreadPoolExecutor, runner returns PASS signals for it."""
        with (
            patch("agents.strategies.technical.ClaudeClient"),
            patch("agents.strategies.momentum.ClaudeClient"),
            patch("agents.strategies.mean_reversion.ClaudeClient"),
        ):
            from pipeline.strategy_runner import StrategyRunner
            runner = StrategyRunner()

        candidate = _make_candidate("AAPL")
        context = _make_context()

        # Make technical raise, others succeed
        for agent in runner.agents:
            if agent.name == "technical":
                agent.analyze = MagicMock(side_effect=RuntimeError("Thread crash"))
            else:
                agent.analyze = MagicMock(return_value=[
                    StrategySignal(
                        ticker="AAPL",
                        agent_name=agent.name,
                        action="BUY",
                        confidence=0.6,
                        reasoning="OK",
                    )
                ])

        result = runner.run([candidate], context)

        # Should still get signals — 1 PASS from failed agent + 2 from successful agents
        assert len(result) == 3

        # The failed agent's signal should be a PASS
        tech_signals = [s for s in result if s.agent_name == "technical"]
        assert len(tech_signals) == 1
        assert tech_signals[0].action == "PASS"
        assert tech_signals[0].confidence == pytest.approx(0.0)

    def test_runner_continues_when_one_agent_fails(self):
        """Other agents' results are still collected even when one agent fails."""
        with (
            patch("agents.strategies.technical.ClaudeClient"),
            patch("agents.strategies.momentum.ClaudeClient"),
            patch("agents.strategies.mean_reversion.ClaudeClient"),
        ):
            from pipeline.strategy_runner import StrategyRunner
            runner = StrategyRunner()

        candidate = _make_candidate("AAPL")
        context = _make_context()

        for agent in runner.agents:
            if agent.name == "mean_reversion":
                agent.analyze = MagicMock(side_effect=Exception("Unexpected crash"))
            else:
                agent.analyze = MagicMock(return_value=[
                    StrategySignal(
                        ticker="AAPL",
                        agent_name=agent.name,
                        action="HOLD",
                        confidence=0.5,
                        reasoning="Fine",
                    )
                ])

        result = runner.run([candidate], context)

        successful_agents = {s.agent_name for s in result if s.action != "PASS"}
        assert "technical" in successful_agents
        assert "momentum" in successful_agents
