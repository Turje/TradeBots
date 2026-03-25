"""
StrategyRunner — runs all three strategy agents in parallel using ThreadPoolExecutor.

Each agent receives an immutable snapshot of candidates. Results are merged
by collecting all StrategySignal objects returned across all agents.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, Future

from agents.strategies.technical import TechnicalAgent
from agents.strategies.momentum import MomentumAgent
from agents.strategies.mean_reversion import MeanReversionAgent
from core.context import PipelineContext
from core.models import Candidate, StrategySignal


class StrategyRunner:
    """
    Runs all three strategy agents in parallel using ThreadPoolExecutor.
    Each agent receives an immutable snapshot of candidates.
    Results are merged by ticker after all threads complete.
    """

    def __init__(self):
        self.agents = [TechnicalAgent(), MomentumAgent(), MeanReversionAgent()]
        self.logger = logging.getLogger("strategy_runner")

    def run(
        self,
        candidates: list[Candidate],
        context: PipelineContext,
    ) -> list[StrategySignal]:
        """
        Runs all agents in parallel via ThreadPoolExecutor(max_workers=3).
        Each agent gets a copy of candidates (tuple conversion for immutability signal).
        Collects all signals, logs any agent failures.
        Returns merged list of all StrategySignals across all agents.
        Agent failures return PASS signals (from BaseStrategyAgent contract) — not exceptions.
        """
        if not candidates:
            self.logger.info("StrategyRunner: no candidates to analyze.")
            return []

        # Convert to tuple as an immutability signal — agents should not mutate the list
        candidates_snapshot = tuple(candidates)

        all_signals: list[StrategySignal] = []

        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_agent: dict[Future, str] = {
                executor.submit(agent.analyze, list(candidates_snapshot), context): agent.name
                for agent in self.agents
            }

            for future, agent_name in future_to_agent.items():
                try:
                    signals = future.result()
                    self.logger.info(
                        "StrategyRunner: agent '%s' returned %d signals",
                        agent_name,
                        len(signals),
                    )
                    all_signals.extend(signals)
                except Exception as exc:
                    # Defensive: BaseStrategyAgent contract says errors become PASS signals,
                    # but guard here in case of unexpected thread-level failures.
                    self.logger.error(
                        "StrategyRunner: agent '%s' raised an unexpected exception: %s",
                        agent_name,
                        exc,
                    )
                    # Emit PASS signals for all candidates from the failed agent
                    for candidate in candidates_snapshot:
                        all_signals.append(
                            StrategySignal(
                                ticker=candidate.ticker,
                                agent_name=agent_name,
                                action="PASS",
                                confidence=0.0,
                                reasoning=f"Agent raised unexpected exception: {exc}",
                            )
                        )

        self.logger.info(
            "StrategyRunner: collected %d total signals from %d agents",
            len(all_signals),
            len(self.agents),
        )
        return all_signals
