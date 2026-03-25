# TradeBots

AI-powered multi-agent paper trading system. Five-stage pipeline where each stage is a specialized Claude agent.

**[▶ Live Demo →](https://turje.github.io/TradeBots/)**

---

## Pipeline Architecture

```
[🔍 ScannerBot] ──▶ [⚡ Strategy Agents (x3)] ──▶ [🧠 MetaOrchestrator] ──▶ [💼 Executor] ──▶ [👁️ WatcherBot]
```

| Stage | Agent | Description |
|-------|-------|-------------|
| 1 | **ScannerBot** | Scans S&P 500, surfaces candidates with strong momentum/reversal signals |
| 2 | **TechnicalAgent** | Evaluates RSI, MACD, Bollinger Bands |
| 2 | **MomentumAgent** | Evaluates volume ratio and price change velocity |
| 2 | **MeanReversionAgent** | Evaluates oversold/overbought conditions |
| 3 | **MetaOrchestrator** | Synthesizes the 3 strategy votes into a BUY/HOLD decision |
| 4 | **Executor** | Places paper trades via broker integration *(in progress)* |
| 5 | **WatcherBot** | Monitors open positions, fires stop-loss / take-profit *(in progress)* |

All agents use Claude (`claude-sonnet-4-6`) with structured JSON outputs.

## Key Design Decisions

- **Pre-filter:** Tickers where 2+ agents return PASS are dropped before reaching the Orchestrator — no wasted API calls
- **Rolling memory:** Orchestrator maintains a 10-decision rolling context window
- **Paper trading:** Full paper broker with position sizing (max 10% portfolio per position, max 10 positions)
- **Thread-safe portfolio:** All portfolio mutations are lock-protected for concurrent watcher/executor access

## Project Structure

```
agents/
  scanner.py            # ScannerBot
  orchestrator.py       # MetaOrchestrator
  strategies/
    base.py             # BaseStrategyAgent (ABC)
    technical.py        # TechnicalAgent
    momentum.py         # MomentumAgent
    mean_reversion.py   # MeanReversionAgent
brokers/
  market_data.py        # Market data provider (yfinance + ccxt)
  paper.py              # Paper broker
core/
  models.py             # Dataclasses (Candidate, StrategySignal, TradeDecision, …)
  portfolio.py          # Thread-safe portfolio state
  context.py            # PipelineContext
  claude_client.py      # Claude API wrapper with retry + backoff
pipeline/
  strategy_runner.py    # Parallel strategy agent runner (ThreadPoolExecutor)
docs/
  index.html            # Live demo site (GitHub Pages)
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add ANTHROPIC_API_KEY
python -m pytest tests/ -v
```

## Status

Tasks 1–7 complete. Tasks 8 (Executor) and 9 (WatcherBot) in progress.
