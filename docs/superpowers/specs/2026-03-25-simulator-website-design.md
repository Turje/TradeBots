# TradeBots Simulator Website — Design Spec

**Date:** 2026-03-25
**Status:** Approved
**Author:** Claude (brainstormed with Saar Turjeman)

---

## Overview

A single-file static website (`docs/index.html`) hosted on GitHub Pages that presents TradeBots as a live-looking trading simulation dashboard. The site uses mock data animated with JavaScript to demonstrate the full 5-stage pipeline in real time. No backend required.

**Goal:** Shareable demo link for investors, technical reviewers, and general audiences.

---

## Visual Style — Dark Finance

| Token | Value |
|-------|-------|
| Background | `#050810` |
| Surface | `#0a0e1a` |
| Panel | `#0f172a` |
| Border | `#1e293b` |
| Accent indigo | `#4338ca` / `#818cf8` |
| Accent purple | `#7c3aed` / `#c4b5fd` |
| Green (gain) | `#34d399` |
| Red (loss) | `#f87171` |
| Cyan (executor) | `#67e8f9` |
| Amber (watcher) | `#fbbf24` |
| Body font | `system-ui, -apple-system, sans-serif` |
| Code/times | `SF Mono, Fira Code, monospace` |

Stage-specific colors: Scanner=green, Strategy=indigo, Orchestrator=purple, Executor=cyan, Watcher=amber.

Visual reference mockup (brainstorm artifact, not tracked in git):
`.superpowers/brainstorm/70405-1774467698/design-mockup.html`

---

## Architecture

- **Deployment:** `Turje/TradeBots` (new standalone GitHub repo, public) — created specifically for this project, separate from any existing parent git tree
- **GitHub Pages config:** Branch `main`, folder `/docs`
- **URL:** `https://turje.github.io/TradeBots/`
- **Single file:** `docs/index.html` — all HTML, CSS, JS inline
- **Dependencies:** None. Vanilla HTML + CSS + JS.
- **Data:** All mock, generated in JS at runtime. No API calls.
- **Responsive:** Columns stack vertically at ≤900px viewport width.

---

## Implementation Status — Which Stages Are Real

| Stage | Backend code exists? | Simulator behavior |
|-------|---------------------|--------------------|
| Scanner Bot | ✅ `agents/scanner.py` | Mock ticker scan modeled on real priority scoring |
| Strategy (3 agents) | ✅ `agents/strategies/` | Mock signals with agent-specific indicator logic |
| Orchestrator | ✅ `agents/orchestrator.py` | Mock mirrors real weighting rules (see below) |
| **Executor** | ❌ Not yet implemented | Speculative — represents design intent for Task 8 |
| **Watcher** | ❌ Not yet implemented | Speculative — represents design intent for Task 9 |

The Executor and Watcher panels on the demo site are forward-looking. They show what the completed pipeline will do, not what is currently built.

---

## Page Structure

### 1. Header Bar
Portfolio value, total P&L %, cash available, open positions count, cycle counter, UTC clock, LIVE badge. All values update during the simulation.

### 2. Pipeline Flow Bar
Five stage nodes connected by animated arrows. Each node glows + pulses when its stage is active. Arrow lights up as data flows left-to-right:

```
[🔍 Scanner] ──▶ [⚡ Strategy (3 agents)] ──▶ [🧠 Orchestrator] ──▶ [💼 Executor] ──▶ [👁️ Watcher]
```

### 3. Content Panels

**Row 1 — 3-column grid:**

| Panel | Content |
|-------|---------|
| Scanner Bot | Candidate tickers, price change %, volume ratio, RSI, MACD direction, priority score bar |
| Strategy Agents | Per-ticker: 3 agent rows with BUY / HOLD / PASS pill + confidence score |
| Orchestrator | Synthesized decisions with one-line reasoning. Pre-filtered tickers shown separately. |

**Row 2 — 2-column grid:**

| Panel | Content |
|-------|---------|
| Executor | Last 4 trade executions (FILLED/PENDING), shares, price, cost, time. Cash + position count footer. |
| Watcher Bot | Open positions with P&L %, stop-loss price, hold duration, HOLD / REASSESS / SELL status |

### 4. Activity Log
Scrolling feed capped at 10 events, color-coded by stage. New entries slide in at the top.

---

## Simulation Loop

JavaScript state machine, ~20-second cycle:

| Phase | Duration | What happens |
|-------|----------|--------------|
| SCANNING | 3.2s | Scanner node pulses; 4–6 candidates generated and rendered |
| STRATEGY | 4.0s | Strategy node pulses; agent signals generated per candidate |
| ORCHESTRATING | 2.2s | Orchestrator node pulses; decisions rendered with reasoning |
| EXECUTING | 2.0s | Executor node pulses; buy orders logged, cash deducted |
| IDLE | 9.0s | Watcher runs stop-loss/take-profit checks; portfolio drifts |

Loop runs indefinitely. Minimum 1 candidate is always selected per cycle.

---

## Orchestrator Weighting Rules (mirrors real `agents/orchestrator.py`)

1. **Pre-filter:** 2+ agents return PASS → immediate HOLD, skip Claude (in simulator: skip decision logic)
2. **Strong BUY:** 3/3 agents BUY with avg confidence > 0.62 → BUY with high confidence
3. **Moderate BUY:** 2/3 agents BUY with avg confidence > 0.54 → BUY with reduced confidence; dissent noted
4. **HOLD:** Fewer than 2 BUY signals or avg confidence too low → HOLD
5. **Block:** 1/3 split where any agent has confidence = 0.0 is treated as HOLD (zero-confidence override blocked)

Note: The real orchestrator also has a Claude-driven override path (upgrade weak signal with written rationale). This is omitted from the simulator as it requires live LLM inference.

---

## Mock Data

- **Ticker pool:** NVDA, AAPL, MSFT, TSLA, META, GOOGL, AMZN, AMD, NFLX, ORCL, JPM, V (12 tickers)
- **Per scan cycle:** 4–6 candidates (minimum 1 guaranteed) with realistic price, volume, RSI, MACD
- **Portfolio:** Starts $100,000. Cash deducted on BUY, returned on SELL. Floor: $0 (no shorting).
- **Position sizing:** 8% of cash per trade, max 5 open positions
- **Stop-loss:** −5%. Take-profit: +15%. Watcher fires automatically.

---

## Edge Cases

- Candidate selection always returns ≥ 1 ticker
- If cash is insufficient for a trade, Executor logs the skip and continues
- If max positions reached, Executor skips further buys that cycle
- Portfolio value floor: $0 (not enforced in simulator — unreachable in normal operation)
- Loop runs indefinitely, no stop condition

---

## Deployment Steps

1. Initialize standalone git repo inside `TradeBots/` directory
2. `gh repo create Turje/TradeBots --public`
3. Commit all TradeBots source + `docs/index.html`
4. Push to `main`
5. Enable GitHub Pages: branch `main`, folder `/docs`
6. URL live at: `https://turje.github.io/TradeBots/`
