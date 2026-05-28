# Investment Fund

Multi-agent LLM trading research system. Five debating analyst agents
form a view, two researcher agents argue bull vs. bear, a trader proposes
a position, three risk debators stress-test it, and a Risk Judge plus an
adversarial CRO review produce the final BUY/SELL/HOLD. A meta layer
(JANUS) blends the outputs of multiple configuration cohorts, an
auto-research loop rewrites underperforming agent prompts, and a
Darwinian weighting scheme down-weights historically inaccurate agents.

The system is designed to be backtested, ablated, and benchmarked
against simple statistical baselines.

## Subsystems

1. **Darwinian Weights** — per-agent accuracy tracking; quartile-based
   weights in `[0.3, 2.5]` injected into the Research Manager and Risk
   Manager prompts.
2. **CRO Adversarial Review** — final review pass scoring the trade on
   six risk dimensions; can override to HOLD.
3. **Forward Context** — FOMC / CPI / NFP / earnings dates within the
   lookahead window injected into all four analyst prompts.
4. **Autoresearch** — rolling Sharpe scorecard per agent; an LLM
   modifier rewrites prompts of underperforming agents.
5. **JANUS Meta-Weighting** — multiple configuration cohorts run in
   parallel, blended with a softmax bounded to `[0.2, 0.8]`.

All subsystems are gated behind `enable_*` flags in
`fund/default_config.py` and can be turned on independently.

## Quick start

```bash
# Install
uv sync   # or: pip install -e .

# Configure API keys
cp .env.example .env
# edit .env: OPENAI_API_KEY (or OpenRouter), ALPHA_VANTAGE_API_KEY,
#           FINNHUB_API_KEY, FRED_API_KEY (optional)

# Single trade
python main.py

# Backtest
python backtest.py \
    --tickers AAPL MSFT NVDA \
    --start 2026-01-06 --end 2026-06-30 \
    --freq-weeks 1 --hold-days 7 \
    --enable-cro --enable-forward-context

# Full ablation sweep
./run_ablation.sh
```

## Layout

```
fund/
├── agents/              # analysts, researchers, trader, risk debators
├── autoresearch/        # rolling scorecard + prompt modifier loop
├── context/             # forward-looking macro/earnings context
├── dataflows/           # market, fundamental, news, macro providers
├── graph/               # LangGraph orchestrator and routing
├── meta/                # JANUS multi-cohort blending
├── prompts/             # externalized agent prompt templates
└── scoring/             # Darwinian per-agent weight tracking
backtest.py              # backtest harness with CLI flags per subsystem
baselines.py             # buy-and-hold / momentum / random baselines
run_ablation.sh          # subsystem on/off sweep across tickers
BENCHMARK_PLAN.md        # current benchmark methodology and open issues
```

## Status

Active research. See `BENCHMARK_PLAN.md` for the current evaluation
universe, period, baselines, metrics, and where the last benchmark
discussion left off.
