# mmar-sim

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rifur/mmar-sim/blob/main/mmar_colab.ipynb)

**MMAR multifractal market simulator** — Monte Carlo price paths with heavy tails, volatility clustering, regime persistence, path-dependent liquidity shocks, multi-model tail risk, goodness-of-fit validation, and scenario tables for **TW / US / JP / KR** equities and bond ETFs.

## Disclaimer / 免責聲明

**Research and education only — not investment advice.**  
本工具僅供研究與教育，**不構成投資建議**；模擬結果為假設性情境，過去分布不代表未來報酬。

See **[DISCLAIMER.md](DISCLAIMER.md)** for the full text (中文 / English).

## Features

| Layer | Method |
|-------|--------|
| Heavy tails | α-stable sub-Gaussian (CMS); Hill / **truncated MLE (TW only)**; monthly α; **α bootstrap + KDE pool** per path |
| Long memory | MFDFA h(2) → fGn (Cholesky); R/S Hurst reported for diagnostics only |
| Multifractal | MFDFA singularity width → lognormal MMAR cascade |
| Regime | **Semi-Markov persistence kernel** (normal / stress / crisis); dynamic Dirichlet weights; Bayesian α shrinkage; crisis dynamic H; HAR-RV + GJR |
| Liquidity | **Path-dependent spread kernel** `S_t` — vol & jump intensity co-evolve per path; **auto-damped when `n_steps ≤ 5`** (intraday / short-horizon) |
| Asymmetry | Monthly one-tail Hill (Taleb-inspired); GJR down-day vol ×1.5; **symmetric tail calibration** (left & right) |
| Calibration | Recent-window body anchor; stress left-tail pool; synthetic paths when history is short |
| Multi-model | **Historical bootstrap** + **Student-t** + MMAR → **worst-case envelope**; model disagreement index |
| Stress | **Black Swan event catalog** (TW / US / JP / KR); frequency sweep (0.1%–10% injection) |
| Bonds | **Yield-duration constraint** for known bond ETFs (auto-detect or manual); carry + mean-reversion; damped liquidity kernel |
| Survival | **Fragility curve**; probability of ruin; CVaR99; reflexivity (Soros) impact |
| Validation | KS test, Q-Q plots, percentile MAE, exceedance rates |
| Scenarios | Pullback entry, spot buy, VaR/CVaR, dense-zone bands |

## Supported markets

Ticker suffixes are auto-detected via `detect_market()`:

| Market | Examples | Index | Sim cap (log-return) | Truncated α MLE |
|--------|----------|-------|----------------------|-----------------|
| **TW** | `2330.TW`, `0050.TW`, `^TWII` | `^TWII` | ±10% (0.0953) | Yes — daily returns are price-limit censored |
| **US** | `NVDA`, `MU`, `^GSPC` | `^GSPC` | 0.20 (stability guard) | No |
| **JP** | `7203.T`, `^N225` | `^N225` | 0.25 (tiered limits approx.) | No |
| **KR** | `005930.KS`, `^KS11` | `^KS11` | 0.30 (±30% limit approx.) | No |

When the ticker **is** a local index (e.g. `^TWII`), the CLI uses `^GSPC` as a global factor to avoid degenerate self-regression (β≈1, ε≈0).

**Bond ETFs:** known tickers (`00679B`, `TLT`, `AGG`, `HYG`, …) auto-apply duration/yield constraints and conservative liquidity settings. Override with `--bond-duration` / `--bond-yield`.

Each run prints a **full text report** before charts: parameter estimates, regime mix, percentiles, scenario tables, Taleb survival metrics, fragility curve, stress-frequency sweep, worst-case model envelope, model disagreement, bond constraint stats (if applicable), and GOF.

## Quick start (local)

Requires [uv](https://docs.astral.sh/uv/) or Python 3.12+.

```bash
git clone https://github.com/Rifur/mmar-sim.git
cd mmar-sim
uv sync
uv run real_fractal_sim.py 2330.TW --sims 10000
uv run real_fractal_sim.py NVDA --steps 252 --sims 5000
uv run real_fractal_sim.py 7203.T --steps 60 --sims 5000    # Toyota (JP)
uv run real_fractal_sim.py 00679B.TW --steps 60 --sims 3000 # TW bond ETF
```

Outputs land in `output/`:

- `{TICKER}_mmar.png` — path fan chart (P20/P50/P80 representative paths), return distribution, probability bars
- `{TICKER}_mmar_gof.png` — goodness-of-fit diagnostics (English labels)

### CLI options

```
uv run real_fractal_sim.py TICKER [options]

  --market TICKER         Market index override (auto-detected by default)
  --start DATE            Simulation start date (default: latest trading day)
  --hist-start DATE       History window start (default: 2020-01-01)
  --steps N               Simulation horizon in trading days (default: 252)
  --sims N                Number of paths (default: 10000)
  --seed N                Random seed (default: 42)
  --cascade-levels K      MMAR cascade depth (default: 12)
  --out PATH              Output PNG path (default: output/{TICKER}_mmar.png)
  --no-body-calibrate     Disable terminal body percentile anchoring
  --weight-halflife N     Exponential-decay half-life for weighted estimation (default: 504)
  --calib-recent-days N   Recent window for calibration (default: 1260)
  --stress-tail-weight F  Weight for worst rolling-window left tail (default: 0.40; bond ETFs auto 0.15)
  --no-left-tail-enforce  Disable left-tail exceedance enforcement

  Bond ETF (auto-detected for known tickers):
  --bond-duration D       Effective duration in years (e.g. 16.5)
  --bond-yield Y          Current yield as decimal (e.g. 0.045 = 4.5%)
  --bond-yield-floor Y    Yield floor for price ceiling (default: 0.005 = 0.5%)
  --bond-yield-ceil Y     Yield ceiling for price floor (default: 0.12 = 12%)
  --bond-no-st-liquidity  Damp liquidity-spread kernel (φ/η/ξ) for bond ETFs
```

## Google Colab

**You do not need a separate Colab-only script.** The repo ships:

1. **`real_fractal_sim.py`** — core library + CLI + `run_colab()`
2. **`mmar_colab.ipynb`** — ready-to-run notebook (Step 1 Setup + Step 2 report & charts)
3. **`real_fractal_sim_colab.py`** — thin shim for older `from real_fractal_sim_colab import run_colab` imports

### Option A — Open in Colab (recommended)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rifur/mmar-sim/blob/main/mmar_colab.ipynb)

1. Click the badge above (or the one at the top of this README)
2. **Runtime → Run all**
3. In **Step 2**, edit `TICKER`, `N_STEPS`, and `N_SIMS`

`run_colab()` prints the **full text report** (regime mix, survival metrics, model envelope, scenarios, GOF) and then shows both charts. Scroll up in the output cell to read the report.

```python
from real_fractal_sim_colab import run_colab

result = run_colab("2330.TW", n_steps=20, n_sims=10000, seed=42)
# Optional: re-print or show charts later
# from real_fractal_sim_colab import print_report, display_charts
# print_report(result); display_charts(result)
```

`run_colab()` defaults to `n_sims=5000`; pass `n_sims=10000` to match the CLI default. Market preset (`mkt_key`, cap, stress catalog) is inferred from the ticker automatically.

### Option B — Clone inside a notebook

```python
!pip install -q yfinance matplotlib pandas scipy curl-cffi
!git clone https://github.com/Rifur/mmar-sim.git
%cd mmar-sim

%matplotlib inline
from real_fractal_sim import run_colab

result = run_colab("2330.TW", hist_start="2020-01-01", n_sims=5000, seed=42)
print(result["mmar_path"], result["gof_path"])
```

### Option C — Import after pip install from Git

```bash
pip install git+https://github.com/Rifur/mmar-sim.git
```

```python
from real_fractal_sim import run_colab
run_colab("NVDA", n_sims=3000)
```

## Model overview

### Core MMAR generator (layers 1–3)

Single-factor MMAR return (market or residual leg):

```
r_t = σ · √A_t · fGn_t(H) · [n·Δθ_t / E(n·Δθ)]^H

A_t   ~ |S_{α/2}(1,1,0)|     sub-Gaussian mixing (heavy tails)
fGn_t ~ fractional Gaussian noise (H from MFDFA h(2))
Δθ_t  ~ lognormal MMAR cascade (volatility clustering)
```

Stock returns: `R_s = α + β·R_m + R_ε` — two-factor model with joint block-bootstrap estimation of H and Δα for market and residual legs.

**Long memory:** Simulation uses MFDFA h(2) (with crisis H boost when triggered). Classical R/S Hurst is computed and printed for cross-check only; it does not drive the generator.

**Regime persistence (Semi-Markov):** Instead of a static Dirichlet prior `[7,2,1]`, the model infers the current regime (normal / stress / crisis) from recent vol ratio and crisis level, then computes **marginal regime weights at horizon T** via a sojourn-time kernel (mean dwell: ~40 / 10 / 5 days) and a sparse transition matrix. Per-path regime labels still drive α/H/λ² draws; Dirichlet concentration is derived from these dynamic weights (fallback `[7,2,1]` when history is thin).

**Liquidity spread kernel:** After base MMAR paths are generated, a path-dependent state `S_t` co-evolves with each simulation:

```
S_{t+1} = S_t · exp( φ·|r_t|/σ_hist − κ·(S_t − 1) )
σ_eff   = σ_base · (1 + η·(S_t − 1))     → amplified vol when spreads widen
λ_eff   = λ_0 · (1 + ξ·(S_t − 1))        → more jumps under liquidity stress
```

`S_0` is set by regime (normal=1, stress=2, crisis=4). Large moves widen spreads; mean reversion prevents runaway explosion.

**Parameter scaling:** Default equity params are φ=0.30, η=0.50, ξ=3.00. When **`n_steps ≤ 5`**, the kernel automatically uses damped params (φ=0.05, η=0.10, ξ=0.50) — same as bond mode — so 3-day intraday-style runs are not dominated by liquidity jumps. Bond ETFs also get damped params via `--bond-no-st-liquidity` (auto-enabled for known bond tickers). The run log tags `[short-horizon ≤5d: …]` or `[bond mode: …]` when damping applies.

### Downside–upside asymmetry (Taleb-inspired)

Losses can be heavier-tailed than gains. The path generator uses **symmetric** α-stable mixing; asymmetry is layered through estimation, volatility forecasting, and calibration:

- **Monthly one-tail Hill** on left vs right monthly returns (daily returns are distorted by TW price limits)
- If the left tail is materially heavier (`α_left < α_right`, gap > 0.15), the residual tail index `α_ε` may be tightened — only when that makes the estimate more conservative
- **HAR-RV + GJR:** down-day realized volatility weighted ×1.5 in the vol forecast
- **Post-simulation calibration:** left tail anchored at blend 90% (P1/P5), right tail at blend 80% (P95/P99), stress-window left-tail pool, optional left- and right-tail exceedance enforcement

### Bond ETF layer

For government and corporate bond ETFs, an optional **duration–yield constraint** sits on top of MMAR paths:

- Implied yield `y_t = y_0 − ln(P_t/P_0) / D` is clipped to `[yield_floor, yield_ceil]`
- Daily carry (`y_0/252`) corrects for coupon income missing in price-only history
- Mean-reversion drift pulls yields back toward long-run equilibrium
- Reflexivity drawdown threshold is relaxed vs equities

Known tickers in `_BOND_ETF_TABLE` (TW: `00679B`, `00720B`, …; US: `TLT`, `AGG`, `HYG`, …) trigger this layer automatically.

### Multi-model risk layer (layers 4–5)

After MMAR paths are generated and calibrated, the report adds independent tail-risk views:

| Component | Role |
|-----------|------|
| **Historical bootstrap** | Non-parametric rolling-window resampling — no model assumptions |
| **Student-t Monte Carlo** | Finite-moment heavy tail; df from excess kurtosis |
| **Worst-case envelope** | At each percentile, take the **most pessimistic** of MMAR / Bootstrap / Student-t (not a weighted average) |
| **Model disagreement** | Spread across models on the left tail — epistemic uncertainty vs calibration bias |
| **Stress catalog** | TW / US / JP / KR historical crash events injected at configurable frequencies |
| **Fragility curve** | Instant shock (5%–60%) applied to terminal returns; fragility index measures convexity of losses |
| **Reflexivity** | Soros-style feedback: drawdown > threshold amplifies path returns ×2 until recovery |

**Taleb survival metrics** in the report: probability of ruin (>50% loss), CVaR99, worst 0.1%, fragility index, and model disagreement level.

### Pure theory vs. this implementation

The generator follows Calvet–Fisher MMAR structure, but outputs are also shaped by:

- Terminal body percentile anchoring to recent history
- Stress left-tail windows and left/right-tail exceedance enforcement
- Synthetic terminal paths blended in when individual-stock history is short (Bayesian α shrinkage)
- Regime persistence kernel, liquidity spread co-evolution, multi-model envelope, stress injection sweep, and reflexivity overlay
- Bond yield constraints when simulating fixed-income ETFs

Use `--no-body-calibrate` and `--no-left-tail-enforce` to inspect closer-to-raw generator output. The multi-model envelope and stress sweep are diagnostic layers on top of the primary MMAR paths.

## Project layout

```
mmar-sim/
├── real_fractal_sim.py       # Core module, CLI, run_colab()
├── real_fractal_sim_colab.py # Compatibility shim
├── mmar_colab.ipynb          # Colab notebook
├── pyproject.toml
├── DISCLAIMER.md
└── README.md
```

## References

- Mandelbrot (1963) — α-stable price variations
- Mandelbrot & Wallis (1969) — R/S Hurst analysis
- Calvet & Fisher (1997, 2002) — MMAR / multifractal volatility
- Kantelhardt et al. (2002) — MFDFA
- Chambers, Mallows & Stuck (1976) — α-stable sampling (CMS)
- Taleb — fat tails, asymmetry, fragility, and model disagreement
- Soros — reflexivity and feedback in market dynamics

## License

MIT — see [LICENSE](LICENSE). Use of this software is subject to [DISCLAIMER.md](DISCLAIMER.md).