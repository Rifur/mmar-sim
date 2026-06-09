# mmar-sim

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rifur/mmar-sim/blob/main/mmar_colab.ipynb)

**MMAR multifractal market simulator** — Monte Carlo price paths with heavy tails, volatility clustering, long memory, goodness-of-fit validation, and risk scenario tables for Taiwan & US equities.

## Disclaimer / 免責聲明

**Research and education only — not investment advice.**  
本工具僅供研究與教育，**不構成投資建議**；模擬結果為假設性情境，過去分布不代表未來報酬。

See **[DISCLAIMER.md](DISCLAIMER.md)** for the full text (中文 / English).

## Features

| Layer | Method |
|-------|--------|
| Heavy tails | α-stable sub-Gaussian (CMS), Hill / truncated MLE, monthly α |
| Long memory | MFDFA h(2) → fGn (Cholesky); R/S Hurst reported for diagnostics only |
| Multifractal | MFDFA singularity width → lognormal MMAR cascade |
| Regime | Bayesian α shrinkage, crisis dynamic H, HAR-RV + GJR |
| Asymmetry | Monthly one-tail Hill (Taleb-inspired); GJR down-day vol ×1.5; left-tail calibration |
| Calibration | Recent-window body anchor, stress left-tail pool, synthetic paths when history is short |
| Validation | KS test, Q-Q plots, percentile MAE, exceedance rates |
| Scenarios | Pullback entry, spot buy, VaR/CVaR, dense-zone bands |

Supports **TW** tickers (e.g. `2330.TW`, `0050.TW`) with daily price-limit handling, and **US** tickers (e.g. `NVDA`, `MU`).

## Quick start (local)

Requires [uv](https://docs.astral.sh/uv/) or Python 3.12+.

```bash
git clone https://github.com/Rifur/mmar-sim.git
cd mmar-sim
uv sync
uv run real_fractal_sim.py 2330.TW --sims 10000
uv run real_fractal_sim.py NVDA --steps 252 --sims 5000
```

Outputs land in `output/`:

- `{TICKER}_mmar.png` — path fan chart, return distribution, probability bars
- `{TICKER}_mmar_gof.png` — goodness-of-fit diagnostics (English labels)

### CLI options

```
uv run real_fractal_sim.py TICKER [options]

  --market TICKER       Market index override (auto-detected by default)
  --start DATE          Simulation start date (default: latest trading day)
  --hist-start DATE     History window start (default: 2020-01-01)
  --steps N             Simulation horizon in trading days (default: 252)
  --sims N              Number of paths (default: 10000)
  --seed N              Random seed (default: 42)
  --cascade-levels K    MMAR cascade depth (default: 12)
  --out PATH            Output PNG path (default: output/{TICKER}_mmar.png)
  --no-body-calibrate   Disable terminal body percentile anchoring
  --weight-halflife N   Exponential-decay half-life for weighted estimation (default: 504)
  --calib-recent-days N Recent window for calibration (default: 1260)
  --stress-tail-weight  Weight for worst rolling-window left tail (default: 0.40)
  --no-left-tail-enforce  Disable left-tail exceedance enforcement
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

`run_colab()` prints the **full text report** (percentiles, scenario tables, VaR, GOF) and then shows both charts. Scroll up in the output cell to read the report.

```python
from real_fractal_sim_colab import run_colab

result = run_colab("2330.TW", n_steps=20, n_sims=10000, seed=42)
# Optional: re-print or show charts later
# from real_fractal_sim_colab import print_report, display_charts
# print_report(result); display_charts(result)
```

`run_colab()` defaults to `n_sims=5000`; pass `n_sims=10000` to match the CLI default.

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

Single-factor MMAR return (market or residual leg):

```
r_t = σ · √A_t · fGn_t(H) · [n·Δθ_t / E(n·Δθ)]^H

A_t   ~ |S_{α/2}(1,1,0)|     sub-Gaussian mixing (heavy tails)
fGn_t ~ fractional Gaussian noise (H from MFDFA h(2))
Δθ_t  ~ lognormal MMAR cascade (volatility clustering)
```

Stock returns: `R_s = α + β·R_m + R_ε` — two-factor model with joint block-bootstrap estimation of H and Δα for market and residual legs.

**Long memory:** Simulation uses MFDFA h(2) (with crisis H boost when triggered). Classical R/S Hurst is computed and printed for cross-check only; it does not drive the generator.

### Downside–upside asymmetry (Taleb-inspired)

Losses can be heavier-tailed than gains. The path generator uses **symmetric** α-stable mixing; asymmetry is layered through estimation, volatility forecasting, and calibration:

- **Monthly one-tail Hill** on left vs right monthly returns (daily returns are distorted by TW price limits)
- If the left tail is materially heavier (`α_left < α_right`, gap > 0.15), the residual tail index `α_ε` may be tightened — only when that makes the estimate more conservative
- **HAR-RV + GJR:** down-day realized volatility weighted ×1.5 in the vol forecast
- **Post-simulation calibration:** left tail anchored more strongly (P1/P5 blend 90%) than the right (P95/P99 blend 35%), stress-window left-tail pool, optional left-tail exceedance enforcement

### Pure theory vs. this implementation

The generator follows Calvet–Fisher MMAR structure, but outputs are also shaped by:

- Terminal body percentile anchoring to recent history
- Stress left-tail windows and left-tail exceedance enforcement
- Synthetic terminal paths blended in when individual-stock history is short (Bayesian α shrinkage)

Use `--no-body-calibrate` and `--no-left-tail-enforce` to inspect closer-to-raw generator output.

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
- Taleb — fat tails and downside–upside asymmetry (monthly one-tail diagnostics)

## License

MIT — see [LICENSE](LICENSE). Use of this software is subject to [DISCLAIMER.md](DISCLAIMER.md).