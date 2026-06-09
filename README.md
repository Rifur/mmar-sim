# mmar-sim

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rifur/mmar-sim/blob/main/mmar_colab.ipynb)

**MMAR multifractal market simulator** — Monte Carlo price paths with heavy tails, volatility clustering, long memory, goodness-of-fit validation, and risk scenario tables for Taiwan & US equities.

> **Disclaimer:** Research and education tool only. **Not investment advice.** Past distributions do not predict future returns. Simulation includes empirical post-processing (body/tail calibration) beyond pure theoretical MMAR.

## Features

| Layer | Method |
|-------|--------|
| Heavy tails | α-stable sub-Gaussian (CMS), Hill / truncated MLE, monthly α |
| Long memory | R/S Hurst, fractional Gaussian noise (Cholesky) |
| Multifractal | MFDFA singularity width → lognormal MMAR cascade |
| Regime | Bayesian α shrinkage, crisis dynamic H, HAR-RV + GJR |
| Calibration | Recent-window body anchor, stress left-tail pool |
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

  --hist-start DATE     History window start (default: 2020-01-01)
  --steps N             Simulation horizon in trading days (default: 252)
  --sims N              Number of paths (default: 10000)
  --cascade-levels K    MMAR cascade depth (default: 12)
  --no-body-calibrate   Disable terminal body percentile anchoring
  --calib-recent-days N Recent window for calibration (default: 1260)
  --stress-tail-weight  Weight for worst rolling-window left tail (default: 0.40)
  --no-left-tail-enforce  Disable left-tail exceedance enforcement
```

## Google Colab

**You do not need a separate Colab-only script.** The repo ships:

1. **`real_fractal_sim.py`** — core library + CLI + `run_colab()`
2. **`mmar_colab.ipynb`** — ready-to-run notebook
3. **`real_fractal_sim_colab.py`** — thin shim for older `from real_fractal_sim_colab import run_colab` imports

### Option A — Open in Colab (recommended)

Click the badge at the top of this README, then run all cells.

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

## 0050 batch scan (Taiwan)

Scan all Taiwan 50 index constituents and rank by GOF score:

```bash
uv run batch_0050_scan.py --sims 3000
uv run batch_0050_scan.py --codes 2330,2454,3711 --sims 2000
```

Writes `output/0050_scan_YYYYMMDD/` with per-stock reports and `summary.csv`.

## Model overview

```
r_t = σ · √A_t · fGn_t(H) · [n·Δθ_t / E(n·Δθ)]^H

A_t   ~ |S_{α/2}(1,1,0)|     sub-Gaussian mixing (heavy tails)
fGn_t ~ fractional Gaussian noise (Hurst memory)
Δθ_t  ~ lognormal MMAR cascade (volatility clustering)
```

Stock returns: `R_s = α + β·R_m + R_ε` with joint block-bootstrap estimation of H and Δα.

**Pure theory vs. this implementation:** The generator follows Calvet–Fisher MMAR structure, but terminal distributions are also anchored to recent historical quantiles and stress left-tail windows. Disable with `--no-body-calibrate` to inspect raw generator output.

## Project layout

```
mmar-sim/
├── real_fractal_sim.py       # Core module, CLI, run_colab()
├── real_fractal_sim_colab.py # Compatibility shim
├── mmar_colab.ipynb          # Colab notebook
├── batch_0050_scan.py        # 0050 constituent batch scanner
├── pyproject.toml
└── README.md
```

## References

- Mandelbrot (1963) — α-stable price variations
- Mandelbrot & Wallis (1969) — R/S Hurst analysis
- Calvet & Fisher (1997, 2002) — MMAR / multifractal volatility
- Kantelhardt et al. (2002) — MFDFA
- Chambers, Mallows & Stuck (1976) — α-stable sampling (CMS)

## License

MIT — see [LICENSE](LICENSE).