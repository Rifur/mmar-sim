"""
real_fractal_sim.py — 完整忠於曼德博精神的碎形市場模擬

三層曼德博研究的完整實作：

  第一層（1963 棉花）
    原著：Price Variation in Speculative Markets
    核心：α-穩定分布（Lévy stable）
    意義：真正的重尾，α < 2 代表變異數無限，α < 1 連期望值都無限
    實作：CMS 演算法（Chambers-Mallows-Stuck 1976）
          Hill 估計量估計尾部指數 α

  第二層（1968 fBm / 尼羅河）
    原著：Fractional Brownian Motions, Fractional Noises and Applications
    核心：分數布朗運動（fBm），Hurst 指數 H 刻畫長程記憶
    意義：H ≠ 0.5 代表市場有記憶，非隨機漫步
    實作：R/S 分析法（曼德博偏好的估計方法，非 lag-variance）
          Cholesky 精確法產生 fGn

  第三層（MFDFA 碎形譜 + MMAR 級聯）
    原著：Kantelhardt et al. (2002) MFDFA；Calvet & Fisher (1997) MMAR 級聯
    核心：MFDFA 奇異度譜寬 Δα 刻畫間歇性，映射至對數常態級聯強度
    意義：波動率群聚、多尺度碎形、尾部瀑布
    實作：MFDFA → Δα + h(2)；對數常態乘法級聯（K 層，2^K 個時間格）

完整模型方程式：
  r_t = σ · √A_t · fGn_t(H) · [n·Δθ_t / E(n·Δθ)]^H

  A_t   ~ |S_{α/2}(1,1,0)|   次高斯混合因子（重尾）
  fGn_t ~ 分數高斯噪音(H)     長程記憶（Cholesky）
  Δθ_t  ~ 對數常態級聯        多重碎形交易時間（波動率群聚）

與 fractal_sim.py 的主要差異：
  fractal_sim  → fGn（有限變異數）+ Cauchy 跳躍（工程妥協）
  real_fractal → α-穩定次高斯 + MMAR 多重碎形時間（曼德博原典）

用法（本機）：
  uv run real_fractal_sim.py TICKER [選項]
  uv run real_fractal_sim.py 2330.TW
  uv run real_fractal_sim.py NVDA --sims 10000

Google Colab：
  開啟 mmar_colab.ipynb，或：
  from real_fractal_sim import run_colab
  run_colab("2330.TW", n_sims=5000)
"""

import argparse
import os
import sys
import warnings

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")


def _setup_matplotlib() -> None:
    """Plot text in English + DejaVu Sans (avoids missing CJK glyphs)."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.unicode_minus": False,
    })


def _gof_verdict_en(score: float) -> str:
    if score >= 75:
        return "Good fit"
    if score >= 50:
        return "Acceptable"
    return "Poor fit"


_setup_matplotlib()

# ── Google Colab ──────────────────────────────────────────────
IN_COLAB = False
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    pass

OUTPUT_DIR = "/content/output" if IN_COLAB else "output"


def _colab_install_deps() -> None:
    import importlib.util
    import subprocess
    needed = ["yfinance", "matplotlib", "pandas", "scipy"]
    missing = [p for p in needed if importlib.util.find_spec(p) is None]
    if missing:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q",
             "yfinance>=1.4", "matplotlib>=3.8", "pandas>=2.0",
             "scipy>=1.11", "curl-cffi>=0.15"],
        )


def _colab_matplotlib_inline() -> None:
    if not IN_COLAB:
        return
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None:
            ip.run_line_magic("matplotlib", "inline")
    except Exception:
        pass


def _ensure_output_dir(path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return path


def _save_or_show(fig, out_path: str, show_inline: bool = True) -> None:
    _ensure_output_dir(out_path)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show_inline and IN_COLAB:
        plt.show()
    plt.close(fig)


if IN_COLAB:
    _colab_install_deps()
    _colab_matplotlib_inline()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 5_000   # 每批路徑數（記憶體較 fractal_sim 高，故設較小）
K_CASCADE  = 9       # 對數常態級聯層數（CLI --cascade-levels 可調整）
_WEIGHT_HALFLIFE = 504       # 指數衰減半衰期（≈2 年交易日）
_CALIB_RECENT_DAYS = 1260    # 主體校準用近 N 交易日（≈5 年，減少「未來像全歷史」）
_BAYES_ALPHA_N_FULL = 1260   # 貝葉斯收縮：此天數以上視為資料充足（≈5 年），不收縮
_SYNTH_N_PATHS = 3000        # 合成歷史路徑數（曼德博盲樣合成法，補強短歷史體量校準）
_CRISIS_H_MAX        = 0.85  # 危機期 H 上限（Hurst 尼羅河實證約 0.7–0.9）
_CRISIS_DD_THRESH    = 0.10  # 回撤觸發門檻（10%）
_CRISIS_DD_FULL      = 0.30  # 回撤達此值 → 危機強度=1.0（30%）
_CRISIS_CONSIST_MIN  = 0.60  # 方向一致性最低要求（10日中 6日同向）
_STRESS_WORST_N = 200        # 左尾壓力池：最糟滾動窗個數
_STRESS_TAIL_WEIGHT = 0.40   # 左尾分位數混入壓力池權重
_FULL_TAIL_WEIGHT = 0.25     # 左尾分位數混入全樣本權重
_BODY_KNOT_PCTS  = [1, 5, 10, 25, 50, 75, 90, 95, 99]  # 主體+左尾節點
_LEFT_TAIL_BLEND = 0.90      # P1/P5 硬錨左尾
_RIGHT_TAIL_BLEND = 0.35     # P95/P99 軟錨右尾（保留 MMAR 重尾）
_TAIL_BLEND      = _RIGHT_TAIL_BLEND  # 相容舊參數名


# ── 市場設定 ─────────────────────────────────────────────────

MARKET_PRESETS = {
    # cap 以 log 報酬表示：台股漲跌停板 ±10%（價格）
    # ln(1.10)=0.0953（漲停），ln(0.90)=0.1054（跌停），非對稱
    # 用對稱 clip=0.0953：漲停剛好 +10%，跌停為 -9.09%（略保守）
    "TW": {"index": "^TWII", "cap": 0.0953, "currency": "TWD"},
    "US": {"index": "^GSPC", "cap": 0.20,   "currency": "USD"},
}

def detect_market(ticker: str) -> dict:
    if ticker.endswith(".TW") or ticker.endswith(".TWO"):
        return MARKET_PRESETS["TW"]
    return MARKET_PRESETS["US"]


def download_adjusted(ticker: str, start: str, end: str) -> pd.Series:
    """下載價格並以調整因子（AF）偵測並修正基金分割。
    分割特徵：Close 大跳空 + AF 幾乎不變（非配息）。
    使用 Adj Close 確保配息還原，再補正分割造成的跳空。
    """
    raw = yf.download(ticker, start=start, end=end,
                      auto_adjust=False, progress=False)
    if raw.empty:
        return pd.Series(dtype=float)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)
    close = raw["Close"].astype(float)
    adj   = raw["Adj Close"].astype(float)
    af    = (adj / close).replace([np.inf, -np.inf], np.nan).ffill()

    close_r = np.log(close).diff()
    af_r    = np.log(af).diff().fillna(0)

    # 分割日：(1) Close 跳幅 > 20%，(2) AF 幾乎不變（非配息），(3) 比率接近整數比（非崩盤）
    split_ratios = [1/5, 1/4, 1/3, 1/2, 2, 3, 4, 5]
    candidates = close_r[(np.abs(close_r) > 0.20) & (np.abs(af_r) < 0.02)]
    for date, lr in candidates.items():
        ratio = np.exp(lr)
        best = min(split_ratios, key=lambda x: abs(ratio - x))
        if abs(ratio - best) / best > 0.05:
            continue  # 不接近整數比 → 真實市場大跌，不調整
        idx = adj.index.get_loc(date)
        adj.iloc[:idx] = adj.iloc[:idx] * ratio
        denom = round(1 / ratio) if ratio < 1 else round(ratio)
        direction = f"1:{denom}" if ratio < 1 else f"{denom}:1"
        print(f"  ⚙️  {ticker} {date.date()} 偵測到 {direction} 基金分割，已回調歷史 AdjClose")

    return adj.ffill()


# ══════════════════════════════════════════════════════════════
#  第一層：α-穩定分布（1963 棉花）
# ══════════════════════════════════════════════════════════════

def hill_estimator(returns: np.ndarray, k_frac: float = 0.10) -> float:
    """Hill 估計量：估計尾部指數 α（日報酬版）。

    原理：假設尾部 P(|X| > x) ~ x^{-α}，
    用超過第 k 大值的比例估計冪次。
    α < 2 → 變異數無限（真正的 Mandelbrot 重尾）
    α < 1 → 期望值也無限（極端市場）

    注意：日報酬受漲跌停截斷，α 往往趨近 2.0，低估尾部厚度。
    """
    abs_r = np.sort(np.abs(returns))[::-1]
    k = max(15, int(len(abs_r) * k_frac))
    alpha = k / np.sum(np.log(abs_r[:k] / abs_r[k]))
    return float(np.clip(alpha, 1.01, 1.99))


def hill_estimator_monthly(s_hist: pd.Series, k_frac: float = 0.20) -> float:
    """月報酬 Hill 估計量：不受日漲跌停截斷，估出真正的尾部 α。

    碎形理論的自相似性意味著跨時間尺度應有相同的冪次律。
    日報酬被漲跌停人為截斷（台股±10%），尾部被削掉，
    Hill 估計量看不到真正的極端尾部，導致 α 趨近 1.99（上限截斷值）。

    月報酬（累積，無截斷限制）的理論行為：
    - 日報酬 α_daily < 2（無限變異數，真正重尾）
    - 月報酬 = 21 個日報酬之和 → 中央極限定理使 α_monthly > α_daily
    - 若 α_daily = 1.6，則 α_monthly ≈ 2.0~2.5（仍比常態重）
    - 若 α_monthly < 2：月報酬本身就是無限變異數，市場極端不穩定
    - 若 α_monthly 2~4：有限變異數但明顯重尾

    結果解讀：
    - α_monthly < 2.0 → 真正黑天鵝市場，α-穩定模型完全適用
    - α_monthly 2.0~3.0 → 中度重尾，CLT 尚未完全收斂
    - α_monthly > 3.0 → 趨近常態，但比 GBM 仍有更多極端事件

    回傳值不截斷上限（允許 > 2），模擬時取 min(α, 1.99) 以符合穩定分布要求。
    """
    monthly = np.log(s_hist).resample("ME").last().diff().dropna()
    if len(monthly) < 24:
        return 1.80  # 資料不足，保守預設值
    abs_r = np.sort(np.abs(monthly.values))[::-1]
    k = max(10, int(len(abs_r) * k_frac))
    alpha = k / np.sum(np.log(abs_r[:k] / abs_r[k]))
    return float(np.clip(alpha, 0.80, 5.00))  # 不截上限，顯示真實值


def hill_one_tail_monthly(s_hist: pd.Series, side: str = "left",
                           k_frac: float = 0.22) -> float:
    """月報酬單尾 Hill 估計量（Taleb 不對稱性量化）。

    日報酬被漲跌停截斷，無法分辨左右尾差異。
    月報酬不受日截斷，可直接量化：
    - alpha_left  ≡ 負月報酬的冪次指數（跌的尾部）
    - alpha_right ≡ 正月報酬的冪次指數（漲的尾部）
    若 alpha_left < alpha_right → 左尾更重 → Taleb 不對稱成立。
    """
    monthly = np.log(s_hist).resample("ME").last().diff().dropna()
    if len(monthly) < 20:
        return 1.99
    r = monthly.values
    vals = (-r[r < 0] if side == "left" else r[r > 0])
    if len(vals) < 10:
        return 1.99
    vals = np.sort(vals)[::-1]
    k = min(max(5, int(len(vals) * k_frac)), len(vals) - 1)
    if vals[k] <= 0:
        return 1.99
    alpha = k / np.sum(np.log(vals[:k] / vals[k]))
    return float(np.clip(alpha, 0.80, 5.00))


def hill_estimator_truncated(returns: np.ndarray, cap: float,
                              k_frac: float = 0.10) -> float:
    """截斷校正 Hill 估計量（Truncated Pareto MLE）。

    漲跌停（cap）截斷了真實尾部，標準 Hill 從截斷樣本算出的 h 滿足：
        E[h] = 1/α − log(ρ) / (ρ^α − 1)，ρ = cap / x_k
    對此隱式方程求根即可還原真實 α（比標準 Hill 更小、尾部更重）。
    極限行為：cap → ∞（無截斷）→ correction → 0，退化為標準 Hill。
    """
    from scipy.optimize import brentq

    abs_r = np.sort(np.abs(returns))[::-1]
    abs_r = abs_r[abs_r < cap * 0.999]    # 排除觸板觀測（截斷點已知但非完整）
    if len(abs_r) < 20:
        return 1.99

    k = max(10, int(len(abs_r) * k_frac))
    k = min(k, len(abs_r) - 1)
    x_k = float(abs_r[k])
    if x_k <= 0:
        return 1.99

    rho = cap / x_k                                   # 截斷比率（>1）
    h   = float(np.mean(np.log(abs_r[:k] / x_k)))    # 標準 Hill 統計量
    if h <= 0:
        return 1.99

    def mle_eq(alpha: float) -> float:
        # f(α) = 1/α − h − log(ρ)/(ρ^α − 1) = 0
        rho_a = rho ** alpha
        correction = np.log(rho) / (rho_a - 1.0) if rho_a > 1 + 1e-10 else 1.0 / alpha
        return 1.0 / alpha - h - correction

    try:
        fa, fb = mle_eq(0.20), mle_eq(10.0)
        if fa * fb > 0:
            return float(np.clip(1.0 / h, 0.20, 5.0))
        alpha_c = brentq(mle_eq, 0.20, 10.0, xtol=1e-6, maxiter=300)
    except (ValueError, RuntimeError):
        alpha_c = 1.0 / h

    return float(np.clip(alpha_c, 0.20, 5.0))


def resolve_alpha_sim(
    alpha_trunc: float,
    alpha_monthly: float,
    floor: float = 1.40,
    cap: float = 1.98,
) -> tuple:
    """日報酬 α 退化（截斷校正 ≥2）時，決定模擬用 α 與來源說明。

    台股漲跌停讓日報酬 Hill / 截斷 MLE 常 ≥2，α-stable 次高斯層關閉。
    月報酬不受日漲跌停截斷，可還原中度重尾（碎形自相似，經驗映射）。
    """
    if alpha_trunc < 1.99:
        return float(np.clip(alpha_trunc, floor, cap)), "截斷校正 MLE"

    if alpha_monthly < 2.0:
        a = float(np.clip(alpha_monthly, floor, cap))
        return a, f"日報酬退化，月報酬 α={alpha_monthly:.2f}<2"

    # 連續映射：月報酬 α∈[2,5] → 日報酬 α∈[1.95,1.59]（無 3.5 硬切斷）
    if alpha_monthly < 5.0:
        a = 1.95 - 0.12 * (alpha_monthly - 2.0)
        return float(np.clip(a, floor, cap)), (
            f"日報酬退化，由月報酬 α={alpha_monthly:.2f} 連續反推"
        )

    return float(floor), f"月報酬 α={alpha_monthly:.2f}≥5，取 α={floor:.2f}（極輕尾）"


def cms_stable(alpha: float, beta: float, size: int) -> np.ndarray:
    """Chambers-Mallows-Stuck（1976）演算法生成 α-穩定分布。

    alpha ∈ (0,2]：穩定指數（α=2 退化為常態，α=1 為柯西）
    beta  ∈ [-1,1]：偏態參數
    """
    U = np.random.uniform(-np.pi / 2, np.pi / 2, size)
    E = np.random.exponential(1.0, size)
    zeta = -beta * np.tan(np.pi * alpha / 2)
    xi   = np.arctan(-zeta) / alpha
    X = ((1 + zeta**2) ** (1 / (2 * alpha))
         * np.sin(alpha * (U + xi)) / np.cos(U) ** (1 / alpha)
         * (np.cos(U - alpha * (U + xi)) / E) ** ((1 - alpha) / alpha))
    return X


def positive_stable_mixing(alpha: float, size: int) -> np.ndarray:
    """生成次高斯混合因子 A ~ |S_{α/2}(1, 1, 0)|，正規化至中位數=1。

    sub-Gaussian 表示式：X = √A · G，其中 G ~ N(0,Σ)
    賦予 X 穩定分布的重尾特性，同時保留 G 的相關結構（長程記憶）。

    正規化理由：CMS 公式在 α/2 趨近 1 時尺度因子趨於無窮，
    除以中位數讓「典型路徑」保持 A=1（正常波動率），
    極端路徑仍有 A >> 1（波動率風暴），符合曼德博間歇性精神。
    α=2.0 退化為高斯（A=1）。α < 1.7 時重尾效果顯著。
    """
    if alpha >= 2.0:
        return np.ones(size)    # α=2 高斯極限：A=1，無重尾混合
    a = float(np.clip(alpha / 2, 0.51, 0.97))
    raw = np.abs(cms_stable(a, 1.0, size))
    med = np.median(raw)
    A   = raw / (med + 1e-10)
    return np.clip(A, 1e-4, 8.0)


# ══════════════════════════════════════════════════════════════
#  第二層：Hurst 指數 R/S 分析（1968 fBm / 尼羅河）
# ══════════════════════════════════════════════════════════════

def estimate_hurst_rs(ts: pd.Series) -> float:
    """R/S 重新調整極差分析（Mandelbrot & Wallis 1969）。

    曼德博在分析尼羅河水文資料時偏好此方法。
    R = 累積偏差的極差，S = 標準差。
    E[R/S(n)] ~ C · n^H，取對數迴歸得 H。

    相較 lag-variance 法：對非線性趨勢更穩健，但對短序列偏高估。
    """
    log_r = np.log(ts).diff().dropna().values
    n = len(log_r)

    sizes     = np.unique(np.round(np.geomspace(10, n // 2, 25)).astype(int))
    rs_means  = []
    valid_sizes = []

    for size in sizes:
        size = int(size)
        n_blocks = n // size
        if n_blocks < 2:
            continue
        rs_vals = []
        for b in range(n_blocks):
            block = log_r[b * size:(b + 1) * size]
            devs  = np.cumsum(block - block.mean())
            R = devs.max() - devs.min()
            S = block.std(ddof=1)
            if S > 0:
                rs_vals.append(R / S)
        if rs_vals:
            valid_sizes.append(size)
            rs_means.append(np.mean(rs_vals))

    if len(valid_sizes) < 3:
        return 0.5

    H = np.polyfit(np.log(valid_sizes), np.log(rs_means), 1)[0]
    return float(np.clip(H, 0.05, 0.95))


def build_cholesky_fgn(n: int, H: float):
    """建構 fGn 自協方差矩陣的 Cholesky 因子。"""
    if abs(H - 0.5) < 1e-4:
        return None
    g = np.zeros(n)
    g[0] = 1.0
    for k in range(1, n):
        km1 = abs(k - 1)
        g[k] = 0.5 * ((k + 1) ** (2 * H) - 2 * k ** (2 * H)
                       + (km1 ** (2 * H) if km1 > 0 else 0.0))
    Sigma = np.array([[g[abs(i - j)] for j in range(n)] for i in range(n)])
    Sigma += np.eye(n) * 1e-9
    return np.linalg.cholesky(Sigma)


# ══════════════════════════════════════════════════════════════
#  第三層：MFDFA 碎形譜寬 + MMAR 對數常態級聯
# ══════════════════════════════════════════════════════════════

_MFDFA_Q = np.array([-5, -3, -1, 1, 2, 3, 5, 7], dtype=float)
_DELTA_ALPHA_SCALE = 2.5   # Δα≈0.5 ↔ 級聯 λ²≈0.04（典型股票校準）
_MFDFA_BOOT_POOL = 120      # 自助法池大小（向量化批次估計，足夠 CI 用）
_DETREND_CACHE: dict = {}  # s → (pinv, T)


def _detrend_mats(s: int, order: int = 2) -> tuple:
    """快取去趨勢矩陣，回傳 pinv (order+1,s)、T (s,order+1)。"""
    key = (s, order)
    if key not in _DETREND_CACHE:
        t = np.arange(s, dtype=float)
        T = np.column_stack([t ** k for k in range(order, -1, -1)])
        _DETREND_CACHE[key] = (np.linalg.pinv(T), T)
    return _DETREND_CACHE[key]


def _segment_vars_at_scale(profiles: np.ndarray, s: int, order: int = 2) -> np.ndarray:
    """單一尺度 s 的段內方差，profiles shape=(B,N) → (B, 2*n_seg)。"""
    profiles = np.atleast_2d(profiles)
    B, N = profiles.shape
    n_seg = N // s
    if s < 12 or n_seg < 4:
        return np.empty((B, 0))

    fwd = profiles[:, : n_seg * s].reshape(B, n_seg, s)
    bwd_idx = np.arange(N - n_seg * s, N, dtype=int).reshape(n_seg, s)
    bwd = profiles[:, bwd_idx]
    segs = np.concatenate([fwd, bwd], axis=1)  # (B, 2*n_seg, s)

    pinv, T = _detrend_mats(s, order)
    fitted = (segs @ pinv.T) @ T.T                 # (B, M, s)
    res = segs - fitted
    return np.mean(res ** 2, axis=2)


def _fq_from_vars(seg_vars: np.ndarray, q_vals: np.ndarray) -> np.ndarray:
    """段內方差 → F_q，seg_vars (B,M) → Fq (B,Q)。"""
    arr = np.maximum(seg_vars, 1e-20)
    q = q_vals.astype(float)
    Fq = np.empty((arr.shape[0], len(q)))
    q0 = np.abs(q) < 1e-9
    if q0.any():
        Fq[:, q0] = np.exp(0.5 * np.mean(np.log(arr), axis=1))[:, None]
    nz = ~q0
    if nz.any():
        qn = q[nz]
        powered = arr[:, :, None] ** (qn[None, None, :] / 2.0)
        Fq[:, nz] = np.mean(powered, axis=1) ** (1.0 / qn)
    return Fq


def _mfdfa_fluctuations_batch(
    profiles: np.ndarray,
    scales: np.ndarray,
    q_vals: np.ndarray,
    order: int = 2,
) -> tuple:
    """向量化 MFDFA：profiles (B,N) → scales (S,), Fq (B,S,Q)。"""
    profiles = np.asarray(profiles, dtype=float)
    if profiles.ndim == 1:
        profiles = profiles[None, :]
    q_vals = np.asarray(q_vals, dtype=float)
    valid_scales = []
    fq_list = []

    for s in scales:
        s = int(s)
        seg_vars = _segment_vars_at_scale(profiles, s, order)
        if seg_vars.shape[1] < 8:
            continue
        valid_scales.append(s)
        fq_list.append(_fq_from_vars(seg_vars, q_vals))

    if not valid_scales:
        return np.empty(0), np.empty((profiles.shape[0], 0, len(q_vals)))

    return np.asarray(valid_scales, dtype=float), np.stack(fq_list, axis=1)


def _mfdfa_fluctuations(profile: np.ndarray, scales: np.ndarray,
                         q_vals: np.ndarray, order: int = 2) -> dict:
    """單序列包裝（相容舊介面）。"""
    sc, Fq_arr = _mfdfa_fluctuations_batch(profile, scales, q_vals, order)
    Fq = {float(q): Fq_arr[0, :, i].tolist() for i, q in enumerate(q_vals)}
    return {"scales": sc, "Fq": Fq}


def _hq_from_fq_batch(log_sc: np.ndarray, fq_arr: np.ndarray) -> np.ndarray:
    """log-log 斜率 → h(q)，fq_arr (B,S,Q) → h (B,Q)。"""
    log_fq = np.log(np.maximum(fq_arr, 1e-20))
    x = log_sc - log_sc.mean()
    denom = float(x @ x)
    y = log_fq - log_fq.mean(axis=1, keepdims=True)
    return np.einsum("bsq,s->bq", y, x) / denom


def _singularity_spectrum_batch(h_q: np.ndarray, q_vals: np.ndarray) -> np.ndarray:
    """批次奇異度譜寬，h_q (B,Q) → Δα (B,)。"""
    q = q_vals.astype(float)
    B, Q = h_q.shape
    if Q < 4:
        return np.full(B, 0.30)

    dh = np.gradient(h_q, q, axis=1)
    alpha = h_q + q[None, :] * dh
    tau = q[None, :] * h_q - 1.0
    f_alpha = q[None, :] * alpha - tau
    mask = f_alpha > 0
    delta = np.empty(B)
    for b in range(B):
        m = mask[b]
        if m.sum() < 3:
            m = np.ones(Q, dtype=bool)
        a = alpha[b, m]
        delta[b] = np.clip(a.max() - a.min(), 0.0, 2.0)
    return delta


def _hq_to_lambda2_batch(h_q: np.ndarray, q_vals: np.ndarray) -> np.ndarray:
    """直接擬合 h(q) = H − λ²/2·q，回傳 λ² (B,)。
    斜率 = −λ²/2，由 MFDFA 資料直接決定，無外部校準常數。"""
    q = q_vals.astype(float)
    q_c = q - q.mean()
    q_var = float(q_c @ q_c)
    if q_var < 1e-10:
        return np.full(h_q.shape[0], 0.02)
    h_c = h_q - h_q.mean(axis=1, keepdims=True)
    slope = (h_c * q_c[None, :]).sum(axis=1) / q_var   # slope = −λ²/2
    return np.clip(-2.0 * slope, 0.005, 0.30)


def _mfdfa_default() -> dict:
    return dict(H=0.5, delta_alpha=0.30, lambda2=0.014, h2=0.5,
                alpha_min=0.0, alpha_max=0.3, q_vals=_MFDFA_Q)


def _estimate_mfdfa_profiles_batch(
    profiles: np.ndarray,
    q_vals: np.ndarray | None = None,
) -> tuple:
    """profiles (B,N) → H (B,), Δα (B,), h_q (B,Q)。"""
    q_vals = _MFDFA_Q if q_vals is None else np.asarray(q_vals, dtype=float)
    profiles = np.asarray(profiles, dtype=float)
    if profiles.ndim == 1:
        profiles = profiles[None, :]
    B, N = profiles.shape
    if N < 500:
        d = _mfdfa_default()
        return (np.full(B, d["H"]), np.full(B, d["delta_alpha"]),
                np.full((B, len(q_vals)), d["h2"]), np.full(B, d["lambda2"]))

    scales = np.unique(np.round(np.geomspace(16, N // 4, 20)).astype(int))
    sc, fq = _mfdfa_fluctuations_batch(profiles, scales, q_vals)
    if len(sc) < 4:
        d = _mfdfa_default()
        return (np.full(B, d["H"]), np.full(B, d["delta_alpha"]),
                np.full((B, len(q_vals)), d["h2"]), np.full(B, d["lambda2"]))

    h_q = _hq_from_fq_batch(np.log(sc), fq)
    delta_alpha = _singularity_spectrum_batch(h_q, q_vals)
    lambda2 = _hq_to_lambda2_batch(h_q, q_vals)
    h2 = np.array([np.interp(2.0, q_vals, h_q[b]) for b in range(B)])
    H = np.clip(h2, 0.05, 0.95)
    return H, delta_alpha, h_q, lambda2


def _block_bootstrap_batch(
    returns: np.ndarray,
    n_pool: int,
    block_size: int = 252,
) -> np.ndarray:
    """向量化區塊自助法，回傳 (n_pool, usable_len)。"""
    r = np.asarray(returns, dtype=float)
    n = len(r)
    n_blocks = max(n // block_size, 4)
    usable = n_blocks * block_size
    blocks = np.zeros((n_blocks, block_size))
    for j in range(n_blocks):
        sl = r[j * block_size:min((j + 1) * block_size, n)]
        blocks[j, :len(sl)] = sl
    pick = np.random.randint(0, n_blocks, size=(n_pool, n_blocks))
    return blocks[pick].reshape(n_pool, usable)


def _singularity_spectrum(h_q: np.ndarray, q_vals: np.ndarray) -> tuple:
    """由 h(q) 計算 τ(q)、α(q)、f(α)，回傳譜寬 Δα。"""
    q = q_vals.astype(float)
    h = h_q.astype(float)
    if len(q) < 4:
        return 0.5, 0.0, q, h, q, h

    dh = np.gradient(h, q)
    tau = q * h - 1.0
    alpha = h + q * dh
    f_alpha = q * alpha - tau

    # 取 f(α)>0 的物理分支，避免數值噪聲
    mask = f_alpha > 0
    if mask.sum() < 3:
        mask = np.ones_like(q, dtype=bool)

    alpha_valid = alpha[mask]
    delta_alpha = float(np.max(alpha_valid) - np.min(alpha_valid))
    return float(np.clip(delta_alpha, 0.0, 2.0)), alpha, q, h, alpha, f_alpha


def estimate_mfdfa(returns: np.ndarray) -> dict:
    """MFDFA 估計 H=h(2) 與奇異度譜寬 Δα。

    Δα 越大 → 波動群聚越強（多碎形程度越高）
    H 取自 q=2 的廣義 Hurst 指數 h(q)
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 500:
        return _mfdfa_default()

    profile = np.cumsum(r - np.mean(r))
    H_b, da_b, h_q, lam2_b = _estimate_mfdfa_profiles_batch(profile[None, :])
    H = float(H_b[0])
    delta_alpha = float(da_b[0])
    h_arr = h_q[0]
    h2 = float(np.interp(2.0, _MFDFA_Q, h_arr))
    _, alpha, _, _, _, _ = _singularity_spectrum(h_arr, _MFDFA_Q)

    return dict(
        H=H, h2=h2, delta_alpha=delta_alpha,
        lambda2=float(lam2_b[0]),   # 直接擬合 h(q) 斜率，無外部校準常數
        alpha_min=float(np.min(alpha)), alpha_max=float(np.max(alpha)),
        q_vals=_MFDFA_Q, h_q=h_arr,
    )


def delta_alpha_to_lambda2(delta_alpha: float) -> float:
    """MFDFA 譜寬 Δα → 對數常態級聯 λ²（模擬內部用）。"""
    return float(np.clip((delta_alpha / _DELTA_ALPHA_SCALE) ** 2, 0.005, 0.30))


def delta_alpha_to_lambda2_arr(delta_alpha: np.ndarray) -> np.ndarray:
    """向量化 Δα → λ²。"""
    return np.clip((delta_alpha / _DELTA_ALPHA_SCALE) ** 2, 0.005, 0.30)


def _joint_block_bootstrap_batch(
    returns_m: np.ndarray,
    returns_eps: np.ndarray,
    n_pool: int,
    block_size: int = 252,
) -> tuple:
    """同一組區塊索引重抽市場與殘差，保留共變結構。"""
    r_m = np.asarray(returns_m, dtype=float)
    r_e = np.asarray(returns_eps, dtype=float)
    n = min(len(r_m), len(r_e))
    r_m, r_e = r_m[:n], r_e[:n]
    n_blocks = max(n // block_size, 4)
    usable = n_blocks * block_size
    blocks_m = np.zeros((n_blocks, block_size))
    blocks_e = np.zeros((n_blocks, block_size))
    for j in range(n_blocks):
        sl = slice(j * block_size, min((j + 1) * block_size, n))
        blocks_m[j, :sl.stop - sl.start] = r_m[sl]
        blocks_e[j, :sl.stop - sl.start] = r_e[sl]
    pick = np.random.randint(0, n_blocks, size=(n_pool, n_blocks))
    boot_m = blocks_m[pick].reshape(n_pool, usable)
    boot_e = blocks_e[pick].reshape(n_pool, usable)
    return boot_m, boot_e


def mfdfa_joint_bootstrap_samples(
    returns_m: np.ndarray,
    returns_eps: np.ndarray,
    n_samples: int = 300,
    block_size: int = 252,
) -> tuple:
    """聯合區塊自助法：同日區塊重抽市場+殘差，配對估 H 與 Δα。"""
    r_m = np.asarray(returns_m, dtype=float)
    r_e = np.asarray(returns_eps, dtype=float)
    n = min(len(r_m[np.isfinite(r_m)]), len(r_e[np.isfinite(r_e)]))
    if n < 500:
        est_m = estimate_mfdfa(r_m)
        est_e = estimate_mfdfa(r_e)
        return (np.full(n_samples, est_m["H"]),
                np.full(n_samples, est_m["delta_alpha"]),
                np.full(n_samples, est_m["lambda2"]),
                np.full(n_samples, est_e["H"]),
                np.full(n_samples, est_e["delta_alpha"]),
                np.full(n_samples, est_e["lambda2"]))

    n_pool = min(n_samples, _MFDFA_BOOT_POOL)
    boot_m, boot_e = _joint_block_bootstrap_batch(r_m, r_e, n_pool, block_size)
    prof_m = np.cumsum(boot_m - boot_m.mean(axis=1, keepdims=True), axis=1)
    prof_e = np.cumsum(boot_e - boot_e.mean(axis=1, keepdims=True), axis=1)
    H_m, da_m, _, lam2_m = _estimate_mfdfa_profiles_batch(prof_m)
    H_e, da_e, _, lam2_e = _estimate_mfdfa_profiles_batch(prof_e)

    if n_samples <= n_pool:
        return H_m[:n_samples], da_m[:n_samples], lam2_m[:n_samples], H_e[:n_samples], da_e[:n_samples], lam2_e[:n_samples]
    pick = np.random.randint(0, n_pool, size=n_samples)
    return H_m[pick], da_m[pick], lam2_m[pick], H_e[pick], da_e[pick], lam2_e[pick]


def mfdfa_bootstrap_samples(
    returns: np.ndarray,
    n_samples: int = 300,
    block_size: int = 252,
) -> tuple:
    """區塊自助法估計 H 與 Δα 的不確定性（向量化批次 MFDFA）。"""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 500:
        est = estimate_mfdfa(r)
        return np.full(n_samples, est["H"]), np.full(n_samples, est["delta_alpha"]), np.full(n_samples, est["lambda2"])

    n_pool = min(n_samples, _MFDFA_BOOT_POOL)
    boot = _block_bootstrap_batch(r, n_pool, block_size)
    profiles = np.cumsum(boot - boot.mean(axis=1, keepdims=True), axis=1)
    H_pool, da_pool, _, lam2_pool = _estimate_mfdfa_profiles_batch(profiles)

    if n_samples <= n_pool:
        return H_pool[:n_samples], da_pool[:n_samples], lam2_pool[:n_samples]
    pick = np.random.randint(0, n_pool, size=n_samples)
    return H_pool[pick], da_pool[pick], lam2_pool[pick]


def lognormal_cascade_batch(K: int, lambda2: float, n_steps: int,
                             n_batch: int) -> np.ndarray:
    """批次生成對數常態乘法級聯（向量化）。

    Mandelbrot-Calvet-Fisher MMAR 的交易時間核心。
    每一層 k 有 2^k 個區塊，各自有獨立的對數常態乘數。
    最終 2^K 個格子的權重即為多重碎形測度（交易時間增量）。

    回傳 shape = (n_steps, n_batch)，每列（路徑）正規化和為 1。
    """
    N = 2 ** K
    sigma_M = np.sqrt(lambda2 * np.log(2))
    mu_M    = -lambda2 * np.log(2) / 2   # E[M] = 1 條件

    # 每個最終格子累積 K 層乘數的對數
    # 在第 k 層，每 N/2^k 個相鄰格子共享同一個乘數
    log_w = np.zeros((N, n_batch))
    for k in range(K):
        n_groups   = 2 ** k
        group_size = N // n_groups
        M_k = np.random.normal(mu_M, sigma_M, (n_groups, n_batch))
        log_w += np.repeat(M_k, group_size, axis=0)

    weights = np.exp(log_w)  # shape: (N, n_batch)

    # 重採樣：N 格 → n_steps 格
    if N >= n_steps:
        g = N // n_steps
        extra = N - g * n_steps
        if extra > 0:
            weights = weights[:g * n_steps]
        weights = weights.reshape(n_steps, g, n_batch).sum(axis=1)
    else:
        # 插值（理論上不應發生）
        idx = np.round(np.linspace(0, N - 1, n_steps)).astype(int)
        weights = weights[idx]

    # 正規化：每條路徑的 Δθ 加總為 1
    col_sum = weights.sum(axis=0, keepdims=True)
    return weights / (col_sum + 1e-30)


# ══════════════════════════════════════════════════════════════
#  核心批次模擬（三層合一）
# ══════════════════════════════════════════════════════════════

def simulate_mmar_batch(
    L_m, L_res,
    n_steps: int, n_batch: int,
    H_m: float, H_res: float,
    alpha_m: float, alpha_eps: float,
    lambda2_m: float, lambda2_res: float,
    sigma_m: float, sigma_eps: float, mu_m: float,
    alpha_reg: float, beta_reg: float,
    levy_cap: float, last_price: float,
    mean_rs: float, std_rs: float,
    std_rs_har: float = 0.0,
    har_tau: int = 0,
) -> tuple:
    """完整 MMAR 批次模擬（三層曼德博）。

    市場因子：
      R_m = clip( μ_m + σ_m · √A_m · fGn_m(H_m) · cascade_m^H_m )

    個股特有因子：
      R_ε = σ_ε · √A_ε · fGn_ε(H_res) · cascade_ε^H_res

    個股報酬：
      R_s = clip( α + β·R_m + R_ε )

    其中 cascade^H 為多重碎形時間縮放（n·Δθ / mean(n·Δθ))^H，
    正規化確保整體波動率與歷史 σ 一致。
    """

    def _mmar_factor(L, H, sigma, lambda2, alpha_tail, n_steps, n_batch):
        """生成單一 MMAR 因子（α-穩定 fGn × 多重碎形時間）。"""
        # 1. fGn（長程記憶）
        Z = np.random.normal(0, 1, (n_steps, n_batch))
        fgn = (L @ Z) * sigma if L is not None else Z * sigma
        del Z

        # 2. 次高斯混合（重尾，α-穩定）：每條路徑的「波動率等級」不同
        A = positive_stable_mixing(alpha_tail, n_batch)  # shape (n_batch,)
        fgn *= np.sqrt(A)[np.newaxis, :]                # 廣播到每個時間步

        # 3. 多重碎形交易時間（波動率群聚）
        dtheta = lognormal_cascade_batch(K_CASCADE, lambda2, n_steps, n_batch)
        # 正規化縮放：per-path 均值為 1，不改變整體波動率水準
        mf = (n_steps * dtheta) ** H
        mf = mf / (mf.mean(axis=0, keepdims=True) + 1e-15)
        # 對稱 clip：log(mf) 標準差 ≈ H × sqrt(K × λ² × ln2)，取 ±2.5σ
        log_mf_std = H * np.sqrt(K_CASCADE * lambda2 * np.log(2))
        clip_lo = float(np.clip(np.exp(-2.5 * log_mf_std), 0.10, 0.50))
        clip_hi = float(np.clip(np.exp( 2.5 * log_mf_std),  2.0,  8.0))
        mf = np.clip(mf, clip_lo, clip_hi)

        return fgn * mf

    # 市場因子（α_m）與殘差因子（α_eps）
    R_m = np.clip(mu_m + _mmar_factor(L_m,  H_m,  sigma_m,   lambda2_m,  alpha_m,   n_steps, n_batch),
                  -levy_cap, levy_cap)

    # 個股特有因子
    R_eps = _mmar_factor(L_res, H_res, sigma_eps, lambda2_res, alpha_eps, n_steps, n_batch)

    # 個股總報酬（σ 校準：HAR-RV 起點 → 歷史均值回歸）
    R_raw = alpha_reg + beta_reg * R_m + R_eps
    r_mu  = float(R_raw.mean())
    r_sig = float(R_raw.std())
    if r_sig > 1e-12:
        _use_har = har_tau > 0 and std_rs_har > 1e-12
        _target0 = std_rs_har if _use_har else std_rs
        R_raw = r_mu + (R_raw - r_mu) * (_target0 / r_sig)
        if _use_har:
            # 逐步回歸：σ(t) = σ_hist + (σ_har - σ_hist) · exp(-t/τ)
            t = np.arange(1, n_steps + 1, dtype=float)[:, np.newaxis]
            scale = std_rs / std_rs_har + (1.0 - std_rs / std_rs_har) * np.exp(-t / har_tau)
            R_raw = R_raw * scale
    R_s = np.clip(R_raw, -levy_cap, levy_cap)
    del R_m, R_eps, R_raw

    fractal = np.empty((n_steps + 1, n_batch))
    fractal[0] = last_price
    fractal[1:] = last_price * np.exp(np.cumsum(R_s, axis=0))
    del R_s

    # GBM 對照
    gbm_r = np.random.normal(mean_rs, std_rs, (n_steps, n_batch))
    gbm = np.empty((n_steps + 1, n_batch))
    gbm[0] = last_price
    gbm[1:] = last_price * np.exp(np.cumsum(gbm_r, axis=0))

    return fractal, gbm


# ══════════════════════════════════════════════════════════════
#  通用校準：加權估計 + 終點主體分位數錨定
# ══════════════════════════════════════════════════════════════

def _exp_weights(n: int, halflife: int = _WEIGHT_HALFLIFE) -> np.ndarray:
    """指數衰減權重，最近資料權重高（oldest→0, newest→n-1）。"""
    ages = np.arange(n, dtype=float)[::-1]
    w = np.exp(-ages / max(halflife, 1))
    return w / w.sum()


def _wmean(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * x))


def _wstd(x: np.ndarray, w: np.ndarray, mu: float | None = None) -> float:
    mu = _wmean(x, w) if mu is None else mu
    return float(np.sqrt(np.sum(w * (x - mu) ** 2)))


_HAR_LEV_MULT = 1.5   # GJR 槓桿係數：下跌日 RV 貢獻放大（Taleb 下跌比上漲更重）


def har_rv_forecast(returns: np.ndarray) -> float:
    """HAR-RV（Corsi 2009）+ GJR 槓桿效應（Taleb 不對稱）。

    RV(t+1) = α + β_d·RV_lev_d(t) + β_w·RV_lev_w(t) + β_m·RV_lev_m(t)
    其中 RV_lev = RV × (1.5 if r<0 else 1.0)：下跌日波動率貢獻放大。
    回歸後自然估計更高係數 → 跌後預測波動率更高（槓桿效應）。
    """
    rv = returns ** 2
    # GJR 槓桿：下跌日 RV 貢獻 × 1.5（不對稱風險感知）
    rv_lev = rv * np.where(returns < 0, _HAR_LEV_MULT, 1.0)
    rv_d = rv_lev
    rv_w = pd.Series(rv_lev).rolling(5).mean().values
    rv_m = pd.Series(rv_lev).rolling(22).mean().values
    start = 22
    if len(rv) < start + 30:
        return float(np.std(returns))
    y = rv_d[start:]
    X = np.column_stack([
        np.ones(len(y)),
        rv_d[start - 1:-1],
        rv_w[start - 1:-1],
        rv_m[start - 1:-1],
    ])
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]
    if len(y) < 30:
        return float(np.std(returns))
    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
    last_w = rv_w[-1] if np.isfinite(rv_w[-1]) else float(np.mean(rv[-5:]))
    last_m = rv_m[-1] if np.isfinite(rv_m[-1]) else float(np.mean(rv[-22:]))
    rv_hat = float(coefs[0] + coefs[1] * rv_d[-1] + coefs[2] * last_w + coefs[3] * last_m)
    rv_hat = max(rv_hat, (returns.std() * 0.2) ** 2)   # 下界：歷史 σ 的 20%
    return float(np.sqrt(rv_hat))


def _weighted_beta(r_s: np.ndarray, r_m: np.ndarray,
                   w: np.ndarray) -> tuple:
    """加權 OLS：r_s = α + β·r_m + ε。"""
    mu_s = _wmean(r_s, w)
    mu_m = _wmean(r_m, w)
    cov  = float(np.sum(w * (r_s - mu_s) * (r_m - mu_m)))
    var_m = float(np.sum(w * (r_m - mu_m) ** 2))
    beta  = cov / var_m
    alpha = mu_s - beta * mu_m
    res   = r_s - (alpha + beta * r_m)
    return beta, alpha, res


def _hist_rolling_terminal_pct(s_hist: pd.Series, n_steps: int,
                                recent_days: int | None = None) -> np.ndarray:
    """歷史滾動 n 日累積報酬（%）。recent_days 限制取樣窗（近 N 交易日）。"""
    s = s_hist.astype(float)
    if recent_days and len(s) > recent_days:
        s = s.iloc[-recent_days:]
    daily = np.log(s).diff()
    roll = daily.rolling(n_steps).sum().dropna()
    return (np.exp(roll.values) - 1.0) * 100.0


def _hist_stress_terminal_pct(hist_full: np.ndarray,
                               n_worst: int = _STRESS_WORST_N) -> np.ndarray:
    """歷史最糟滾動窗（結構性下跌記憶，供左尾校準）。"""
    if len(hist_full) <= n_worst:
        return hist_full.copy()
    return np.sort(hist_full)[:n_worst]


def _compute_crisis_h_boost(
    r_s: np.ndarray,
    s_hist: pd.Series,
) -> tuple:
    """動態 H：偵測危機體制並計算 H 提升量（約瑟效應強化）。

    觸發條件（兩者同時成立）：
      1. 近 63 日回撤超過門檻（股價離高點夠遠）
      2. 近 10 日方向一致性夠高（持續往同方向走，不只是波動大）
      3. 趨勢方向為下跌（只強化下跌持續性，上漲不調整）

    危機強度 ∈ [0, 1]，線性映射 H_base → _CRISIS_H_MAX。
    """
    # 近 63 日回撤（從高點量）
    prices = s_hist.astype(float).values
    window = prices[-63:] if len(prices) >= 63 else prices
    peak    = float(window.max())
    current = float(prices[-1])
    drawdown = (peak - current) / peak if peak > 0 else 0.0

    # 近 10 日方向一致性
    recent_r = r_s[-10:] if len(r_s) >= 10 else r_s
    trend_sign = int(np.sign(float(recent_r.sum())))
    consistency = float(np.mean(np.sign(recent_r) == trend_sign)) if len(recent_r) > 0 else 0.5

    # 只在下跌趨勢觸發（上漲不強化，避免對稱性誤用）
    if trend_sign >= 0 or drawdown < _CRISIS_DD_THRESH or consistency < _CRISIS_CONSIST_MIN:
        return 0.0, drawdown, consistency, trend_sign

    # 回撤強度（0→1）× 一致性修正
    dd_level = min(1.0, (drawdown - _CRISIS_DD_THRESH) /
                        (_CRISIS_DD_FULL - _CRISIS_DD_THRESH))
    consist_factor = (consistency - _CRISIS_CONSIST_MIN) / (1.0 - _CRISIS_CONSIST_MIN)
    crisis_level = dd_level * consist_factor

    return float(crisis_level), drawdown, consistency, trend_sign


def _synthetic_terminal_pct(
    n_steps: int,
    H_m: float, H_res: float,
    lambda2_m: float, lambda2_res: float,
    alpha_m: float, alpha_eps: float,
    sigma_m: float, sigma_eps: float,
    mu_m: float, alpha_reg: float, beta_reg: float,
    mean_rs: float, std_rs: float,
    levy_cap: float,
    n_paths: int = _SYNTH_N_PATHS,
) -> np.ndarray:
    """曼德博盲樣合成法：用碎形參數生成合成終點報酬分布（%）。

    不做 HAR-RV 調整、不做 body 校準——純碎形結構的「假設歷史」。
    即使個股歷史短暫，只要參數估得到，就能合成出含正確頻率極端事件的分布。
    """
    L_m   = build_cholesky_fgn(n_steps, H_m)
    L_res = build_cholesky_fgn(n_steps, H_res)
    paths, _ = simulate_mmar_batch(
        L_m=L_m, L_res=L_res,
        n_steps=n_steps, n_batch=n_paths,
        H_m=H_m, H_res=H_res,
        lambda2_m=lambda2_m, lambda2_res=lambda2_res,
        alpha_m=alpha_m, alpha_eps=alpha_eps,
        sigma_m=sigma_m, sigma_eps=sigma_eps,
        mu_m=mu_m,
        alpha_reg=alpha_reg, beta_reg=beta_reg,
        levy_cap=levy_cap, last_price=100.0,
        mean_rs=mean_rs, std_rs=std_rs,
        std_rs_har=0.0, har_tau=0,   # 無 HAR-RV：合成歷史用無條件分布
    )
    return (paths[-1] / 100.0 - 1.0) * 100.0   # 終點累積報酬 %


def _build_calibration_knots(
    hist_recent: np.ndarray,
    hist_full: np.ndarray,
    knot_pcts: list | None = None,
    stress_tail_weight: float = _STRESS_TAIL_WEIGHT,
    full_tail_weight: float = _FULL_TAIL_WEIGHT,
) -> tuple:
    """體制感知參考分位數：P10–P90 用近期；左尾混入全樣本+壓力窗。"""
    knots = _BODY_KNOT_PCTS if knot_pcts is None else knot_pcts
    hist_stress = _hist_stress_terminal_pct(hist_full)
    ref_k = np.percentile(hist_recent, knots)
    full_k = np.percentile(hist_full, knots)
    stress_k = np.percentile(hist_stress, knots)

    for i, p in enumerate(knots):
        if p <= 5:
            w_s = stress_tail_weight
            w_f = full_tail_weight
            w_r = max(0.0, 1.0 - w_s - w_f)
            ref_k[i] = w_r * ref_k[i] + w_f * full_k[i] + w_s * stress_k[i]
        elif p <= 10:
            w_f = full_tail_weight * 0.5
            ref_k[i] = (1.0 - w_f) * ref_k[i] + w_f * full_k[i]
        elif p >= 95:
            w_f = full_tail_weight * 0.5
            ref_k[i] = (1.0 - w_f) * ref_k[i] + w_f * full_k[i]
    return knots, ref_k


def calibrate_terminal_body(
    sim_ret_pct: np.ndarray,
    hist_ref_knots: np.ndarray,
    knot_pcts: list | None = None,
    left_tail_blend: float = _LEFT_TAIL_BLEND,
    right_tail_blend: float = _RIGHT_TAIL_BLEND,
    tail_blend: float | None = None,
) -> np.ndarray:
    """單調分位數映射：P10–P90 錨近期參考；左尾硬錨、右尾軟錨。"""
    sim_ret_pct = np.asarray(sim_ret_pct, dtype=float)
    knots = _BODY_KNOT_PCTS if knot_pcts is None else knot_pcts
    hist_k = np.asarray(hist_ref_knots, dtype=float)
    if tail_blend is not None:
        right_tail_blend = tail_blend

    sim_k = np.percentile(sim_ret_pct, knots)
    for i in range(1, len(sim_k)):
        if sim_k[i] <= sim_k[i - 1]:
            sim_k[i] = sim_k[i - 1] + 1e-6

    hist_k = hist_k.copy()
    for i, p in enumerate(knots):
        if p <= 5:
            b = left_tail_blend
            hist_k[i] = (1 - b) * sim_k[i] + b * hist_k[i]
        elif p >= 95:
            b = right_tail_blend
            hist_k[i] = (1 - b) * sim_k[i] + b * hist_k[i]

    return np.interp(sim_ret_pct, sim_k, hist_k)


def enforce_left_tail_exceedance(
    all_paths: np.ndarray,
    last_price: float,
    hist_ref: np.ndarray,
    thresholds: tuple = (-30.0, -20.0, -10.0),
    tolerance: float = 0.92,
) -> np.ndarray:
    """若左尾超越率低於歷史，將最弱路徑下壓至對應門檻（結構性風險補強）。"""
    lp = float(last_price)
    out = all_paths.copy()
    sim_ret = (out[-1] / lp - 1.0) * 100.0

    for thr in sorted(thresholds, reverse=True):  # -10 → -20 → -30，避免重複下壓
        h_rate = float(np.mean(hist_ref <= thr))
        if h_rate < 0.003:
            continue
        m_rate = float(np.mean(sim_ret <= thr))
        if m_rate >= h_rate * tolerance:
            continue
        target = min(h_rate, m_rate + (h_rate - m_rate))
        n_push = int(round((target - m_rate) * len(sim_ret)))
        if n_push < 1:
            continue
        above = np.where(sim_ret > thr)[0]
        if len(above) == 0:
            continue
        pick = above[np.argsort(sim_ret[above])][:n_push]
        jitter = np.random.uniform(0.0, min(3.0, abs(thr) * 0.08), size=len(pick))
        target_end = lp * (1.0 + (thr - jitter) / 100.0)
        ratio = target_end / np.maximum(out[-1, pick], 1e-12)
        out[:, pick] *= ratio[np.newaxis, :]
        sim_ret = (out[-1] / lp - 1.0) * 100.0
    return out


def apply_terminal_body_calibration(
    all_paths: np.ndarray,
    last_price: float,
    hist_recent: np.ndarray,
    hist_full: np.ndarray | None = None,
    stress_tail_weight: float = _STRESS_TAIL_WEIGHT,
    full_tail_weight: float = _FULL_TAIL_WEIGHT,
    left_tail_blend: float = _LEFT_TAIL_BLEND,
    right_tail_blend: float = _RIGHT_TAIL_BLEND,
    enforce_left_tail: bool = True,
) -> tuple:
    """體制感知終點校準：近期主體 + 壓力左尾 + 可選超越率補強。"""
    lp = float(last_price)
    hist_full = hist_recent if hist_full is None else hist_full
    knots, ref_k = _build_calibration_knots(
        hist_recent, hist_full,
        stress_tail_weight=stress_tail_weight,
        full_tail_weight=full_tail_weight,
    )

    sim_end = all_paths[-1].astype(float)
    sim_ret = (sim_end / lp - 1.0) * 100.0
    cal_ret = calibrate_terminal_body(
        sim_ret, ref_k, knot_pcts=knots,
        left_tail_blend=left_tail_blend,
        right_tail_blend=right_tail_blend,
    )
    target_end = lp * (1.0 + cal_ret / 100.0)
    ratio = target_end / np.maximum(sim_end, 1e-12)
    calibrated = all_paths * ratio[np.newaxis, :]

    if enforce_left_tail:
        calibrated = enforce_left_tail_exceedance(
            calibrated, lp, hist_full, tolerance=0.92)

    cal_ret = (calibrated[-1] / lp - 1.0) * 100.0
    return calibrated, cal_ret, ratio


# ══════════════════════════════════════════════════════════════
#  主模擬流程
# ══════════════════════════════════════════════════════════════

def run_simulation(ticker, market_ticker, sim_start, hist_start,
                   n_steps, n_sims, levy_cap, seed, k_cascade=9,
                   calibrate_body=True, weight_halflife=_WEIGHT_HALFLIFE,
                   calibration_recent_days: int = _CALIB_RECENT_DAYS,
                   stress_tail_weight: float = _STRESS_TAIL_WEIGHT,
                   full_tail_weight: float = _FULL_TAIL_WEIGHT,
                   left_tail_blend: float = _LEFT_TAIL_BLEND,
                   enforce_left_tail: bool = True):
    global K_CASCADE
    K_CASCADE = k_cascade
    np.random.seed(seed)

    print(f"\n下載資料（{hist_start} ~ {sim_start}）...")
    s_hist = download_adjusted(ticker,        hist_start, sim_start)
    m_hist = download_adjusted(market_ticker, hist_start, sim_start)
    common = s_hist.index.intersection(m_hist.index)
    s_hist = s_hist.loc[common]
    m_hist = m_hist.loc[common]

    if len(s_hist) < 100:
        sys.exit(f"資料不足（{len(s_hist)} 筆）")
    if len(s_hist) < 500:
        print(f"  ⚠️  資料僅 {len(s_hist)} 筆，參數估計可能偏差")

    print(f"  資料期間：{common[0].date()} ~ {common[-1].date()}（{len(common)} 日）")

    # 對數報酬
    r_s = np.log(s_hist).diff().dropna()
    r_m = np.log(m_hist).diff().dropna()
    idx = r_s.index.intersection(r_m.index)
    r_s = r_s.loc[idx].values
    r_m = r_m.loc[idx].values

    # Beta / 殘差（指數衰減加權：近年權重高）
    w = _exp_weights(len(r_s), weight_halflife)
    beta_reg, alpha_reg, res = _weighted_beta(r_s, r_m, w)
    mu_m      = _wmean(r_m, w)
    sigma_m   = _wstd(r_m, w, mu_m)
    mu_res    = _wmean(res, w)
    sigma_eps = _wstd(res, w, mu_res)
    mean_rs   = _wmean(r_s, w)
    std_rs    = _wstd(r_s, w, mean_rs)
    ann_vol   = std_rs * np.sqrt(252) * 100
    _recent_vol_63 = float(np.std(r_s[-63:])) * np.sqrt(252) * 100
    _recent_vol_22 = float(np.std(r_s[-22:])) * np.sqrt(252) * 100
    # HAR-RV 條件化：用日/週/月 RV 預測近期波動率體制
    _sigma_m_hist   = sigma_m
    _sigma_eps_hist = sigma_eps
    sigma_m   = har_rv_forecast(r_m)
    sigma_eps = har_rv_forecast(res)
    # HAR-RV 上限：不超過歷史 σ 的 1.5 倍，避免短期 RV 尖峰過度放大
    sigma_m   = min(sigma_m,   _sigma_m_hist   * 1.5)
    sigma_eps = min(sigma_eps, _sigma_eps_hist * 1.5)
    std_rs_har   = float(np.sqrt(sigma_eps**2 + beta_reg**2 * sigma_m**2))
    _har_ann_vol = std_rs_har * np.sqrt(252) * 100
    # τ 自適應：上限 10 天（vol 半衰期），短模擬（<15步）縮至 5 天
    _HAR_TAU = min(10, max(5, n_steps // 2))
    hist_full = _hist_rolling_terminal_pct(s_hist, n_steps)
    hist_recent = _hist_rolling_terminal_pct(
        s_hist, n_steps, recent_days=calibration_recent_days)
    hist_term = hist_recent

    # ── 第一層：α-穩定尾部估計 ──
    alpha_s_daily   = hill_estimator(r_s)          # 標準 Hill（漲跌停截斷，偏高）
    alpha_m_daily   = hill_estimator(r_m)
    alpha_s_monthly = hill_estimator_monthly(s_hist)  # 月報酬（CLT 稀釋，偏高）
    alpha_m_monthly = hill_estimator_monthly(m_hist)
    # Taleb 不對稱：月報酬分別估計左尾（下跌）和右尾（上漲）的冪次指數
    # 日報酬受漲跌停截斷，無法分辨左右尾差異；月報酬可直接量化
    alpha_s_m_left  = hill_one_tail_monthly(s_hist, "left")   # 下跌方向尾部
    alpha_s_m_right = hill_one_tail_monthly(s_hist, "right")  # 上漲方向尾部
    # 截斷校正：利用已知截斷點 cap，對截斷 Pareto 做 MLE 還原真實 α
    alpha_s_trunc = hill_estimator_truncated(r_s, levy_cap)
    alpha_m_trunc = hill_estimator_truncated(r_m, levy_cap)
    # 模擬用 α：日報酬退化時以月報酬還原（台股漲跌停結構性問題）
    alpha_s, alpha_s_src = resolve_alpha_sim(alpha_s_trunc, alpha_s_monthly)
    alpha_m, alpha_m_src = resolve_alpha_sim(alpha_m_trunc, alpha_m_monthly)
    # Taleb 不對稱修正：若下跌尾部（alpha_left < 2）顯著重於上漲尾部，
    # 以月報酬左尾估計值取代整體 α（對下行風險更保守）
    _asym_gap = alpha_s_m_right - alpha_s_m_left   # >0 代表下跌更重尾
    _DID_ASYM = _asym_gap > 0.15 and alpha_s_m_left < 2.0
    if _DID_ASYM:
        _alpha_asym_left, _ = resolve_alpha_sim(1.99, alpha_s_m_left)  # 月左尾→模擬用
        if _alpha_asym_left < alpha_s:  # 僅在比現有估計更保守時採用
            alpha_s = _alpha_asym_left
            alpha_s_src = f"Taleb 左尾，月α_left={alpha_s_m_left:.2f}<α_right={alpha_s_m_right:.2f}"

    # ── 貝葉斯收縮：資料不足且尾部偏薄時向市場 α 先驗靠攏 ──
    # 只在個股 α > α_m（個股尾部比市場還薄）時才收縮，避免把重尾股拉輕。
    _bayes_w = float(min(1.0, len(r_s) / _BAYES_ALPHA_N_FULL))
    _alpha_s_own = alpha_s
    if _bayes_w < 1.0 and _alpha_s_own > alpha_m:
        alpha_s = _bayes_w * _alpha_s_own + (1.0 - _bayes_w) * alpha_m
        alpha_s = float(np.clip(alpha_s, 1.40, 1.98))

    # ── 第二層：Hurst R/S 分析 ──
    H_m_rs   = estimate_hurst_rs(m_hist)
    H_s_rs   = estimate_hurst_rs(s_hist)
    res_level = pd.Series(res).cumsum()
    res_level = res_level - res_level.min() + 1e-6
    H_res_rs  = estimate_hurst_rs(res_level)
    H_res_rs  = float(np.clip(H_res_rs, 0.05, 0.95))

    # ── 第三層：MFDFA 譜寬 Δα + MMAR 級聯 ──
    mfdfa_s   = estimate_mfdfa(r_s)
    mfdfa_m   = estimate_mfdfa(r_m)
    mfdfa_res = estimate_mfdfa(res)
    H_s_joint   = mfdfa_s["H"]
    H_m_joint   = mfdfa_m["H"]
    H_res_joint = mfdfa_res["H"]
    delta_alpha_s   = mfdfa_s["delta_alpha"]
    delta_alpha_m   = mfdfa_m["delta_alpha"]
    delta_alpha_res = mfdfa_res["delta_alpha"]
    lambda2_s   = mfdfa_s["lambda2"]
    lambda2_m   = mfdfa_m["lambda2"]
    lambda2_res = mfdfa_res["lambda2"]
    # 聯合自助法：同日區塊重抽市場+殘差（模擬驅動參數，配對一致）
    (_H_post_m, _da_post_m, _lam2_post_m,
     _H_post_res, _da_post_res, _lam2_post_res) = mfdfa_joint_bootstrap_samples(r_m, res, n_samples=n_sims)
    # 個股整體 MFDFA 僅作診斷對照
    _H_post_s, _da_post_s, _ = mfdfa_bootstrap_samples(r_s, n_samples=min(n_sims, 500))
    # ── 動態 H：危機體制偵測（約瑟效應強化）──
    _crisis_level, _crisis_dd, _crisis_consist, _crisis_sign = \
        _compute_crisis_h_boost(r_s, s_hist)
    if _crisis_level > 0:
        _H_boost_m   = (_CRISIS_H_MAX - H_m_joint)   * _crisis_level
        _H_boost_res = (_CRISIS_H_MAX - H_res_joint)  * _crisis_level
        _H_post_m   = np.clip(_H_post_m   + _H_boost_m,   0.50, _CRISIS_H_MAX)
        _H_post_res = np.clip(_H_post_res + _H_boost_res, 0.50, _CRISIS_H_MAX)

    _Hm_p5,  _Hm_p95  = float(np.percentile(_H_post_m, 5)),   float(np.percentile(_H_post_m, 95))
    _Hr_p5,  _Hr_p95  = float(np.percentile(_H_post_res, 5)),  float(np.percentile(_H_post_res, 95))
    _dam_p5, _dam_p95 = float(np.percentile(_da_post_m, 5)),  float(np.percentile(_da_post_m, 95))
    _dar_p5, _dar_p95 = float(np.percentile(_da_post_res, 5)), float(np.percentile(_da_post_res, 95))

    # 尖峰厚尾統計
    kurt  = float(stats.kurtosis(r_s, fisher=True))
    skew  = float(stats.skew(r_s))
    _, jb_p = stats.jarque_bera(r_s)

    last_price = float(s_hist.iloc[-1])
    end_date   = pd.bdate_range(start=sim_start, periods=n_steps + 1)[-1]
    end_label  = f"{end_date.year}/Q{(end_date.month-1)//3+1}"

    # α 估計品質診斷
    _trunc_is_heavy = alpha_s_trunc < 1.99   # 截斷校正後仍有重尾
    if _trunc_is_heavy:
        _trunc_tag  = f"← 截斷 Pareto MLE，還原真實重尾（修正量 {alpha_s_daily - alpha_s_trunc:+.3f}）"
    else:
        _trunc_tag  = f"← 日報酬 α≥2（漲跌停截斷）"
    _alpha_active = alpha_s < 1.99 or alpha_m < 1.99
    if _alpha_active:
        _alpha_note = (f"個股 {alpha_s_src}；市場 {alpha_m_src}"
                       if alpha_s < 1.99 or alpha_m < 1.99 else alpha_s_src)
    else:
        _alpha_note = ("α-stable 關閉；尾部由 Δα_m={:.4f}/Δα_ε={:.4f} 瀑布主導".format(
            delta_alpha_m, delta_alpha_res))

    _asym_tag = (f"↑ 左<右  Taleb 不對稱成立，差距 {_asym_gap:.2f}"
                 if _asym_gap > 0.10
                 else f"≈ 對稱（差距 {_asym_gap:.2f}）")

    print(f"\n── 三層曼德博參數估計 ──")
    print(f"  [第一層 α-穩定]")
    print(f"    月報酬 α={alpha_s_monthly:.3f}（市場={alpha_m_monthly:.3f}）← CLT 稀釋，偏高")
    print(f"    月報酬 左尾(↓) α={alpha_s_m_left:.3f}  右尾(↑) α={alpha_s_m_right:.3f}"
          f"  {_asym_tag}")
    print(f"    日報酬 α={alpha_s_daily:.3f}（市場={alpha_m_daily:.3f}）← 漲跌停截斷，偏高")
    print(f"    截斷校正 α={alpha_s_trunc:.3f}（市場={alpha_m_trunc:.3f}）{_trunc_tag}")
    _a_tag = "✓ 次高斯活躍" if _alpha_active else "○ 瀑布主導"
    _did_shrink = _bayes_w < 1.0 and _alpha_s_own > alpha_m
    if _did_shrink:
        _bayes_note = (f"  貝葉斯收縮 w={_bayes_w:.2f}"
                       f"（{len(r_s)}日＜{_BAYES_ALPHA_N_FULL}日，"
                       f"α_own={_alpha_s_own:.3f}>{alpha_m:.3f}=α_m → 收縮後 {alpha_s:.3f}）")
    elif _bayes_w < 1.0:
        _bayes_note = (f"  資料短（{len(r_s)}日）但 α_own={_alpha_s_own:.3f}≤α_m={alpha_m:.3f}，"
                       f"不收縮（個股已比市場重尾）")
    else:
        _bayes_note = f"  資料充足（{len(r_s)}日），不收縮"
    print(f"    模擬用  α_m={alpha_m:.3f}（市場）  α_ε={alpha_s:.3f}（殘差）  {_a_tag}")
    print(f"            ← {_alpha_note}")
    print(f"            ←{_bayes_note}")
    print(f"  [第二層 Hurst]  H(R/S)={H_s_rs:.4f}（市場={H_m_rs:.4f}，殘差={H_res_rs:.4f}）← 對照用")
    if _crisis_level > 0:
        print(f"  [危機 H ↑]  回撤={_crisis_dd*100:.1f}%  一致性={_crisis_consist*100:.0f}%"
              f"  強度={_crisis_level:.2f}"
              f"  H_m {H_m_joint:.4f}→{float(np.median(_H_post_m)):.4f}"
              f"  H_res {H_res_joint:.4f}→{float(np.median(_H_post_res)):.4f}")
    else:
        _dd_str = f"回撤={_crisis_dd*100:.1f}%"
        _cs_str = f"一致性={_crisis_consist*100:.0f}%"
        _why = ("上漲趨勢" if _crisis_sign > 0
                else f"回撤未達{_CRISIS_DD_THRESH*100:.0f}%" if _crisis_dd < _CRISIS_DD_THRESH
                else f"一致性未達{_CRISIS_CONSIST_MIN*100:.0f}%")
        print(f"  [危機 H —]  {_dd_str}  {_cs_str}  未觸發（{_why}），H 維持估計值")
    print(f"  [第三層 MFDFA]  模擬驅動（市場+殘差聯合自助法）")
    _hm_ci  = f"[{_Hm_p5:.4f},{_Hm_p95:.4f}]"
    _hr_ci  = f"[{_Hr_p5:.4f},{_Hr_p95:.4f}]"
    _dam_ci = f"[{_dam_p5:.4f},{_dam_p95:.4f}]"
    _dar_ci = f"[{_dar_p5:.4f},{_dar_p95:.4f}]"
    print(f"    市場  H=h(2)={H_m_joint:.4f}  Δα={delta_alpha_m:.4f}"
          f"  → λ²={lambda2_m:.4f}  不確定性 H {_hm_ci}  Δα {_dam_ci}")
    print(f"    殘差  H={H_res_joint:.4f}  Δα={delta_alpha_res:.4f}"
          f"  → λ²={lambda2_res:.4f}  不確定性 H {_hr_ci}  Δα {_dar_ci}"
          f"  K={K_CASCADE}（2^{K_CASCADE}={2**K_CASCADE} 格）")
    print(f"  [診斷] 個股整體 MFDFA（不直接進模擬，僅對照）"
          f"  H={H_s_joint:.4f}  Δα={delta_alpha_s:.4f}"
          f"  [α∈{mfdfa_s['alpha_min']:.3f},{mfdfa_s['alpha_max']:.3f}]")
    _kurt_note = "（受截斷分布限制，有限峰度不代表輕尾）" if kurt < 10 and not _trunc_is_heavy else ""
    print(f"  峰度={kurt:.2f}（常態=0）  偏態={skew:.2f}  JB p={jb_p:.1e}{_kurt_note}")
    print(f"  Beta={beta_reg:.4f}  σ_hist={ann_vol:.1f}%/yr"
          f"  近63日={_recent_vol_63:.1f}%  近22日={_recent_vol_22:.1f}%"
          f"  HAR-RV σ={_har_ann_vol:.1f}%/yr"
          f"  （τ={_HAR_TAU}d 均值回歸，加權半衰期 {weight_halflife} 日）")
    if calibrate_body:
        p50_r = float(np.percentile(hist_recent, 50))
        p50_f = float(np.percentile(hist_full, 50))
        print(f"  [校準] 體制感知：P10–P90 錨近 {calibration_recent_days} 日"
              f"（{len(hist_recent):,} 樣本）")
        print(f"         左尾混入全樣本+壓力窗（worst {_STRESS_WORST_N}）"
              f"  P50 近期 {p50_r:+.1f}% vs 全樣本 {p50_f:+.1f}%")
        if enforce_left_tail:
            print(f"         左尾超越率補強：≤-10/-20/-30% 對齊全樣本")
    print(f"  起始價：{last_price:,.2f}  |  終點：{end_date.strftime('%Y/%m/%d')}（{end_label}）")

    # 批次模擬（每批 MFDFA 抽樣 H 與 Δα→λ²，動態建構 Cholesky）
    n_batches = (n_sims + BATCH_SIZE - 1) // BATCH_SIZE
    all_paths = np.empty((n_steps + 1, n_sims))
    gbm_paths = np.empty((n_steps + 1, n_sims))

    batch_kw = dict(
        n_steps=n_steps,
        alpha_m=alpha_m, alpha_eps=alpha_s,
        sigma_m=sigma_m, sigma_eps=sigma_eps, mu_m=mu_m,
        alpha_reg=alpha_reg, beta_reg=beta_reg,
        levy_cap=levy_cap, last_price=last_price,
        mean_rs=mean_rs, std_rs=std_rs,
        std_rs_har=std_rs_har, har_tau=_HAR_TAU,
    )

    # 每批從 MFDFA 自助法抽 (H, Δα→λ²)
    _batch_idx = np.random.randint(0, n_sims, size=n_batches)

    print(f"執行 MMAR 模擬（{n_sims:,} 條路徑，{n_batches} 批次，聯合 H_m/H_ε+Δα 採樣）...")
    for b in range(n_batches):
        bi = _batch_idx[b]
        H_m_b   = float(_H_post_m[bi])
        H_res_b = float(_H_post_res[bi])
        L_m   = build_cholesky_fgn(n_steps, H_m_b)
        L_res = build_cholesky_fgn(n_steps, H_res_b)
        lo, hi = b * BATCH_SIZE, min((b + 1) * BATCH_SIZE, n_sims)
        fp, gp = simulate_mmar_batch(
            n_batch=hi - lo,
            L_m=L_m, L_res=L_res,
            H_m=H_m_b, H_res=H_res_b,
            lambda2_m=float(_lam2_post_m[bi]),
            lambda2_res=float(_lam2_post_res[bi]),
            **batch_kw,
        )
        all_paths[:, lo:hi] = fp
        gbm_paths[:, lo:hi] = gp
        print(f"\r  進度：{hi:,} / {n_sims:,}", end="", flush=True)
    print()

    # ── 曼德博盲樣合成法：資料不足時用碎形參數合成假設歷史補強體量校準 ──
    if _bayes_w < 1.0 and calibrate_body:
        _H_m_med    = float(np.median(_H_post_m))
        _H_res_med  = float(np.median(_H_post_res))
        _lam2_m_med = float(np.median(_lam2_post_m))
        _lam2_r_med = float(np.median(_lam2_post_res))
        print(f"  [合成] 曼德博盲樣合成（w={_bayes_w:.2f}，{_SYNTH_N_PATHS:,} 條路徑補強體量校準）...")
        synth_ret = _synthetic_terminal_pct(
            n_steps=n_steps,
            H_m=_H_m_med, H_res=_H_res_med,
            lambda2_m=_lam2_m_med, lambda2_res=_lam2_r_med,
            alpha_m=alpha_m, alpha_eps=alpha_s,
            sigma_m=_sigma_m_hist, sigma_eps=_sigma_eps_hist,
            mu_m=mu_m, alpha_reg=alpha_reg, beta_reg=beta_reg,
            mean_rs=mean_rs, std_rs=std_rs,
            levy_cap=levy_cap,
        )
        # 混入數量：(1-w)/w 倍的真實窗口數，不超過合成路徑總數
        _n_real = len(hist_recent)
        _n_add  = min(int((1.0 - _bayes_w) / _bayes_w * _n_real), len(synth_ret))
        _synth_mix = np.random.choice(synth_ret, size=_n_add, replace=False)
        hist_recent = np.concatenate([hist_recent, _synth_mix])
        hist_full   = np.concatenate([hist_full,   _synth_mix])
        print(f"         真實 {_n_real} 窗 + 合成 {_n_add} 窗 → 共 {len(hist_recent)} 窗")

    body_calibrated = False
    if calibrate_body and len(hist_recent) >= 50:
        all_paths, _, _ = apply_terminal_body_calibration(
            all_paths, last_price, hist_recent, hist_full,
            stress_tail_weight=stress_tail_weight,
            full_tail_weight=full_tail_weight,
            left_tail_blend=left_tail_blend,
            enforce_left_tail=enforce_left_tail,
        )
        body_calibrated = True

    return dict(
        ticker=ticker, last_price=last_price, ann_vol=ann_vol,
        beta=beta_reg, alpha_s=alpha_s,
        alpha_s_daily=alpha_s_daily, alpha_s_monthly=alpha_s_monthly,
        alpha_s_trunc=alpha_s_trunc,
        alpha_s_m_left=alpha_s_m_left, alpha_s_m_right=alpha_s_m_right,
        alpha_s_src=alpha_s_src, alpha_m_src=alpha_m_src,
        alpha_m=alpha_m,
        H_s=H_s_joint, H_m=H_m_joint, H_res=H_res_joint,
        H_s_rs=H_s_rs, H_m_rs=H_m_rs, H_res_rs=H_res_rs,
        delta_alpha_s=delta_alpha_s, delta_alpha_m=delta_alpha_m,
        delta_alpha_res=delta_alpha_res,
        lambda2_s=lambda2_s, lambda2_m=lambda2_m, lambda2_res=lambda2_res,
        kurt=kurt, skew=skew, jb_p=jb_p,
        sim_start=sim_start, end_date=end_date, end_label=end_label,
        n_steps=n_steps, n_sims=n_sims,
        all_paths=all_paths, gbm_paths=gbm_paths,
        sample_paths=all_paths[:, :5].copy(),
        sim_end=all_paths[-1], gbm_end=gbm_paths[-1],
        ret=(all_paths[-1] / last_price - 1) * 100,
        s_hist=s_hist, levy_cap=levy_cap,
        body_calibrated=body_calibrated,
        weight_halflife=weight_halflife,
        recent_vol_63=_recent_vol_63, recent_vol_22=_recent_vol_22,
        hist_full=hist_full, hist_recent=hist_recent,
        calibration_recent_days=calibration_recent_days,
        enforce_left_tail=enforce_left_tail,
    )


# ══════════════════════════════════════════════════════════════
#  吻合度檢定：模擬 vs 歷史分布
# ══════════════════════════════════════════════════════════════

_GOF_PCTS = [1, 5, 10, 25, 50, 75, 90, 95, 99]
_GOF_BODY_PCTS = [5, 10, 25, 50, 75, 90, 95]   # 主體評分用，排除 P1/P99 極端
_GOF_RECENT_DAYS = 1260                          # 近 5 年滾動窗（體制較近）


def _pct_table(x: np.ndarray, pcts: list) -> dict:
    vals = np.percentile(x, pcts)
    return {p: float(v) for p, v in zip(pcts, vals)}


def _exceedance_rate(x: np.ndarray, thr_pct: float) -> float:
    """累積報酬（%）超越門檻的比率（雙尾）。"""
    return float(np.mean((x <= thr_pct) | (x >= abs(thr_pct))))


def validate_simulation_fit(d: dict) -> dict:
    """模擬分布 vs 歷史分布吻合度（終點 + 日報酬）。"""
    s_hist = d["s_hist"]
    n_steps = d["n_steps"]
    lp = d["last_price"]
    levy_cap = d.get("levy_cap", 0.10)

    # ── 終點累積報酬 ──
    hist_term = (d["hist_full"] if "hist_full" in d
                 else _hist_rolling_terminal_pct(s_hist, n_steps))
    hist_calib = (d["hist_recent"] if "hist_recent" in d
                  else _hist_rolling_terminal_pct(
                      s_hist, n_steps,
                      recent_days=d.get("calibration_recent_days", _GOF_RECENT_DAYS)))
    sim_term  = d["ret"].astype(float)
    gbm_term  = (d["gbm_end"] / lp - 1.0) * 100.0

    ks_mmar, p_mmar = stats.ks_2samp(hist_term, sim_term)
    ks_gbm,  p_gbm  = stats.ks_2samp(hist_term, gbm_term)

    pct_hist = _pct_table(hist_term, _GOF_PCTS)
    pct_calib = _pct_table(hist_calib, _GOF_PCTS)
    pct_mmar = _pct_table(sim_term,  _GOF_PCTS)
    pct_gbm  = _pct_table(gbm_term,  _GOF_PCTS)
    pct_err_mmar = {p: pct_mmar[p] - pct_hist[p] for p in _GOF_PCTS}
    pct_err_calib = {p: pct_mmar[p] - pct_calib[p] for p in _GOF_PCTS}
    pct_err_gbm  = {p: pct_gbm[p]  - pct_hist[p] for p in _GOF_PCTS}
    pct_mae_mmar = float(np.mean([abs(v) for v in pct_err_mmar.values()]))
    pct_mae_gbm  = float(np.mean([abs(v) for v in pct_err_gbm.values()]))
    pct_mae_body_mmar = float(np.mean([abs(pct_err_calib[p]) for p in _GOF_BODY_PCTS]))
    pct_mae_body_gbm  = float(np.mean([abs(pct_err_gbm[p])  for p in _GOF_BODY_PCTS]))

    hist_recent_gof = hist_calib
    pct_hist_recent = _pct_table(hist_recent_gof, _GOF_BODY_PCTS)
    pct_err_recent  = {p: pct_mmar[p] - pct_hist_recent[p] for p in _GOF_BODY_PCTS}
    pct_mae_recent  = float(np.mean([abs(v) for v in pct_err_recent.values()]))

    term_ex_thr = [-40, -30, -20, -10, 10, 20, 30, 50]
    term_ex = {}
    for thr in term_ex_thr:
        if thr < 0:
            h = float(np.mean(hist_term <= thr))
            m = float(np.mean(sim_term  <= thr))
            g = float(np.mean(gbm_term  <= thr))
        else:
            h = float(np.mean(hist_term >= thr))
            m = float(np.mean(sim_term  >= thr))
            g = float(np.mean(gbm_term  >= thr))
        term_ex[thr] = dict(hist=h, mmar=m, gbm=g,
                            ratio_m=float(m / h) if h > 1e-6 else np.nan,
                            ratio_g=float(g / h) if h > 1e-6 else np.nan)

    # ── 日報酬 ──
    hist_daily = np.log(s_hist.astype(float)).diff().dropna().values
    sim_daily  = np.diff(np.log(d["all_paths"]), axis=0).ravel()

    ks_d_mmar, p_d_mmar = stats.ks_2samp(hist_daily, sim_daily)
    gbm_daily_ref = np.random.default_rng(42).normal(
        hist_daily.mean(), hist_daily.std(), size=len(sim_daily))
    ks_d_gbm, p_d_gbm = stats.ks_2samp(hist_daily, gbm_daily_ref)

    h_std = float(np.std(hist_daily))
    h_cap = levy_cap
    daily_ex = {
        f"|r|>{h_cap*100:.0f}%": dict(
            hist=float(np.mean(np.abs(hist_daily) >= h_cap * 0.99)),
            mmar=float(np.mean(np.abs(sim_daily)  >= h_cap * 0.99)),
        ),
        "|r|>2σ": dict(
            hist=float(np.mean(np.abs(hist_daily) >= 2 * h_std)),
            mmar=float(np.mean(np.abs(sim_daily)  >= 2 * h_std)),
        ),
        "|r|>3σ": dict(
            hist=float(np.mean(np.abs(hist_daily) >= 3 * h_std)),
            mmar=float(np.mean(np.abs(sim_daily)  >= 3 * h_std)),
        ),
    }
    for k, v in daily_ex.items():
        v["ratio"] = float(v["mmar"] / v["hist"]) if v["hist"] > 1e-8 else np.nan

    moments = dict(
        hist=dict(mean=float(np.mean(hist_daily)), std=h_std,
                  skew=float(stats.skew(hist_daily)),
                  kurt=float(stats.kurtosis(hist_daily, fisher=True))),
        mmar=dict(mean=float(np.mean(sim_daily)), std=float(np.std(sim_daily)),
                  skew=float(stats.skew(sim_daily)),
                  kurt=float(stats.kurtosis(sim_daily, fisher=True))),
    )

    # 綜合評分（百分位 MAE 60% + 尾部比率 40%）
    tail_ratios = [v["ratio_m"] for v in term_ex.values()
                   if np.isfinite(v.get("ratio_m", np.nan)) and v["hist"] > 0.001]
    tail_score = 100.0
    if tail_ratios:
        tail_dev = float(np.mean([abs(np.log(r)) for r in tail_ratios if r > 0]))
        tail_score = max(0.0, 100.0 - tail_dev * 80.0)
    body_score = max(0.0, 100.0 - pct_mae_body_mmar * 5.0)
    recent_score = max(0.0, 100.0 - pct_mae_recent * 5.0)
    score = 0.45 * body_score + 0.25 * recent_score + 0.30 * tail_score
    if score >= 75:
        verdict = "佳 — 模擬分布與歷史接近"
    elif score >= 50:
        verdict = "可接受 — 主體吻合，尾部略有偏差"
    else:
        verdict = "偏差大 — 建議調整參數或增加路徑數"

    return dict(
        n_steps=n_steps,
        hist_term=hist_term, sim_term=sim_term, gbm_term=gbm_term,
        hist_daily=hist_daily, sim_daily=sim_daily,
        ks_terminal=dict(mmar=ks_mmar, mmar_p=p_mmar, gbm=ks_gbm, gbm_p=p_gbm),
        ks_daily=dict(mmar=ks_d_mmar, mmar_p=p_d_mmar, gbm=ks_d_gbm, gbm_p=p_d_gbm),
        pct_hist=pct_hist, pct_mmar=pct_mmar, pct_gbm=pct_gbm,
        pct_err_mmar=pct_err_mmar, pct_err_calib=pct_err_calib,
        pct_err_gbm=pct_err_gbm,
        pct_mae_mmar=pct_mae_mmar, pct_mae_gbm=pct_mae_gbm,
        pct_mae_body_mmar=pct_mae_body_mmar, pct_mae_body_gbm=pct_mae_body_gbm,
        hist_recent=hist_recent_gof, pct_hist_recent=pct_hist_recent,
        hist_calib=hist_calib, pct_calib=pct_calib,
        pct_mae_recent=pct_mae_recent,
        term_ex=term_ex, daily_ex=daily_ex, moments=moments,
        score=score, body_score=body_score, recent_score=recent_score,
        tail_score=tail_score, verdict=verdict,
        verdict_en=_gof_verdict_en(score),
    )


def print_fit_validation(fit: dict, end_label: str, n_steps: int,
                         recent_vol_63: float = 0.0, recent_vol_22: float = 0.0):
    """印出吻合度檢定報告。"""
    print(f"\n── 吻合度檢定（模擬 vs 歷史）──")
    print(f"  終點：{n_steps} 日累積報酬  |  全樣本 {len(fit['hist_term']):,}  |  "
          f"校準窗 {len(fit.get('hist_calib', fit['hist_recent'])):,}  |  "
          f"模擬 {len(fit['sim_term']):,}")
    if "pct_calib" in fit:
        dr = fit["pct_calib"][50] - fit["pct_hist"][50]
        print(f"  體制漂移：校準窗 P50 {fit['pct_calib'][50]:+.1f}%"
              f"  vs 全樣本 {fit['pct_hist'][50]:+.1f}%  （Δ {dr:+.1f}pp）")
    print(f"  綜合評分：{fit['score']:.0f}/100"
          f"（主體 {fit['body_score']:.0f}  近5年 {fit.get('recent_score', 0):.0f}"
          f"  尾部 {fit['tail_score']:.0f}）  → {fit['verdict']}")

    ks = fit["ks_terminal"]
    print(f"\n  KS 檢定（終點累積報酬）")
    print(f"    MMAR  D={ks['mmar']:.4f}  p={ks['mmar_p']:.2e}"
          f"  |  GBM  D={ks['gbm']:.4f}  p={ks['gbm_p']:.2e}")
    print(f"    主體 MAE（P5–P95，vs 校準窗）：MMAR {fit['pct_mae_body_mmar']:.2f}%  "
          f"GBM {fit['pct_mae_body_gbm']:.2f}%  "
          f"{'← MMAR 較佳' if fit['pct_mae_body_mmar'] < fit['pct_mae_body_gbm'] else '← GBM 較佳'}")
    print(f"    主體 MAE（vs 全樣本）："
          f"{float(np.mean([abs(fit['pct_err_mmar'][p]) for p in _GOF_BODY_PCTS])):.2f}%")
    print(f"    全域 MAE（含 P1/P99）：MMAR {fit['pct_mae_mmar']:.2f}%  "
          f"GBM {fit['pct_mae_gbm']:.2f}%")
    if "pct_mae_recent" in fit:
        print(f"    近5年主體 MAE：{fit['pct_mae_recent']:.2f}%"
              f"（歷史窗 {len(fit['hist_recent']):,} 樣本，體制較近）")
    p50_gap = fit["pct_mmar"][50] - fit["pct_hist"][50]
    if abs(p50_gap) > 5 and abs(fit["pct_mmar"][50] - fit["pct_gbm"][50]) < 3:
        print(f"    ⚠️  P50 偏高 {p50_gap:+.1f}pp：MMAR 與 GBM 均值漂移相近，"
              f"可能因全樣本 μ 含早期熊市，與近期體制不同")

    print(f"\n  百分位對照（終點累積報酬 %，至 {end_label}）")
    print(f"    {'':>4}  {'歷史':>8}  {'MMAR':>8}  {'Δ':>7}  {'GBM':>8}  {'Δ':>7}")
    print(f"    {'-'*46}")
    for p in _GOF_PCTS:
        h, m, g = fit["pct_hist"][p], fit["pct_mmar"][p], fit["pct_gbm"][p]
        print(f"    P{p:<3}  {h:>+8.1f}  {m:>+8.1f}  {m-h:>+7.1f}  {g:>+8.1f}  {g-h:>+7.1f}")

    print(f"\n  超越率（終點累積報酬，歷史 vs MMAR vs GBM）")
    print(f"    {'門檻':>6}  {'歷史':>7}  {'MMAR':>7}  {'GBM':>7}  {'MMAR/歷史':>9}")
    print(f"    {'-'*42}")
    for thr, v in fit["term_ex"].items():
        tag = f"≤{thr}%" if thr < 0 else f"≥+{thr}%"
        r = f"{v['ratio_m']:.2f}x" if np.isfinite(v["ratio_m"]) else "—"
        print(f"    {tag:>6}  {v['hist']*100:>6.2f}%  {v['mmar']*100:>6.2f}%  "
              f"{v['gbm']*100:>6.2f}%  {r:>9}")

    ks_d = fit["ks_daily"]
    print(f"\n  日報酬分布")
    print(f"    KS  MMAR D={ks_d['mmar']:.4f} p={ks_d['mmar_p']:.2e}  "
          f"GBM D={ks_d['gbm']:.4f} p={ks_d['gbm_p']:.2e}")
    print(f"    {'指標':<8}  {'歷史':>8}  {'MMAR':>8}  {'MMAR/歷史':>9}")
    print(f"    {'-'*38}")
    for k, v in fit["daily_ex"].items():
        r = f"{v['ratio']:.2f}x" if np.isfinite(v["ratio"]) else "—"
        print(f"    {k:<8}  {v['hist']*100:>7.2f}%  {v['mmar']*100:>7.2f}%  {r:>9}")

    mo = fit["moments"]
    _rv63 = recent_vol_63 / np.sqrt(252) / 100
    _rv22 = recent_vol_22 / np.sqrt(252) / 100
    _rv63_s = f"/{_rv63*100:.3f}%(近63日)/{_rv22*100:.3f}%(近22日)" if _rv63 > 0 else ""
    print(f"    矩估計  μ={mo['hist']['mean']*100:.3f}%/{mo['mmar']['mean']*100:.3f}%  "
          f"σ={mo['hist']['std']*100:.3f}%{_rv63_s}/{mo['mmar']['std']*100:.3f}%  "
          f"偏態={mo['hist']['skew']:.2f}/{mo['mmar']['skew']:.2f}  "
          f"峰度={mo['hist']['kurt']:.2f}/{mo['mmar']['kurt']:.2f}  （歷史/近期/MMAR）")


def plot_fit_validation(fit: dict, d: dict, out_path: str,
                        show_inline: bool = True):
    """吻合度圖：QQ 圖 + 百分位誤差 + 超越率。"""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    _verdict = fit.get("verdict_en") or _gof_verdict_en(fit["score"])
    fig.suptitle(f"{d['ticker']}  Goodness-of-Fit  "
                 f"score={fit['score']:.0f}/100  ({_verdict})",
                 fontsize=12, fontweight="bold")

    # QQ：終點累積報酬
    ax = axes[0, 0]
    q_pts = np.linspace(0.5, 99.5, 80)
    h_q = np.percentile(fit["hist_term"], q_pts)
    m_q = np.percentile(fit["sim_term"],  q_pts)
    g_q = np.percentile(fit["gbm_term"],  q_pts)
    lim = max(np.abs(h_q).max(), np.abs(m_q).max()) * 1.1
    ax.scatter(h_q, m_q, s=18, color="crimson", alpha=0.7, label="MMAR")
    ax.scatter(h_q, g_q, s=18, color="steelblue", alpha=0.5, label="GBM")
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("Historical quantile (%)")
    ax.set_ylabel("Simulated quantile (%)")
    ax.set_title(f"Q-Q  Terminal {fit['n_steps']}d Return")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # QQ：日報酬
    ax = axes[0, 1]
    q_pts_d = np.linspace(0.5, 99.5, 80)
    hd = np.percentile(fit["hist_daily"] * 100, q_pts_d)
    md = np.percentile(fit["sim_daily"]  * 100, q_pts_d)
    lim_d = max(np.abs(hd).max(), np.abs(md).max()) * 1.1
    ax.scatter(hd, md, s=18, color="crimson", alpha=0.7)
    ax.plot([-lim_d, lim_d], [-lim_d, lim_d], "k--", lw=1)
    ax.set_xlim(-lim_d, lim_d); ax.set_ylim(-lim_d, lim_d)
    ax.set_xlabel("Historical daily quantile (%)")
    ax.set_ylabel("Simulated daily quantile (%)")
    ax.set_title("Q-Q  Daily Return")
    ax.grid(True, alpha=0.3)

    # 百分位誤差棒
    ax = axes[1, 0]
    pcts = [5, 10, 25, 50, 75, 90, 95]
    x = np.arange(len(pcts))
    err_m = [fit["pct_err_mmar"][p] for p in pcts]
    err_g = [fit["pct_err_gbm"][p]  for p in pcts]
    w = 0.35
    ax.bar(x - w/2, err_m, w, label="MMAR-Hist", color="crimson", alpha=0.8)
    ax.bar(x + w/2, err_g, w, label="GBM-Hist",  color="steelblue", alpha=0.6)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{p}" for p in pcts])
    ax.set_ylabel("Percentile error (pp)")
    ax.set_title(f"Percentile Error  MAE={fit['pct_mae_mmar']:.1f}% (MMAR)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    # 超越率比較
    ax = axes[1, 1]
    thrs = [-30, -20, -10, 10, 20, 30]
    labels = [f"<={t}%" if t < 0 else f">=+{t}%" for t in thrs]
    x = np.arange(len(thrs))
    w = 0.25
    h_v = [fit["term_ex"][t]["hist"] * 100 for t in thrs]
    m_v = [fit["term_ex"][t]["mmar"] * 100 for t in thrs]
    g_v = [fit["term_ex"][t]["gbm"]  * 100 for t in thrs]
    ax.bar(x - w, h_v, w, label="Hist", color="black", alpha=0.6)
    ax.bar(x,     m_v, w, label="MMAR", color="crimson", alpha=0.8)
    ax.bar(x + w, g_v, w, label="GBM",  color="steelblue", alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, rotation=30)
    ax.set_ylabel("Exceedance rate (%)")
    ax.set_title("Terminal Return Exceedance")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    _save_or_show(fig, out_path, show_inline=show_inline)


# ══════════════════════════════════════════════════════════════
#  輸出與繪圖
# ══════════════════════════════════════════════════════════════

def quarter_label(dt):
    return f"{dt.year}/Q{(dt.month-1)//3+1}"


def print_results(d: dict, currency: str) -> dict:
    lp      = d["last_price"]
    sim_end = d["sim_end"]
    gbm_end = d["gbm_end"]
    ret     = d["ret"]
    el      = d["end_label"]

    p10, p25, p50, p75, p90 = np.percentile(sim_end, [10, 25, 50, 75, 90])
    gbm_p50 = np.percentile(gbm_end, 50)

    print(f"\n{'='*58}")
    _cal_tag = "  [主體校準✓]" if d.get("body_calibrated") else ""
    print(f"  {d['ticker']}  MMAR 結果（至 {el}，{d['n_sims']:,} 條路徑）{_cal_tag}")
    print(f"{'='*58}")

    print(f"\n── 模擬驅動參數（R_s = α + β·R_m + R_ε）──")
    _a_on = d['alpha_m'] < 1.99 or d['alpha_s'] < 1.99
    print(f"  α_m={d['alpha_m']:.3f}（市場）  α_ε={d['alpha_s']:.3f}（殘差）"
          f"  β={d['beta']:.3f}  {'✓次高斯' if _a_on else '○瀑布'}")
    if d.get('alpha_s_src'):
        print(f"  α來源  個股：{d['alpha_s_src']}  市場：{d.get('alpha_m_src', '—')}")
    print(f"  H_m ={d['H_m']:.4f}  H_ε ={d['H_res']:.4f}  h(2) 記憶（0.5=隨機漫步）"
          f"  [R/S對照 市場{d['H_m_rs']:.4f} 殘差{d['H_res_rs']:.4f}]")
    print(f"  Δα_m={d['delta_alpha_m']:.4f}  Δα_ε={d['delta_alpha_res']:.4f}"
          f"  → λ²_m={d['lambda2_m']:.4f}  λ²_ε={d['lambda2_res']:.4f}（級聯）")
    print(f"  [診斷] 個股整體  H={d['H_s']:.4f}  Δα={d['delta_alpha_s']:.4f}"
          f"  [R/S {d['H_s_rs']:.4f}]  ← 不直接進模擬")
    _a_trunc = d.get('alpha_s_trunc', d['alpha_s'])
    if d['alpha_s'] < 1.99 or d['alpha_m'] < 1.99:
        _tail_note = (f"次高斯+瀑布（α_ε={d['alpha_s']:.2f} α_m={d['alpha_m']:.2f}；"
                      f"日報酬截斷校正={_a_trunc:.2f}）")
    elif _a_trunc < 1.5:
        _tail_note = f"極重尾 ✓（截斷校正 α={_a_trunc:.3f}）"
    elif _a_trunc < 1.99:
        _tail_note = f"重尾（截斷校正 α={_a_trunc:.3f}<2）"
    else:
        _tail_note = (f"α-stable 關閉；尾部由 Δα_m={d['delta_alpha_m']:.4f}/"
                      f"Δα_ε={d['delta_alpha_res']:.4f} 瀑布主導")
    print(f"  峰度={d['kurt']:.2f}（常態=0）  偏態={d['skew']:.2f}  {_tail_note}")

    print(f"\n── 百分位價格 ──")
    for label, val in [("P10 最壞",p10),("P25",p25),("P50 中位",p50),("P75",p75),("P90 最佳",p90)]:
        print(f"  {label:<8}  {val:>10,.2f} {currency}  ({(val/lp-1)*100:+.1f}%)")
    print(f"  GBM P50   {gbm_p50:>10,.2f} {currency}  ({(gbm_p50/lp-1)*100:+.1f}%)  ← 常態對照")

    print(f"\n── 機率表（MMAR vs GBM）──")
    print(f"  {'':12}  {'MMAR':>9}  {'GBM':>9}")
    print(f"  {'-'*34}")
    for sign, thresholds in [(-1, [0.10, 0.20, 0.30, 0.40]), (+1, [0.10, 0.20, 0.30, 0.50])]:
        for t in thresholds:
            if sign == -1:
                pf = np.mean(sim_end < lp*(1-t)); pg = np.mean(gbm_end < lp*(1-t))
                tag = f"下跌 >{t*100:.0f}%"
            else:
                pf = np.mean(sim_end > lp*(1+t)); pg = np.mean(gbm_end > lp*(1+t))
                tag = f"上漲 >{t*100:.0f}%"
            flag = "  (+重尾)" if sign == -1 and (pf - pg) > 0.005 else ""
            print(f"  {tag:<12}  {pf*100:>8.2f}%  {pg*100:>8.2f}%{flag}")

    var95  = lp - np.percentile(sim_end, 5)
    cvar95 = lp - np.mean(sim_end[sim_end < np.percentile(sim_end, 5)])
    var99  = lp - np.percentile(sim_end, 1)
    cvar99 = lp - np.mean(sim_end[sim_end < np.percentile(sim_end, 1)])
    print(f"\n── 風險指標 ──")
    print(f"  95% VaR：   {var95:>10,.2f} {currency}（{var95/lp*100:.1f}%）")
    print(f"  95% CVaR：  {cvar95:>10,.2f} {currency}（{cvar95/lp*100:.1f}%）")
    print(f"  99% VaR：   {var99:>10,.2f} {currency}（{var99/lp*100:.1f}%）")
    print(f"  99% CVaR：  {cvar99:>10,.2f} {currency}（{cvar99/lp*100:.1f}%）")

    # Max Drawdown 分布（路徑級，涵蓋整個模擬期間）
    _paths = d["all_paths"].astype(float)          # (n_steps+1, n_sims)
    _peaks = np.maximum.accumulate(_paths, axis=0)
    _dd    = (_peaks - _paths) / (_peaks + 1e-12)  # 每步回撤率
    _mdd   = _dd.max(axis=0) * 100                 # 每條路徑最大回撤 %
    _mdd_p10 = float(np.percentile(_mdd, 10))
    _mdd_p25 = float(np.percentile(_mdd, 25))
    _mdd_p50 = float(np.percentile(_mdd, 50))
    _mdd_p75 = float(np.percentile(_mdd, 75))
    _mdd_p90 = float(np.percentile(_mdd, 90))
    print(f"\n── 最大回撤分布（路徑期間內峰谷） ──")
    print(f"  P10（最小回撤）  {_mdd_p10:>6.1f}%")
    print(f"  P25              {_mdd_p25:>6.1f}%")
    print(f"  P50（中位）      {_mdd_p50:>6.1f}%")
    print(f"  P75              {_mdd_p75:>6.1f}%")
    print(f"  P90（最大回撤）  {_mdd_p90:>6.1f}%")
    for thr in [10, 20, 30, 40]:
        prob = float(np.mean(_mdd > thr)) * 100
        print(f"  P(MDD>{thr}%)       {prob:>6.1f}%")

    # 密集區
    kde       = stats.gaussian_kde(ret, bw_method=0.15)
    x_range   = np.linspace(ret.min(), ret.max(), 3000)
    y_range   = kde(x_range)
    mode_ret  = float(x_range[np.argmax(y_range)])
    half      = np.max(y_range) * 0.50
    dense     = x_range[y_range >= half]
    dense_lo  = float(dense[0]); dense_hi = float(dense[-1])
    dense_pct = np.mean((ret >= dense_lo) & (ret <= dense_hi)) * 100

    print(f"\n── 密集區（KDE 半峰寬）──")
    print(f"  峰值（最可能）：{mode_ret:+.1f}%  →  {lp*(1+mode_ret/100):>10,.2f} {currency}")
    print(f"  密集區下緣：   {dense_lo:+.1f}%  →  {lp*(1+dense_lo/100):>10,.2f} {currency}")
    print(f"  密集區上緣：   {dense_hi:+.1f}%  →  {lp*(1+dense_hi/100):>10,.2f} {currency}")
    print(f"  涵蓋路徑：     {dense_pct:.1f}%")

    bins  = [(-100,-30),(-30,-10),(-10,10),(10,30),(30,60),(60,100),(100,9999)]
    blabs = ["大跌>-30%","中跌-30~-10%","平盤-10~+10%","小漲+10~+30%","中漲+30~+60%","大漲+60~+100%","暴漲>+100%"]
    print(f"\n── 各報酬區間路徑比例 ──")
    for (lo, hi), lab in zip(bins, blabs):
        pct = np.mean((ret >= lo) & (ret < hi)) * 100
        print(f"  {lab:<20}  {pct:>5.1f}%  {'█'*int(pct/2)}")

    lo_p = lp * (1 + dense_lo/100)
    hi_p = lp * (1 + dense_hi/100)
    # 回檔進場：曾觸及=路徑最高價；終盤=模擬最後一日收盤
    max_paths = np.max(d["all_paths"][1:], axis=0)
    print(f"\n── 等待回檔進場情境（至 {el}）──")
    print(f"  {'回檔':^6}  {'進場價':^10}  {'觸及率':^7}  {'曾>現價':^7}  {'曾>+20%':^8}"
          f"  {'終盤>現價':^8}  {'終盤>+20%':^8}")
    print(f"  {'-'*68}")
    pullback_data = []
    for dip in [0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30]:
        entry   = lp * (1 - dip)
        touched = np.min(d["all_paths"][1:], axis=0) < entry
        max_t   = max_paths[touched]
        end_t   = sim_end[touched]
        if len(max_t) < 50:
            continue
        p_peak_win = np.mean(max_t > lp)
        p_peak_g20 = np.mean(max_t > entry * 1.20)
        p_end_win  = np.mean(end_t > lp)
        p_end_g20  = np.mean(end_t > entry * 1.20)
        p_g5  = np.mean(max_t > entry * 1.05)
        p_g10 = np.mean(max_t > entry * 1.10)
        p_g15 = np.mean(max_t > entry * 1.15)
        pullback_data.append((
            dip, entry, touched.mean(),
            p_peak_win, p_g5, p_g10, p_g15, p_peak_g20,
            p_end_win, p_end_g20,
        ))
        print(f"  -{dip*100:.0f}%    {entry:>10,.2f}   {touched.mean()*100:>5.1f}%  "
              f"{p_peak_win*100:>5.1f}%  {p_peak_g20*100:>6.1f}%  "
              f"{p_end_win*100:>6.1f}%  {p_end_g20*100:>6.1f}%")

    # 以現價買入：觸及=期間最高價；終盤達標=最後一日收盤
    p_end_above_lp = float(np.mean(sim_end > lp))
    p_end_above_20 = float(np.mean(sim_end > lp * 1.20))
    print(f"\n── 以現價買入情境（至 {el}）──")
    print(f"  終盤勝率（收盤>現價）：{p_end_above_lp*100:.1f}%  |  "
          f"終盤獲利>20%：{p_end_above_20*100:.1f}%")
    print(f"  {'目標漲幅':^8}  {'目標價':^10}  {'觸及率':^7}  {'終盤達標':^8}")
    print(f"  {'-'*40}")
    for gain in [0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        target = lp * (1 + gain)
        touch  = float(np.mean(max_paths > target))
        end_ok = float(np.mean(sim_end > target))
        print(f"  +{gain*100:.0f}%      {target:>10,.2f}   {touch*100:>5.1f}%  "
              f"{end_ok*100:>6.1f}%")

    print(f"\n── 操作價位建議（至 {el}）──")
    print(f"  持有不動：    {lo_p:>10,.2f} ~ {hi_p:>10,.2f} {currency}")
    print(f"  停損評估點：  {lo_p:>10,.2f} {currency}  （跌破密集區下緣）")
    print(f"  硬停損：      {lp*0.80:>10,.2f} {currency}  （-20%）")
    print(f"  減碼 1/3：    {hi_p:>10,.2f} {currency}  （+{dense_hi:.0f}%，密集區上緣）")
    print(f"  再減碼 1/3：  {p50:>10,.2f} {currency}  （P50 中位數）")

    return dict(p10=p10, p50=p50, p90=p90,
                mode_ret=mode_ret, dense_lo=dense_lo, dense_hi=dense_hi,
                var95=var95, cvar95=cvar95)


def plot_results(d: dict, sd: dict, currency: str, out_path: str,
                 show_inline: bool = True):
    lp        = d["last_price"]
    ret       = d["ret"]
    el        = d["end_label"]
    all_paths = d["all_paths"]
    gbm_paths = d["gbm_paths"]
    sim_end   = d["sim_end"]
    gbm_end   = d["gbm_end"]
    s_hist    = d["s_hist"]

    future = pd.bdate_range(start=d["sim_start"], periods=d["n_steps"] + 1)
    hist   = s_hist.index[-120:]

    fig = plt.figure(figsize=(15, 12))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.30)
    ax_path = fig.add_subplot(gs[0, :])
    ax_dist = fig.add_subplot(gs[1, 0])
    ax_prob = fig.add_subplot(gs[1, 1])

    # 路徑圖
    med   = np.percentile(all_paths, 50, axis=1)
    upper = np.percentile(all_paths, 90, axis=1)
    lower = np.percentile(all_paths, 10, axis=1)
    ax_path.plot(hist, s_hist.iloc[-120:], color="black", lw=1.8, label="Historical")
    ax_path.plot(future, med,   color="crimson", lw=2.5,  label=f"P50 MMAR ({el})")
    ax_path.plot(future, upper, color="crimson", lw=1.2, ls=":", alpha=0.9, label="P90")
    ax_path.plot(future, lower, color="crimson", lw=1.2, ls=":", alpha=0.9, label="P10")
    ax_path.fill_between(future, lower, upper, color="crimson", alpha=0.10)
    for i, c in enumerate(["royalblue", "forestgreen", "darkorange"]):
        ax_path.plot(future, d["sample_paths"][:, i], color=c, lw=0.9, ls="--", alpha=0.5)
    ax_path.axvline(pd.Timestamp(d["sim_start"]), color="darkgreen", lw=1.5,
                    label=f"Start {d['sim_start']}")
    ax_path.set_title(
        f"{d['ticker']}  MMAR Simulation  "
        f"Hm={d['H_m']:.2f} Hres={d['H_res']:.2f}  "
        f"dAm={d['delta_alpha_m']:.2f} dAe={d['delta_alpha_res']:.2f}  "
        f"vol={d['ann_vol']:.0f}%/yr",
        fontsize=11, fontweight="bold")
    ax_path.set_ylabel(f"Price ({currency})")
    ax_path.grid(True, alpha=0.3)
    ax_path.legend(loc="upper left", fontsize=8, ncol=2)

    # 報酬分布
    ret_g = (gbm_end / lp - 1) * 100
    # 用 P1~P99 決定 x 軸範圍，避免極端尾部路徑把主體壓縮成細線
    x_lo = min(np.percentile(ret, 1),   np.percentile(ret_g, 1))
    x_hi = max(np.percentile(ret, 99),  np.percentile(ret_g, 99))
    x_r  = np.linspace(x_lo, x_hi, 500)
    # hist 只畫落在 x 範圍內的路徑（其餘 1% 尾部不影響視覺）
    ret_clip  = ret[  (ret   >= x_lo) & (ret   <= x_hi)]
    ret_g_clip= ret_g[(ret_g >= x_lo) & (ret_g <= x_hi)]
    ax_dist.hist(ret_clip,   bins=80, density=True, alpha=0.45, color="crimson",   label="MMAR")
    ax_dist.hist(ret_g_clip, bins=80, density=True, alpha=0.35, color="steelblue", label="GBM")
    kde = stats.gaussian_kde(ret, bw_method=0.15)
    ax_dist.plot(x_r, kde(x_r), "crimson", lw=1.8)
    ax_dist.plot(x_r, stats.norm.pdf(x_r, ret.mean(), ret.std()), "r--", lw=1.2, label="Normal Fit")
    lo, hi = sd["dense_lo"], sd["dense_hi"]
    ax_dist.axvspan(lo, hi, alpha=0.10, color="purple")
    ax_dist.axvline(lo, color="purple", lw=1.2, ls="--", label=f"Dense {lo:+.0f}%~{hi:+.0f}%")
    ax_dist.axvline(hi, color="purple", lw=1.2, ls="--")
    ax_dist.axvline(0,  color="black",  lw=0.8, ls="--")
    ax_dist.set_xlim(x_lo, x_hi)
    ax_dist.set_title(f"Return Distribution  alpha={d['alpha_s']:.2f}  kurtosis={d['kurt']:.1f}",
                      fontsize=10, fontweight="bold")
    ax_dist.set_xlabel(f"Return to {el} (%)  [P1~P99 shown]")
    ax_dist.set_ylabel("Density")
    ax_dist.legend(fontsize=8)
    ax_dist.grid(True, alpha=0.3)

    # 機率條形
    labels_b = ["-40%", "-30%", "-20%", "-10%", "+10%", "+20%", "+30%", "+50%"]
    tholds   = [0.60, 0.70, 0.80, 0.90, 1.10, 1.20, 1.30, 1.50]
    signs    = [-1]*4 + [+1]*4
    pf_b = [np.mean(sim_end < lp*t) if s==-1 else np.mean(sim_end > lp*t)
            for t, s in zip(tholds, signs)]
    pg_b = [np.mean(gbm_end < lp*t) if s==-1 else np.mean(gbm_end > lp*t)
            for t, s in zip(tholds, signs)]
    x_b = np.arange(len(labels_b)); w = 0.35
    colors_b = ["#d62728"]*4 + ["#2ca02c"]*4
    ax_prob.bar(x_b-w/2, [v*100 for v in pf_b], w, label="MMAR", color=colors_b, alpha=0.8)
    ax_prob.bar(x_b+w/2, [v*100 for v in pg_b], w, label="GBM",  color=colors_b, alpha=0.35)
    ax_prob.set_xticks(x_b); ax_prob.set_xticklabels(labels_b, fontsize=8)
    ax_prob.set_ylabel("Probability (%)")
    ax_prob.set_title(f"Probability Comparison  MMAR vs GBM  (to {el})",
                      fontsize=10, fontweight="bold")
    ax_prob.legend(fontsize=8)
    ax_prob.axvline(3.5, color="grey", lw=0.8, ls="--", alpha=0.6)
    ax_prob.grid(True, alpha=0.3, axis="y")

    _save_or_show(fig, out_path, show_inline=show_inline)


# ══════════════════════════════════════════════════════════════
#  Colab one-shot runner
# ══════════════════════════════════════════════════════════════

def run_colab(
    ticker: str,
    *,
    market: str | None = None,
    sim_start: str | None = None,
    hist_start: str = "2020-01-01",
    n_steps: int = 252,
    n_sims: int = 5000,
    seed: int = 42,
    cascade_levels: int = 12,
    calibrate_body: bool = True,
    weight_halflife: int = _WEIGHT_HALFLIFE,
    calibration_recent_days: int = _CALIB_RECENT_DAYS,
    stress_tail_weight: float = _STRESS_TAIL_WEIGHT,
    enforce_left_tail: bool = True,
    output_dir: str | None = None,
    show_plots: bool = True,
) -> dict:
    """Run a full MMAR simulation in Google Colab or Jupyter."""
    ticker = ticker.upper()
    preset = detect_market(ticker)
    market = market or preset["index"]
    currency = preset["currency"]
    cap = preset["cap"]
    out_dir = output_dir or OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    tag = ticker.replace("^", "").replace(".", "_")

    if sim_start is None:
        tmp = yf.download(ticker, period="5d", progress=False)
        if tmp.empty:
            raise ValueError(f"Cannot download latest price for {ticker}")
        sim_start = str(tmp.index[-1].date())

    print(f"\n{'='*58}")
    print(f"  MMAR simulation (Colab): {ticker}")
    print(f"  Market: {market}  |  Start: {sim_start}  |  Paths: {n_sims:,}")
    print(f"{'='*58}")

    data = run_simulation(
        ticker, market, sim_start, hist_start,
        n_steps, n_sims, cap, seed,
        k_cascade=cascade_levels,
        calibrate_body=calibrate_body,
        weight_halflife=weight_halflife,
        calibration_recent_days=calibration_recent_days,
        stress_tail_weight=stress_tail_weight,
        enforce_left_tail=enforce_left_tail,
    )
    stats = print_results(data, currency)
    mmar_path = os.path.join(out_dir, f"{tag}_mmar.png")
    plot_results(data, stats, currency, mmar_path, show_inline=show_plots)
    fit = validate_simulation_fit(data)
    print_fit_validation(fit, data["end_label"], data["n_steps"],
                         recent_vol_63=data.get("recent_vol_63", 0.0),
                         recent_vol_22=data.get("recent_vol_22", 0.0))
    gof_path = os.path.join(out_dir, f"{tag}_mmar_gof.png")
    plot_fit_validation(fit, data, gof_path, show_inline=show_plots)
    return dict(data=data, stats=stats, fit=fit,
                mmar_path=mmar_path, gof_path=gof_path,
                currency=currency)


def print_report(result: dict) -> None:
    """Re-print the full text report (same as run_colab stdout)."""
    d = result["data"]
    fit = result["fit"]
    currency = result.get("currency") or detect_market(d["ticker"])["currency"]
    print_results(d, currency)
    print_fit_validation(
        fit, d["end_label"], d["n_steps"],
        recent_vol_63=d.get("recent_vol_63", 0.0),
        recent_vol_22=d.get("recent_vol_22", 0.0),
    )


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="完整忠於曼德博精神的碎形市場模擬（α-穩定 + R/S + MMAR）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("ticker")
    p.add_argument("--market",     help="市場指數（自動偵測）")
    p.add_argument("--start",      default=None)
    p.add_argument("--hist-start", default="2020-01-01",
                   help="歷史訓練窗起點（預設 2020-01-01≈5年；"
                        "要含金融海嘯用 2008-01-01）")
    p.add_argument("--steps",      type=int, default=252)
    p.add_argument("--sims",       type=int, default=10000,
                   help="路徑數（預設 100000）")
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--cascade-levels", type=int, default=12,
                   help="MMAR 級聯層數 K（預設 12=4096格，波動群聚延伸至季~年尺度）")
    p.add_argument("--out",        default=None)
    p.add_argument("--no-body-calibrate", action="store_true",
                   help="關閉終點主體 P10–P90 分位數錨定")
    p.add_argument("--weight-halflife", type=int, default=_WEIGHT_HALFLIFE,
                   help="指數衰減半衰期（日，預設 504≈2年）")
    p.add_argument("--calib-recent-days", type=int, default=_CALIB_RECENT_DAYS,
                   help="主體校準用近 N 交易日（預設 1260≈5年）")
    p.add_argument("--stress-tail-weight", type=float, default=_STRESS_TAIL_WEIGHT,
                   help="左尾混入最糟滾動窗權重（0–1，預設 0.40）")
    p.add_argument("--no-left-tail-enforce", action="store_true",
                   help="關閉左尾超越率補強（≤-10/-20/-30%%）")
    return p.parse_args()


def main():
    args   = parse_args()
    ticker = args.ticker.upper()
    preset = detect_market(ticker)
    market = args.market or preset["index"]
    currency = preset["currency"]
    cap    = preset["cap"]
    tag = ticker.replace("^", "").replace(".", "_")
    out_path = args.out or os.path.join(OUTPUT_DIR, f"{tag}_mmar.png")
    _ensure_output_dir(out_path)

    if args.start:
        sim_start = args.start
    else:
        tmp = yf.download(ticker, period="5d", progress=False)
        sim_start = str(tmp.index[-1].date())

    print(f"\n{'='*58}")
    print(f"  MMAR 碎形模擬（忠於曼德博原典）：{ticker}")
    print(f"  市場指數：{market}  |  貨幣：{currency}")
    print(f"  起點：{sim_start}  |  步數：{args.steps}d  |  路徑：{args.sims:,}")
    print(f"{'='*58}")

    data   = run_simulation(ticker, market, sim_start, args.hist_start,
                            args.steps, args.sims, cap, args.seed,
                            k_cascade=args.cascade_levels,
                            calibrate_body=not args.no_body_calibrate,
                            weight_halflife=args.weight_halflife,
                            calibration_recent_days=args.calib_recent_days,
                            stress_tail_weight=args.stress_tail_weight,
                            enforce_left_tail=not args.no_left_tail_enforce)
    stats_d = print_results(data, currency)
    plot_results(data, stats_d, currency, out_path, show_inline=IN_COLAB)
    fit     = validate_simulation_fit(data)
    print_fit_validation(fit, data["end_label"], data["n_steps"],
                         recent_vol_63=data.get("recent_vol_63", 0.0),
                         recent_vol_22=data.get("recent_vol_22", 0.0))
    gof_path = out_path.replace(".png", "_gof.png")
    if gof_path == out_path:
        gof_path = os.path.join(OUTPUT_DIR, f"{tag}_mmar_gof.png")
    plot_fit_validation(fit, data, gof_path, show_inline=IN_COLAB)


if __name__ == "__main__":
    main()
