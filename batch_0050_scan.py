"""
batch_0050_scan.py — 0050 成分股 MMAR 批次掃描 + 選股

用法：
  uv run batch_0050_scan.py
  uv run batch_0050_scan.py --sims 3000 --top 3
  uv run batch_0050_scan.py --codes 2330,2454,3711   # 只跑指定幾檔

輸出：
  output/0050_scan_YYYYMMDD/
    reports/{code}_{name}.txt   每檔完整文字報告
    summary.csv                 全成分評分表
    TOP3_操作建議.txt           明日開盤參考（非投資建議）
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
import sys
import traceback
from datetime import date, datetime

import numpy as np
import pandas as pd
import yfinance as yf

from real_fractal_sim import (
    MARKET_PRESETS,
    detect_market,
    print_fit_validation,
    print_results,
    run_simulation,
    validate_simulation_fit,
)

# 0050 成分股（代號、名稱、權重%）— 使用者提供
CONSTITUENTS_0050: list[tuple[str, str, float]] = [
    ("2330", "台積電", 58.37),
    ("2454", "聯發科", 6.23),
    ("2308", "台達電", 4.56),
    ("2317", "鴻海", 3.42),
    ("3711", "日月光投控", 1.85),
    ("2303", "聯電", 1.47),
    ("2383", "台光電", 1.43),
    ("3037", "欣興", 1.27),
    ("2345", "智邦", 1.25),
    ("2327", "國巨*", 1.21),
    ("2891", "中信金", 1.21),
    ("2382", "廣達", 1.04),
    ("2881", "富邦金", 1.01),
    ("2360", "致茂", 0.94),
    ("2882", "國泰金", 0.92),
    ("3017", "奇鋐", 0.87),
    ("2885", "元大金", 0.70),
    ("2887", "台新新光金", 0.66),
    ("2357", "華碩", 0.60),
    ("6669", "緯穎", 0.57),
    ("2412", "中華電", 0.55),
    ("2884", "玉山金", 0.52),
    ("3231", "緯創", 0.52),
    ("2886", "兆豐金", 0.51),
    ("1303", "南亞", 0.50),
    ("2344", "華邦電", 0.50),
    ("2368", "金像電", 0.49),
    ("2301", "光寶科", 0.44),
    ("2883", "凱基金", 0.44),
    ("2890", "永豐金", 0.42),
    ("7769", "鴻勁", 0.42),
    ("2408", "南亞科", 0.40),
    ("1216", "統一", 0.37),
    ("2059", "川湖", 0.36),
    ("3008", "大立光", 0.35),
    ("3661", "世芯-KY", 0.34),
    ("2449", "京元電子", 0.33),
    ("2892", "第一金", 0.33),
    ("3653", "健策", 0.33),
    ("2880", "華南金", 0.32),
    ("2603", "長榮", 0.25),
    ("5880", "合庫金", 0.25),
    ("2002", "中鋼", 0.22),
    ("2395", "研華", 0.22),
    ("1301", "台塑", 0.20),
    ("4904", "遠傳", 0.19),
    ("3045", "台灣大", 0.17),
    ("2207", "和泰車", 0.14),
    ("6919", "康霈*", 0.09),
    ("6505", "台塑化", 0.07),
]

TW_CAP = MARKET_PRESETS["TW"]["cap"]
TW_INDEX = MARKET_PRESETS["TW"]["index"]
CURRENCY = MARKET_PRESETS["TW"]["currency"]


def _safe_name(name: str) -> str:
    return name.replace("/", "_").replace("*", "star").replace(" ", "")


def _pullback_metrics(data: dict, dip: float = 0.05) -> dict:
    """回檔進場指標（與 print_results 同邏輯）。"""
    lp = data["last_price"]
    paths = data["all_paths"]
    sim_end = data["sim_end"]
    entry = lp * (1.0 - dip)
    max_paths = np.max(paths[1:], axis=0)
    touched = np.min(paths[1:], axis=0) < entry
    max_t = max_paths[touched]
    end_t = sim_end[touched]
    n = len(max_t)
    if n < 50:
        return dict(touch=np.nan, peak_win=np.nan, peak_p20=np.nan,
                    end_win=np.nan, end_p20=np.nan, entry=entry)
    return dict(
        touch=float(touched.mean()),
        peak_win=float(np.mean(max_t > lp)),
        peak_p20=float(np.mean(max_t > entry * 1.20)),
        end_win=float(np.mean(end_t > lp)),
        end_p20=float(np.mean(end_t > entry * 1.20)),
        entry=entry,
    )


def _spot_metrics(data: dict) -> dict:
    """現價買入終盤指標。"""
    lp = data["last_price"]
    sim_end = data["sim_end"]
    return dict(
        end_win=float(np.mean(sim_end > lp)),
        end_p20=float(np.mean(sim_end > lp * 1.20)),
    )


def compute_trade_score(data: dict, fit: dict, weight_pct: float) -> dict:
    """綜合選股分數（越高越優先列入觀察清單）。

    構成：
      35% 風險調整報酬  median / |P10|
      25% 模擬可信度    GOF 綜合評分
      25% -5% 回檔後獲利>20% 機率
      15% 0050 權重     流動性／指數代表性
    篩選：GOF < 45 或 P50 < 0 者扣分
    """
    lp = float(data["last_price"])
    sim_end = data["sim_end"]
    p10, p50, p90 = np.percentile(sim_end, [10, 50, 90])
    r10 = (p10 / lp - 1.0) * 100.0
    r50 = (p50 / lp - 1.0) * 100.0
    r90 = (p90 / lp - 1.0) * 100.0

    pb5 = _pullback_metrics(data, 0.05)
    pb10 = _pullback_metrics(data, 0.10)
    spot = _spot_metrics(data)

    gof = float(fit["score"])
    downside = max(abs(min(r10, -0.5)), 0.5)
    risk_adj = r50 / downside

    s_risk = np.clip(risk_adj * 8.0, 0, 100)
    s_gof = np.clip(gof, 0, 100)
    s_pb = np.clip((pb5["peak_p20"] or 0) * 100, 0, 100)
    s_end = np.clip(spot["end_win"] * 100, 0, 100)
    s_wt = np.clip(weight_pct * 8.0, 0, 100)  # 58% -> 100 cap

    trade = (0.30 * s_risk + 0.25 * s_gof + 0.20 * s_pb
             + 0.10 * s_end + 0.15 * s_wt)
    if gof < 45:
        trade *= 0.55
    if r50 < 0:
        trade *= 0.40

    return dict(
        last_price=lp,
        p10_price=p10, p50_price=p50, p90_price=p90,
        ret_p10=r10, ret_p50=r50, ret_p90=r90,
        gof_score=gof,
        body_score=float(fit["body_score"]),
        tail_score=float(fit["tail_score"]),
        risk_adj=risk_adj,
        dip5_touch=pb5["touch"],
        dip5_peak_win=pb5["peak_win"],
        dip5_peak_p20=pb5["peak_p20"],
        dip5_end_win=pb5["end_win"],
        dip5_end_p20=pb5["end_p20"],
        dip5_entry=pb5["entry"],
        dip10_touch=pb10["touch"],
        dip10_peak_p20=pb10["peak_p20"],
        spot_end_win=spot["end_win"],
        spot_end_p20=spot["end_p20"],
        trade_score=float(trade),
        weight_pct=weight_pct,
    )


def build_text_report(
    code: str,
    name: str,
    weight: float,
    data: dict,
    stats: dict,
    fit: dict,
    metrics: dict,
) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print(f"0050 成分股 MMAR 報告")
        print(f"代號：{code}  名稱：{name}  0050權重：{weight:.2f}%")
        print(f"產生時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        print_results(data, CURRENCY)
        print_fit_validation(fit, data["end_label"], data["n_steps"])
        print(f"\n── 選股指標摘要 ──")
        print(f"  綜合選股分數：{metrics['trade_score']:.1f}")
        print(f"  P10/P50/P90 報酬：{metrics['ret_p10']:+.1f}% / "
              f"{metrics['ret_p50']:+.1f}% / {metrics['ret_p90']:+.1f}%")
        print(f"  風險調整（P50/|P10|）：{metrics['risk_adj']:.2f}")
        print(f"  -5% 回檔 觸及/曾>現價/曾>+20%/終盤>現價/終盤>+20%："
              f"{(metrics['dip5_touch'] or 0)*100:.1f}% / "
              f"{(metrics['dip5_peak_win'] or 0)*100:.1f}% / "
              f"{(metrics['dip5_peak_p20'] or 0)*100:.1f}% / "
              f"{(metrics['dip5_end_win'] or 0)*100:.1f}% / "
              f"{(metrics['dip5_end_p20'] or 0)*100:.1f}%")
        print(f"  現價買入 終盤>現價/終盤>+20%："
              f"{metrics['spot_end_win']*100:.1f}% / "
              f"{metrics['spot_end_p20']*100:.1f}%")
        print(f"\n⚠️  本報告為統計情境分析，非投資建議。")
    return buf.getvalue()


def build_top3_report(
    picks: list[dict],
    sim_start: str,
    trade_date: str,
    out_dir: str,
) -> str:
    lines = [
        "=" * 60,
        "  0050 成分股 MMAR 掃描 — 明日開盤操作參考",
        "=" * 60,
        f"  掃描輸出：{os.path.abspath(out_dir)}",
        f"  模擬起點：{sim_start}",
        f"  建議參考日：{trade_date} 開盤",
        "",
        "  【免責】以下為模擬統計篩選結果，不構成投資建議。",
        "  實盤請自行確認基本面、產業消息、流動性與風險承受度。",
        "",
        "  選股邏輯（綜合分數）：",
        "    35% 風險調整報酬（一年 P50 / |P10|）",
        "    25% GOF 吻合度（模擬可信度）",
        "    25% -5% 回檔後一年獲利>20% 機率",
        "    15% 0050 權重（流動性）",
        "",
        "─" * 60,
        "  建議觀察前三檔",
        "─" * 60,
    ]
    for i, p in enumerate(picks, 1):
        lines += [
            "",
            f"  #{i}  {p['code']} {p['name']}  （0050 權重 {p['weight_pct']:.2f}%）",
            f"      綜合分數：{p['trade_score']:.1f}  |  GOF：{p['gof_score']:.0f}",
            f"      現價（模擬基準）：{p['last_price']:,.2f} TWD",
            f"      一年情境 P10/P50/P90：{p['ret_p10']:+.1f}% / "
            f"{p['ret_p50']:+.1f}% / {p['ret_p90']:+.1f}%",
            f"      建議進場參考（-5% 回檔限價）：{p['dip5_entry']:,.2f}",
            f"      或：開盤市價買入，停損參考 P10 約 {p['p10_price']:,.2f} "
            f"（{p['ret_p10']:+.1f}%）",
            f"      獲利目標參考 P50 約 {p['p50_price']:,.2f} "
            f"（{p['ret_p50']:+.1f}%）",
        ]
    lines += [
        "",
        "─" * 60,
        "  操作備註",
        "─" * 60,
        "  1. 「明日開盤」= 下一個交易日開盤價附近執行",
        "  2. 限價單可設在「-5% 回檔價」；若未觸及可放寬至 -10%",
        "  3. 分數高 ≠ 保證獲利；僅代表模擬情境下的相對優勢",
        "  4. 金融股、航運股周期性强，請搭配總經判斷",
        "",
    ]
    return "\n".join(lines)


def run_one(
    code: str,
    name: str,
    weight: float,
    sim_start: str,
    hist_start: str,
    n_steps: int,
    n_sims: int,
    seed: int,
) -> tuple[dict | None, str | None]:
    ticker = f"{code}.TW"
    try:
        data = run_simulation(
            ticker, TW_INDEX, sim_start, hist_start,
            n_steps, n_sims, TW_CAP, seed,
            calibrate_body=True,
            enforce_left_tail=True,
        )
        stats = print_results(data, CURRENCY)
        fit = validate_simulation_fit(data)
        metrics = compute_trade_score(data, fit, weight)
        report = build_text_report(code, name, weight, data, stats, fit, metrics)
        row = dict(code=code, name=name, ticker=ticker, **metrics)
        return row, report
    except (SystemExit, ValueError, Exception) as e:
        err = f"{code} {name}: {e}\n{traceback.format_exc()}"
        return None, err


def parse_args():
    p = argparse.ArgumentParser(description="0050 成分股 MMAR 批次掃描")
    p.add_argument("--sims", type=int, default=3000,
                   help="每檔路徑數（預設 3000，50檔約 20–40 分鐘）")
    p.add_argument("--steps", type=int, default=252)
    p.add_argument("--hist-start", default="2020-01-01")
    p.add_argument("--start", default=None, help="模擬起點日 YYYY-MM-DD")
    p.add_argument("--top", type=int, default=3)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--codes", default=None,
                   help="只跑指定代號，逗號分隔，如 2330,2454")
    p.add_argument("--skip-existing", action="store_true",
                   help="略過已有報告的標的")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    today = date.today().strftime("%Y%m%d")
    out_dir = args.out_dir or os.path.join("output", f"0050_scan_{today}")
    rep_dir = os.path.join(out_dir, "reports")
    os.makedirs(rep_dir, exist_ok=True)

    if args.start:
        sim_start = args.start
    else:
        tmp = yf.download("^TWII", period="5d", progress=False)
        sim_start = str(tmp.index[-1].date())

    # 下一個交易日（簡化：若週五則 +3，週六 +2，其餘 +1）
    d0 = pd.Timestamp(sim_start)
    trade_date = str((d0 + pd.offsets.BDay(1)).date())

    pool = CONSTITUENTS_0050
    if args.codes:
        wanted = {c.strip() for c in args.codes.split(",")}
        pool = [x for x in pool if x[0] in wanted]

    print(f"\n{'='*60}")
    print(f"  0050 成分股 MMAR 批次掃描")
    print(f"  標的數：{len(pool)}  |  路徑/檔：{args.sims:,}  |  起點：{sim_start}")
    print(f"  輸出：{os.path.abspath(out_dir)}")
    print(f"{'='*60}\n")

    rows: list[dict] = []
    errors: list[str] = []

    for i, (code, name, weight) in enumerate(pool, 1):
        fname = f"{code}_{_safe_name(name)}.txt"
        fpath = os.path.join(rep_dir, fname)

        if args.skip_existing and os.path.isfile(fpath):
            print(f"[{i}/{len(pool)}] 略過 {code} {name}（已有報告）")
            continue

        print(f"[{i}/{len(pool)}] 執行 {code} {name} ...", flush=True)
        row, text = run_one(
            code, name, weight, sim_start, args.hist_start,
            args.steps, args.sims, args.seed + i,
        )
        if row is None:
            errors.append(text or f"{code} unknown error")
            print(f"  ✗ 失敗：{text.splitlines()[0] if text else 'error'}")
            with open(os.path.join(rep_dir, f"{code}_ERROR.txt"), "w", encoding="utf-8") as f:
                f.write(text or "")
            continue

        with open(fpath, "w", encoding="utf-8") as f:
            f.write(text)
        rows.append(row)
        print(f"  ✓ 分數 {row['trade_score']:.1f}  GOF {row['gof_score']:.0f}  "
              f"P50 {row['ret_p50']:+.1f}%  → {fname}")

    if not rows:
        print("\n無成功結果，結束。")
        sys.exit(1)

    rows.sort(key=lambda r: r["trade_score"], reverse=True)
    picks = rows[: args.top]

    csv_path = os.path.join(out_dir, "summary.csv")
    fields = [
        "code", "name", "weight_pct", "trade_score", "gof_score",
        "body_score", "tail_score", "last_price",
        "ret_p10", "ret_p50", "ret_p90", "risk_adj",
        "dip5_entry", "dip5_touch", "dip5_peak_win", "dip5_peak_p20",
        "dip5_end_win", "dip5_end_p20", "dip10_touch", "dip10_peak_p20",
        "spot_end_win", "spot_end_p20",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    top3_text = build_top3_report(picks, sim_start, trade_date, out_dir)
    top3_path = os.path.join(out_dir, "TOP3_操作建議.txt")
    with open(top3_path, "w", encoding="utf-8") as f:
        f.write(top3_text)

    print(f"\n{top3_text}")
    print(f"\n完成：{len(rows)}/{len(pool)} 檔成功")
    if errors:
        print(f"失敗：{len(errors)} 檔（見 reports/*_ERROR.txt）")
    print(f"摘要：{os.path.abspath(csv_path)}")
    print(f"建議：{os.path.abspath(top3_path)}")


if __name__ == "__main__":
    main()