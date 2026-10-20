"""
Бэктест на минутных свечах.

1. Скачивает 1m свечи всех USDT-перпетуалов (с кэшем в data/).
2. Находит сигналы по тем же правилам, что и живой скринер (окно 1..15 мин).
3. Симулирует шорт: маржа = 1R, стоп = ликвидация, тейк = X%.
   Перебирает плечи × тейки × режимы входа (по сигналу / после подтверждения).
4. Считает MAE (максимальное движение против шорта) и рекомендует плечо.

Запуск:  python backtest.py --days 30
         python backtest.py --days 14 --exchanges binance --top 150
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
from itertools import product

import numpy as np
import pandas as pd

from core import liq_distance_pct, load_config, recommend_leverage, side_label

log = logging.getLogger("backtest")
DATA_DIR = "data"
OUT_DIR = "results"
DAY = 1440


# ---------------------------------------------------------------- данные

async def download(cfg, exchanges, days, top, only_symbols):
    import aiohttp
    from exchanges import KLINES, SYMBOLS, VOL24

    end_ms = (int(time.time()) // 60) * 60_000
    start_ms = end_ms - (days + 1) * DAY * 60_000      # +1 день на разгон суточного объёма
    exclude = set(cfg["detector"].get("exclude_symbols", []))
    frames = {}
    async with aiohttp.ClientSession() as session:
        for ex in exchanges:
            syms = [s for s in await SYMBOLS[ex](session) if s not in exclude]
            if only_symbols:
                syms = [s for s in syms if s in only_symbols]
            if top:
                vols = await VOL24[ex](session)
                syms = sorted(syms, key=lambda s: -vols.get(s, 0))[:top]
            os.makedirs(f"{DATA_DIR}/{ex}", exist_ok=True)
            sem = asyncio.Semaphore(4)
            done = 0

            async def one(sym):
                nonlocal done
                path = f"{DATA_DIR}/{ex}/{sym}.csv.gz"
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    if len(df) and df.t.iloc[0] <= start_ms + 60_000 and df.t.iloc[-1] >= end_ms - 3 * 3600_000:
                        frames[(ex, sym)] = df[(df.t >= start_ms) & (df.t <= end_ms)].reset_index(drop=True)
                        done += 1
                        return
                async with sem:
                    try:
                        rows = await KLINES[ex](session, sym, start_ms, end_ms)
                    except Exception as e:
                        log.warning("%s %s: %s", ex, sym, e)
                        return
                df = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "qv", "tbq"])
                df.to_csv(path, index=False, compression="gzip")
                frames[(ex, sym)] = df
                done += 1
                if done % 20 == 0:
                    log.info("%s: загружено %d/%d", ex, done, len(syms))

            await asyncio.gather(*(one(s) for s in syms))
            log.info("%s: готово %d монет", ex, done)
    return frames


def load_cached(exchanges, days):
    frames = {}
    for ex in exchanges:
        d = f"{DATA_DIR}/{ex}"
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if fn.endswith(".csv.gz"):
                df = pd.read_csv(f"{d}/{fn}")
                if days:
                    df = df[df.t >= df.t.iloc[-1] - (days + 1) * DAY * 60_000]
                frames[(ex, fn[:-7])] = df.reset_index(drop=True)
    return frames


# ---------------------------------------------------------------- детекция

def detect(df: pd.DataFrame, det: dict) -> list:
    """Те же правила, что у живого скринера, но с шагом 1 минута. Сигнал = закрытие минуты."""
    n = len(df)
    if n < DAY + 20:
        return []
    o, c, qv, tbq = (df[k].to_numpy(float) for k in ("o", "c", "qv", "tbq"))
    S = np.concatenate([[0.0], np.cumsum(qv)])
    idx = np.arange(n)
    vol24 = S[idx + 1] - S[np.maximum(0, idx + 1 - DAY)]
    maxk = max(1, det["max_window_sec"] // 60)
    best = np.zeros(n, dtype=int)
    for k in range(1, maxk + 1):
        st = idx - k + 1
        stc = np.clip(st, 0, None)
        cum = S[idx + 1] - S[stc]
        base = np.maximum(vol24 - cum, 1.0)
        pc = (c / o[stc] - 1) * 100
        ok = ((st >= DAY) & (best == 0) & (cum >= det["min_volume_usd"]) &
              (cum / base * 100 >= det["min_pct_of_day"]) & (pc >= det["min_price_change_pct"]))
        best[ok] = k
    min24 = det.get("min_vol24h_usd", 0)
    max24 = det.get("max_vol24h_usd") or float("inf")
    cooldown = det["cooldown_min"]
    out, last = [], -10 ** 9
    for t in np.flatnonzero(best):
        if t - last < cooldown or not (min24 <= vol24[t] <= max24):
            continue
        k = best[t]
        cum = S[t + 1] - S[t + 1 - k]
        buy = np.nansum(tbq[t + 1 - k:t + 1]) if not np.isnan(tbq[t]) else float("nan")
        out.append({"i": int(t), "window_min": int(k), "volume": cum, "vol24h": vol24[t],
                    "pct_of_day": cum / max(vol24[t] - cum, 1) * 100,
                    "price_change": (c[t] / o[t + 1 - k] - 1) * 100,
                    "buy_share": buy / cum if cum else float("nan"),
                    "price": c[t]})
        last = t
    return out


# ---------------------------------------------------------------- симуляция

def entry_index(h, l, c, t, mode, confirm_wait):
    if mode == "signal":
        return t, c[t]
    for j in range(t + 1, min(t + 1 + confirm_wait, len(c))):
        if c[j] < l[j - 1]:
            return j, c[j]
    return None, None


def simulate(h, l, c, e, entry, lev, tp_pct, risk, max_hold):
    """Возвращает (результат в R, исход, минут в сделке)."""
    mmr = risk["maintenance_margin_pct"] / 100
    fee = risk["taker_fee_pct"] / 100
    slip = risk["slippage_pct"] / 100
    entry = entry * (1 - slip)                       # продаём чуть хуже
    liq = entry * (1 + 1 / lev - mmr)
    tp = entry * (1 - tp_pct / 100)
    end = min(e + 1 + max_hold, len(c))
    hh, ll = h[e + 1:end], l[e + 1:end]
    if len(hh) == 0:
        return None
    hit_liq = np.flatnonzero(hh >= liq)
    hit_tp = np.flatnonzero(ll <= tp)
    i_liq = hit_liq[0] if len(hit_liq) else 10 ** 9
    i_tp = hit_tp[0] if len(hit_tp) else 10 ** 9
    fees_r = 2 * fee * lev
    if i_liq <= i_tp and i_liq < 10 ** 9:            # в одной свече — считаем худшее
        return -1.0, "liq", int(i_liq + 1)
    if i_tp < 10 ** 9:
        exit_px = tp * (1 + slip)
        return lev * (entry - exit_px) / entry - fees_r, "tp", int(i_tp + 1)
    exit_px = c[end - 1] * (1 + slip)
    return max(-1.0, lev * (entry - exit_px) / entry - fees_r), "timeout", len(hh)


def mae_mfe(h, l, e, entry, horizon):
    hh, ll = h[e + 1:e + 1 + horizon], l[e + 1:e + 1 + horizon]
    if len(hh) == 0:
        return np.nan, np.nan
    return (hh.max() / entry - 1) * 100, (1 - ll.min() / entry) * 100


def max_drawdown(r):
    eq = np.cumsum(r)
    return float((np.maximum.accumulate(np.concatenate([[0], eq]))[1:] - eq).max()) if len(r) else 0.0


# ---------------------------------------------------------------- main

def run(cfg, frames):
    det, risk, bt = cfg["detector"], cfg["risk"], cfg["backtest"]
    os.makedirs(OUT_DIR, exist_ok=True)
    signals = []
    arrays = {}
    for key, df in frames.items():
        sigs = detect(df, det)
        if not sigs:
            continue
        arrays[key] = (df.t.to_numpy(), df.h.to_numpy(float), df.l.to_numpy(float),
                       df.c.to_numpy(float))
        for s in sigs:
            s["exchange"], s["symbol"] = key
            s["time"] = pd.to_datetime(df.t.iloc[s["i"]] + 60_000, unit="ms", utc=True)
            s["label"] = side_label(s["buy_share"], det["side_threshold"])
            signals.append(s)
    if not signals:
        print("Сигналов не найдено. Увеличьте --days или ослабьте пороги в config.yaml.")
        return
    sig_df = pd.DataFrame(signals).sort_values("time").reset_index(drop=True)
    print(f"\nСигналов: {len(sig_df)} по {sig_df.symbol.nunique()} монетам "
          f"за {(sig_df.time.max() - sig_df.time.min()).days + 1} дн.")

    # MAE/MFE и подбор плеча
    mmr = risk["maintenance_margin_pct"]
    mae_rows = []
    for mode in bt["entry_modes"]:
        for s in signals:
            t_, h, l, c = arrays[(s["exchange"], s["symbol"])]
            e, px = entry_index(h, l, c, s["i"], mode, risk["confirm_wait_min"])
            if e is None:
                continue
            for hz in (60, 240):
                mae, mfe = mae_mfe(h, l, e, px, hz)
                mae_rows.append({"mode": mode, "horizon_min": hz, "mae": mae, "mfe": mfe})
    mae_df = pd.DataFrame(mae_rows)
    lines = ["", "=== Движение после входа (MAE против шорта / MFE в пользу), % ==="]
    for (mode, hz), g in mae_df.groupby(["mode", "horizon_min"]):
        q = g.mae.quantile([0.5, 0.8, 0.9]).values
        lines.append(f"{mode:7s} {hz:>3d} мин: MAE медиана {q[0]:.1f} | p80 {q[1]:.1f} | p90 {q[2]:.1f}"
                     f"   MFE медиана {g.mfe.median():.1f} | p80 {g.mfe.quantile(.8):.1f}"
                     f"   → плечо без ликвидации в 80%: {recommend_leverage(q[1], mmr)}x,"
                     f" в 90%: {recommend_leverage(q[2], mmr)}x")

    # сетка
    rows, trades_by_cfg = [], {}
    for mode, lev, tp in product(bt["entry_modes"], bt["leverages"], bt["take_profits_pct"]):
        tr = []
        for s in signals:
            t_, h, l, c = arrays[(s["exchange"], s["symbol"])]
            e, px = entry_index(h, l, c, s["i"], mode, risk["confirm_wait_min"])
            if e is None:
                continue
            res = simulate(h, l, c, e, px, lev, tp, risk, bt["max_hold_min"])
            if res is None:
                continue
            r, outcome, mins = res
            tr.append({"time": s["time"], "exchange": s["exchange"], "symbol": s["symbol"],
                       "entry": px, "r": r, "outcome": outcome, "minutes": mins})
        if not tr:
            continue
        tdf = pd.DataFrame(tr).sort_values("time")
        r = tdf.r.to_numpy()
        wins, losses = r[r > 0].sum(), -r[r < 0].sum()
        eq = np.prod(1 + r * risk["risk_pct"] / 100)
        rows.append({"mode": mode, "leverage": lev, "tp_pct": tp, "trades": len(r),
                     "winrate_%": round((r > 0).mean() * 100, 1),
                     "liq_%": round((tdf.outcome == "liq").mean() * 100, 1),
                     "avg_R": round(r.mean(), 3), "total_R": round(r.sum(), 1),
                     "profit_factor": round(wins / losses, 2) if losses else np.inf,
                     "max_dd_R": round(max_drawdown(r), 1),
                     f"deposit_x_at_{risk['risk_pct']:g}%": round(eq, 3)})
        trades_by_cfg[(mode, lev, tp)] = tdf
    grid = pd.DataFrame(rows).sort_values("total_R", ascending=False)
    grid.to_csv(f"{OUT_DIR}/grid.csv", index=False)
    sig_df.drop(columns=["i"]).to_csv(f"{OUT_DIR}/signals.csv", index=False)

    min_tr = bt.get("min_trades", 20)
    max_liq = bt.get("max_liq_pct", 100)
    ok = grid[(grid.trades >= min_tr) & (grid["liq_%"] <= max_liq)]
    if not len(ok):
        ok = grid[grid.trades >= min_tr]
    best = (ok if len(ok) else grid).iloc[0]
    bkey = (best["mode"], best["leverage"], best["tp_pct"])
    trades_by_cfg[bkey].to_csv(f"{OUT_DIR}/trades_best.csv", index=False)

    lines += ["", "=== Топ-15 настроек (стоп = ликвидация, маржа = 1R) ===",
              grid.head(15).to_string(index=False), "",
              f"Рекомендация (не больше {max_liq}% ликвидаций): вход={best['mode']}, плечо={best['leverage']}x, тейк={best['tp_pct']}% → "
              f"{best['trades']} сделок, винрейт {best['winrate_%']}%, ликвидаций {best['liq_%']}%, "
              f"средний {best['avg_R']:+.3f}R, просадка {best['max_dd_R']}R.",
              f"Ликвидация при {best['leverage']}x ≈ +{liq_distance_pct(best['leverage'], mmr):.2f}% от входа.",
              "Перенесите эти значения в config.yaml (risk.leverage, risk.take_profit_pct, risk.entry_mode).",
              "Если avg_R ≤ 0 у всех строк — преимущества у стратегии на этом периоде нет."]
    report = "\n".join(lines)
    print(report)
    with open(f"{OUT_DIR}/report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        tdf = trades_by_cfg[bkey]
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(tdf.time, tdf.r.cumsum(), color="#2a6fdb", lw=1.6)
        ax.axhline(0, color="#999", lw=0.8)
        ax.set_title(f"Кривая доходности в R: вход {bkey[0]}, {bkey[1]}x, тейк {bkey[2]}%")
        ax.set_ylabel("Сумма R")
        ax.grid(alpha=0.25)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(f"{OUT_DIR}/equity_best.png", dpi=130)
    except Exception as e:
        log.info("График не построен: %s", e)
    print(f"\nФайлы: {OUT_DIR}/report.txt, grid.csv, signals.csv, trades_best.csv, equity_best.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--exchanges", nargs="*", default=None)
    ap.add_argument("--top", type=int, default=0, help="только N монет с наибольшим объёмом")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--offline", action="store_true", help="только кэш из data/, без скачивания")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    days = args.days or cfg["backtest"]["days"]
    exchanges = args.exchanges or cfg["exchanges"]
    if args.offline:
        frames = load_cached(exchanges, days)
    else:
        frames = asyncio.run(download(cfg, exchanges, days, args.top, set(args.symbols or [])))
    run(cfg, frames)


if __name__ == "__main__":
    main()
