"""
Живой скринер: слушает все сделки по USDT-перпетуалам Binance и Bybit,
ищет всплески объёма, шлёт сигнал в Telegram с планом шорта.

Запуск:   python scanner.py
Проверка Telegram:   python scanner.py --test-telegram
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import time
from collections import deque
from datetime import datetime
from zoneinfo import ZoneInfo

import aiohttp

from core import (find_burst, format_signal, fmt_price, load_config, short_plan,
                  side_label, tg_send)
from exchanges import SYMBOLS, TRADES, VOL24

log = logging.getLogger("scanner")
STATE_FILE = "state.json"
LOG_FILE = "signals_live.csv"


class SymState:
    __slots__ = ("buckets", "total")

    def __init__(self):
        self.buckets = deque()      # [sec, vol, buy_vol, first_px, last_px]
        self.total = 0.0


class Pending:
    """Ожидание подтверждения входа: 1-мин свеча закрылась ниже минимума предыдущей."""
    __slots__ = ("exchange", "symbol", "burst", "label", "prev_ago", "deadline",
                 "minute", "o", "h", "l", "c", "prev_low")

    def __init__(self, exchange, symbol, burst, label, prev_ago, deadline):
        self.exchange, self.symbol, self.burst = exchange, symbol, burst
        self.label, self.prev_ago, self.deadline = label, prev_ago, deadline
        self.minute = None
        self.o = self.h = self.l = self.c = None
        self.prev_low = None


class Scanner:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.det = cfg["detector"]
        self.risk = cfg["risk"]
        self.tg = cfg["telegram"]
        self.states: dict = {}
        self.vol24: dict = {}
        self.last_signal: dict = {}
        self.pending: dict = {}
        self.queue: asyncio.Queue = asyncio.Queue()
        self.tz = ZoneInfo(cfg.get("timezone", "UTC"))
        self._load_state()

    # ---------------------------------------------------------- state
    def _load_state(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE) as f:
                    self.last_signal = {tuple(k.split(":", 1)): v for k, v in json.load(f).items()}
            except Exception:
                self.last_signal = {}

    def _save_state(self):
        with open(STATE_FILE, "w") as f:
            json.dump({f"{k[0]}:{k[1]}": v for k, v in self.last_signal.items()}, f)

    # ---------------------------------------------------------- trades
    def on_trade(self, ex, sym, ts_ms, px, qv, is_buy):
        key = (ex, sym)
        st = self.states.get(key)
        if st is None:
            st = self.states[key] = SymState()
        sec = ts_ms // 1000
        b = st.buckets
        if b and b[-1][0] >= sec:
            last = b[-1]
            last[1] += qv
            if is_buy:
                last[2] += qv
            last[4] = px
        else:
            b.append([sec, qv, qv if is_buy else 0.0, px, px])
        st.total += qv

        p = self.pending.get(key)
        if p is not None:
            self._update_pending(key, p, ts_ms, px)

    def _update_pending(self, key, p: Pending, ts_ms, px):
        minute = ts_ms // 60000
        if p.minute is None:
            p.minute, p.o, p.h, p.l, p.c = minute, px, px, px, px
            return
        if minute > p.minute:
            closed_c, closed_l = p.c, p.l
            if p.prev_low is not None and closed_c < p.prev_low:
                self.queue.put_nowait(("confirm", key, p, closed_c))
                del self.pending[key]
                return
            p.prev_low = closed_l
            p.minute, p.o, p.h, p.l, p.c = minute, px, px, px, px
        else:
            p.h, p.l, p.c = max(p.h, px), min(p.l, px), px

    # ---------------------------------------------------------- detection loop
    async def checker(self):
        max_win = self.det["max_window_sec"]
        min_usd = self.det["min_volume_usd"]
        cooldown = self.det["cooldown_min"] * 60
        min_vol24 = self.det.get("min_vol24h_usd", 0)
        max_vol24 = self.det.get("max_vol24h_usd") or float("inf")
        while True:
            await asyncio.sleep(1)
            now = time.time()
            for key, st in list(self.states.items()):
                b = st.buckets
                while b and b[0][0] < now - max_win:
                    st.total -= b.popleft()[1]
                if st.total < min_usd:
                    continue
                v24 = self.vol24.get(key)
                if not v24 or v24 < min_vol24 or v24 > max_vol24:
                    continue
                last = self.last_signal.get(key)
                if last and now - last < cooldown:
                    continue
                burst = find_burst(reversed(b), v24, now, self.det)
                if burst:
                    prev_ago = now - last if last else None
                    self.last_signal[key] = now
                    self._save_state()
                    self.queue.put_nowait(("signal", key, burst, prev_ago))
            # просроченные ожидания подтверждения
            for key, p in list(self.pending.items()):
                if now > p.deadline:
                    del self.pending[key]

    # ---------------------------------------------------------- sender
    async def sender(self, session):
        mode = self.risk.get("entry_mode", "both")
        deposit, risk_pct = self.risk["deposit_usdt"], self.risk["risk_pct"]
        while True:
            item = await self.queue.get()
            try:
                if item[0] == "signal":
                    _, (ex, sym), burst, prev_ago = item
                    label = side_label(burst.buy_share, self.det["side_threshold"])
                    plan = short_plan(burst.price_now, self.risk, deposit)
                    note = ("Вход по цене сигнала." if mode == "signal" else
                            "Не входить сразу: жду 1-мин свечу ниже минимума предыдущей — "
                            "пришлю подтверждение.")
                    text = format_signal(ex, sym, burst, label, prev_ago, plan, deposit,
                                         risk_pct, note)
                    if mode in ("confirm", "both"):
                        self.pending[(ex, sym)] = Pending(
                            ex, sym, burst, label, prev_ago,
                            time.time() + self.risk["confirm_wait_min"] * 60)
                    if mode in ("signal", "both") or self.tg.get("send_raw_signals", True):
                        await tg_send(session, self.tg["bot_token"], self.tg["chat_id"], text)
                    self._log_csv("signal", ex, sym, burst.price_now, burst, label)
                    log.info("Сигнал %s %s %.0f USDT за %d c", ex, sym, burst.volume,
                             burst.window_sec)
                elif item[0] == "confirm":
                    _, (ex, sym), p, price = item
                    plan = short_plan(price, self.risk, deposit)
                    coin = sym[:-4]
                    text = "\n".join([
                        f"✅ <b>#{coin}</b> вход подтверждён ({ex}) — свеча закрылась ниже минимума",
                        f"Цена входа: {fmt_price(price)} "
                        f"({(price / p.burst.price_now - 1) * 100:+.2f}% от сигнала)",
                        f"Плечо {plan['leverage']:g}x, маржа {plan['margin']:.2f} USDT, "
                        f"позиция {plan['notional']:.2f} USDT",
                        f"Стоп = ликвидация ≈ {fmt_price(plan['liq'])} "
                        f"(+{plan['liq_dist_pct']:.2f}%) → −1R",
                        f"Тейк: {fmt_price(plan['tp'])} → +{plan['win_r']:.2f}R",
                    ])
                    await tg_send(session, self.tg["bot_token"], self.tg["chat_id"], text)
                    self._log_csv("confirm", ex, sym, price, p.burst, p.label)
            except Exception as ex_:
                log.exception("Ошибка отправки: %s", ex_)

    def _log_csv(self, kind, ex, sym, price, burst, label):
        new = not os.path.exists(LOG_FILE)
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "kind", "exchange", "symbol", "price", "volume_usd",
                            "window_sec", "pct_of_day", "price_change_pct", "label"])
            w.writerow([datetime.now(self.tz).strftime("%Y-%m-%d %H:%M:%S"), kind, ex, sym,
                        price, round(burst.volume), burst.window_sec,
                        round(burst.pct_of_day, 1), round(burst.price_change, 2), label])

    # ---------------------------------------------------------- 24h volumes
    async def vol24_loop(self, session, exchanges):
        while True:
            for ex in exchanges:
                try:
                    vols = await VOL24[ex](session)
                    for s, v in vols.items():
                        self.vol24[(ex, s)] = v
                except Exception as e:
                    log.warning("Не удалось обновить объёмы %s: %s", ex, e)
            await asyncio.sleep(300)

    # ---------------------------------------------------------- main
    async def run(self):
        exchanges = self.cfg["exchanges"]
        exclude = set(self.det.get("exclude_symbols", []))
        async with aiohttp.ClientSession() as session:
            tasks = [asyncio.create_task(self.vol24_loop(session, exchanges)),
                     asyncio.create_task(self.checker()),
                     asyncio.create_task(self.sender(session))]
            total = 0
            for ex in exchanges:
                syms = [s for s in await SYMBOLS[ex](session) if s not in exclude]
                total += len(syms)
                url = self.cfg["ws"][ex]
                tasks.append(asyncio.create_task(TRADES[ex](session, syms, self.on_trade, url)))
                log.info("%s: %d монет", ex, len(syms))
            await tg_send(session, self.tg["bot_token"], self.tg["chat_id"],
                          f"🟢 Скринер запущен: {', '.join(exchanges)}, {total} монет. "
                          f"Порог: ≥{self.det['min_volume_usd']:,.0f} USDT и "
                          f"≥{self.det['min_pct_of_day']}% суточного за ≤"
                          f"{self.det['max_window_sec'] // 60} мин, рост ≥"
                          f"{self.det['min_price_change_pct']}%.")
            await asyncio.gather(*tasks)


async def test_telegram(cfg):
    async with aiohttp.ClientSession() as s:
        ok = await tg_send(s, cfg["telegram"]["bot_token"], cfg["telegram"]["chat_id"],
                           "✅ Связка с Telegram работает.")
        print("Отправлено" if ok else "Не отправлено — проверьте bot_token и chat_id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--test-telegram", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    if args.test_telegram:
        asyncio.run(test_telegram(cfg))
        return
    try:
        asyncio.run(Scanner(cfg).run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
