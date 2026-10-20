"""
Общая логика: поиск всплеска объёма, план шорта (стоп = ликвидация),
форматирование сообщения и отправка в Telegram.
Используется и живым скринером (scanner.py), и бэктестом (backtest.py).
"""
from __future__ import annotations

import asyncio
import html
import logging
import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import yaml

log = logging.getLogger("core")


# ---------------------------------------------------------------- config

ENV_MAP = {   # переменная окружения → (раздел, ключ, тип)
    "TELEGRAM_BOT_TOKEN": ("telegram", "bot_token", str),
    "TELEGRAM_CHAT_ID": ("telegram", "chat_id", str),
    "DEPOSIT_USDT": ("risk", "deposit_usdt", float),
    "RISK_PCT": ("risk", "risk_pct", float),
    "LEVERAGE": ("risk", "leverage", float),
    "TAKE_PROFIT_PCT": ("risk", "take_profit_pct", float),
    "ENTRY_MODE": ("risk", "entry_mode", str),
    "MAINTENANCE_MARGIN_PCT": ("risk", "maintenance_margin_pct", float),
    "MIN_VOLUME_USD": ("detector", "min_volume_usd", float),
    "MIN_PCT_OF_DAY": ("detector", "min_pct_of_day", float),
    "MIN_PRICE_CHANGE_PCT": ("detector", "min_price_change_pct", float),
    "COOLDOWN_MIN": ("detector", "cooldown_min", int),
}


def load_config(path: str = "config.yaml") -> dict:
    """config.yaml (или config.example.yaml), затем переопределения из переменных окружения."""
    import os
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.example.yaml")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for env, (sec, key, typ) in ENV_MAP.items():
        val = os.getenv(env)
        if val not in (None, ""):
            cfg.setdefault(sec, {})[key] = typ(val)
    if os.getenv("EXCHANGES"):
        cfg["exchanges"] = [x.strip() for x in os.getenv("EXCHANGES").split(",") if x.strip()]
    if os.getenv("TIMEZONE"):
        cfg["timezone"] = os.getenv("TIMEZONE")
    return cfg


# ---------------------------------------------------------------- детектор

@dataclass
class Burst:
    window_sec: int          # длина окна всплеска
    volume: float            # объём в окне, USDT
    buy_volume: float        # из него рыночные покупки (taker buy), USDT; NaN если неизвестно
    vol24h: float            # суточный объём (как у биржи, включая всплеск)
    price_start: float       # цена в начале окна
    price_now: float         # текущая цена

    @property
    def pct_of_day(self) -> float:
        """Доля всплеска от суточного объёма БЕЗ самого всплеска (как считает исходный бот)."""
        base = max(self.vol24h - self.volume, 1.0)
        return self.volume / base * 100

    @property
    def price_change(self) -> float:
        return (self.price_now / self.price_start - 1) * 100

    @property
    def buy_share(self) -> float:
        if self.volume <= 0 or self.buy_volume != self.buy_volume:  # NaN
            return float("nan")
        return self.buy_volume / self.volume


def find_burst(buckets_newest_first: Iterable[Sequence[float]], vol24h: float,
               now_sec: float, det: dict, unit_sec: int = 1) -> Optional[Burst]:
    """
    buckets_newest_first: корзины от новой к старой, каждая = (start_sec, vol, buy_vol, first_px, last_px).
    Возвращает самое короткое окно (до max_window_sec), где одновременно:
      объём >= min_volume_usd, объём >= min_pct_of_day % от (суточный - всплеск),
      и цена выросла >= min_price_change %.
    """
    min_usd = det["min_volume_usd"]
    min_pct = det["min_pct_of_day"]
    min_pc = det["min_price_change_pct"]
    max_win = det["max_window_sec"]

    cum = 0.0
    cum_buy = 0.0
    price_now = None
    for b in buckets_newest_first:
        start, vol, buy, first_px, last_px = b
        window = int(now_sec - start + unit_sec)
        if window > max_win:
            break
        if price_now is None:
            price_now = last_px
        cum += vol
        cum_buy += buy
        if cum < min_usd:
            continue
        base = max(vol24h - cum, 1.0)
        if cum / base * 100 < min_pct:
            continue
        pc = (price_now / first_px - 1) * 100
        if pc >= min_pc:
            return Burst(window, cum, cum_buy, vol24h, first_px, price_now)
    return None


def side_label(buy_share: float, threshold: float) -> str:
    if buy_share != buy_share:          # NaN, сторона неизвестна
        return "активность"
    if buy_share >= threshold:
        return "покупают"
    if buy_share <= 1 - threshold:
        return "продают"
    return "активность"


# ---------------------------------------------------------------- риск

def short_plan(entry: float, risk: dict, deposit: float, leverage: float = None,
               tp_pct: float = None) -> dict:
    """
    План шорта на изолированной марже. Маржа = риск (по умолчанию 1% депозита),
    стоп = ликвидация, значит максимальный убыток = маржа = 1R.
    """
    lev = leverage or risk["leverage"]
    tp_pct = tp_pct if tp_pct is not None else risk["take_profit_pct"]
    mmr = risk["maintenance_margin_pct"] / 100
    fee = risk["taker_fee_pct"] / 100

    margin = deposit * risk["risk_pct"] / 100
    notional = margin * lev
    liq = entry * (1 + 1 / lev - mmr)
    tp = entry * (1 - tp_pct / 100)
    fees = notional * fee * 2
    win_usd = notional * tp_pct / 100 - fees
    return {
        "leverage": lev,
        "margin": margin,
        "notional": notional,
        "qty": notional / entry,
        "liq": liq,
        "liq_dist_pct": (liq / entry - 1) * 100,
        "tp": tp,
        "tp_pct": tp_pct,
        "win_usd": win_usd,
        "win_r": win_usd / margin if margin else 0,
        "loss_usd": margin,
    }


def liq_distance_pct(leverage: float, mmr_pct: float) -> float:
    return (1 / leverage - mmr_pct / 100) * 100


def recommend_leverage(mae_pct: float, mmr_pct: float, cap: int = 50) -> int:
    """Максимальное целое плечо, при котором ликвидация дальше заданного движения против нас."""
    for lev in range(cap, 0, -1):
        if liq_distance_pct(lev, mmr_pct) > mae_pct:
            return lev
    return 1


# ---------------------------------------------------------------- форматирование

def fmt_usd(x: float) -> str:
    if x >= 1e9:
        return f"{x / 1e9:.2f}B"
    if x >= 1e6:
        return f"{x / 1e6:.2f}M"
    if x >= 1e3:
        return f"{x / 1e3:.0f}K"
    return f"{x:.0f}"


def fmt_price(p: float) -> str:
    if p <= 0 or p != p:
        return str(p)
    digits = max(2, 5 - int(math.floor(math.log10(p))))
    s = f"{p:.{min(digits, 10)}f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def fmt_window(sec: int) -> str:
    if sec < 60:
        return f"{sec} сек"
    return f"{round(sec / 60)} мин"


def fmt_ago(sec: Optional[float]) -> str:
    if sec is None:
        return "впервые с запуска"
    if sec < 3600:
        return f"{int(sec // 60)} мин назад"
    if sec < 86400:
        return f"{int(sec // 3600)} ч назад"
    return f"{int(sec // 86400)} д назад"


EX_NAMES = {"binance": "Binance Futures", "bybit": "Bybit Futures"}


def trade_link(exchange: str, symbol: str) -> str:
    if exchange == "binance":
        return f"https://www.binance.com/futures/{symbol}"
    return f"https://www.bybit.com/trade/usdt/{symbol}"


def format_signal(exchange: str, symbol: str, burst: Burst, label: str,
                  prev_ago: Optional[float], plan: dict, deposit: float,
                  risk_pct: float, note: str = "") -> str:
    coin = symbol[:-4] if symbol.endswith("USDT") else symbol
    e = html.escape
    lines = [
        f"🎰 <b>#{e(coin)}</b> {label} на {fmt_usd(burst.volume)} USDT за "
        f"{fmt_window(burst.window_sec)} ({int(burst.pct_of_day)}%) — "
        f"<a href=\"{trade_link(exchange, symbol)}\">{EX_NAMES.get(exchange, exchange)}</a>",
        f"P: {fmt_price(burst.price_now)} ⬆️ (+{burst.price_change:.2f}%)",
        f"Объём 24ч: {fmt_usd(burst.vol24h)} USDT",
        f"Предыдущее: {fmt_ago(prev_ago)}",
        "",
        f"📉 <b>План шорта</b> (изолированная маржа)",
        f"Плечо: <b>{plan['leverage']:g}x</b>",
        f"Маржа: <b>{plan['margin']:.2f} USDT</b> ({risk_pct:g}% от {deposit:g})",
        f"Позиция: {plan['notional']:.2f} USDT ≈ {plan['qty']:.6g} {e(coin)}",
        f"Стоп = ликвидация ≈ {fmt_price(plan['liq'])} (+{plan['liq_dist_pct']:.2f}%) → −1R",
        f"Тейк: {fmt_price(plan['tp'])} (−{plan['tp_pct']:g}%) → "
        f"+{plan['win_usd']:.2f} USDT (+{plan['win_r']:.2f}R)",
    ]
    if note:
        lines += ["", f"⚠️ {e(note)}"]
    return "\n".join(lines)


# ---------------------------------------------------------------- telegram

async def tg_send(session, token: str, chat_id, text: str) -> bool:
    if not token or not chat_id:
        log.info("Telegram не настроен, сообщение:\n%s", text)
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML",
               "disable_web_page_preview": True}
    for _ in range(3):
        try:
            async with session.post(url, json=payload, timeout=15) as r:
                if r.status == 200:
                    return True
                data = await r.json(content_type=None)
                if r.status == 429:
                    await asyncio.sleep(data.get("parameters", {}).get("retry_after", 5))
                    continue
                log.warning("Telegram %s: %s", r.status, data)
                return False
        except Exception as ex:  # сеть
            log.warning("Telegram ошибка: %s", ex)
            await asyncio.sleep(2)
    return False


async def tg_send_file(session, token: str, chat_id, path: str, caption: str = "") -> bool:
    """Отправить файл: картинки как фото, остальное как документ."""
    import aiohttp
    import os
    if not token or not chat_id or not os.path.exists(path):
        return False
    is_photo = path.lower().endswith((".png", ".jpg", ".jpeg"))
    method, field = ("sendPhoto", "photo") if is_photo else ("sendDocument", "document")
    form = aiohttp.FormData()
    form.add_field("chat_id", str(chat_id))
    if caption:
        form.add_field("caption", caption[:1000])
    with open(path, "rb") as f:
        form.add_field(field, f.read(), filename=os.path.basename(path))
    try:
        async with session.post(f"https://api.telegram.org/bot{token}/{method}",
                                data=form, timeout=60) as r:
            if r.status != 200:
                log.warning("Telegram файл %s: %s", r.status, await r.text())
            return r.status == 200
    except Exception as ex:
        log.warning("Telegram файл ошибка: %s", ex)
        return False
