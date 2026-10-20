"""
Клиенты Binance USDT-M Futures и Bybit Linear (USDT perpetual):
список монет, суточные объёмы, исторические минутные свечи, поток сделок.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Awaitable, Callable, Dict, List

import aiohttp

log = logging.getLogger("exchanges")

BINANCE_REST = "https://fapi.binance.com"
BYBIT_REST = "https://api.bybit.com"

# колбэк сделки: (exchange, symbol, ts_ms, price, quote_qty_usdt, is_taker_buy)
TradeCB = Callable[[str, str, int, float, float, bool], None]


async def _get(session: aiohttp.ClientSession, url: str, params: dict = None, retries: int = 5):
    delay = 1.0
    for attempt in range(retries):
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as r:
                if r.status in (418, 429):
                    wait = float(r.headers.get("Retry-After", 10))
                    log.warning("Лимит запросов %s, жду %.0f c", url, wait)
                    await asyncio.sleep(wait)
                    continue
                if r.status == 451 or r.status == 403:
                    raise RuntimeError(
                        f"{url}: HTTP {r.status}. Биржа блокирует запросы из этого региона — "
                        f"запускайте на сервере в разрешённой стране.")
                r.raise_for_status()
                return await r.json()
        except RuntimeError:
            raise
        except Exception as ex:
            if attempt == retries - 1:
                raise
            log.warning("Ошибка %s (%s), повтор через %.0f c", url, ex, delay)
            await asyncio.sleep(delay)
            delay *= 2


# ---------------------------------------------------------------- symbols & 24h

async def binance_symbols(session) -> List[str]:
    data = await _get(session, f"{BINANCE_REST}/fapi/v1/exchangeInfo")
    return [s["symbol"] for s in data["symbols"]
            if s.get("contractType") == "PERPETUAL" and s.get("quoteAsset") == "USDT"
            and s.get("status") == "TRADING"]


async def binance_24h(session) -> Dict[str, float]:
    data = await _get(session, f"{BINANCE_REST}/fapi/v1/ticker/24hr")
    return {d["symbol"]: float(d["quoteVolume"]) for d in data}


async def bybit_symbols(session) -> List[str]:
    out, cursor = [], None
    while True:
        params = {"category": "linear", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        data = await _get(session, f"{BYBIT_REST}/v5/market/instruments-info", params)
        res = data["result"]
        for s in res["list"]:
            if (s.get("quoteCoin") == "USDT" and s.get("status") == "Trading"
                    and s.get("contractType") == "LinearPerpetual"):
                out.append(s["symbol"])
        cursor = res.get("nextPageCursor")
        if not cursor:
            return out


async def bybit_24h(session) -> Dict[str, float]:
    data = await _get(session, f"{BYBIT_REST}/v5/market/tickers", {"category": "linear"})
    return {d["symbol"]: float(d.get("turnover24h") or 0) for d in data["result"]["list"]}


SYMBOLS = {"binance": binance_symbols, "bybit": bybit_symbols}
VOL24 = {"binance": binance_24h, "bybit": bybit_24h}


# ---------------------------------------------------------------- klines (для бэктеста)

async def binance_klines(session, symbol: str, start_ms: int, end_ms: int) -> list:
    """Строки: (open_ms, o, h, l, c, quote_vol, taker_buy_quote)."""
    rows, cur = [], start_ms
    while cur < end_ms:
        data = await _get(session, f"{BINANCE_REST}/fapi/v1/klines",
                          {"symbol": symbol, "interval": "1m", "startTime": cur,
                           "endTime": end_ms, "limit": 1500})
        if not data:
            break
        for k in data:
            rows.append((int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]),
                         float(k[7]), float(k[10])))
        cur = int(data[-1][0]) + 60_000
        await asyncio.sleep(0.15)
    return rows


async def bybit_klines(session, symbol: str, start_ms: int, end_ms: int) -> list:
    """Строки: (open_ms, o, h, l, c, turnover, nan) — у Bybit нет разбивки на покупки."""
    rows, cur = [], start_ms
    step = 1000 * 60_000
    while cur < end_ms:
        data = await _get(session, f"{BYBIT_REST}/v5/market/kline",
                          {"category": "linear", "symbol": symbol, "interval": "1",
                           "start": cur, "end": min(cur + step - 1, end_ms), "limit": 1000})
        lst = data.get("result", {}).get("list", [])
        for k in reversed(lst):          # Bybit отдаёт от новых к старым
            rows.append((int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]),
                         float(k[6]), float("nan")))
        cur += step
        await asyncio.sleep(0.12)
    rows.sort(key=lambda r: r[0])
    return rows


KLINES = {"binance": binance_klines, "bybit": bybit_klines}


# ---------------------------------------------------------------- live trades

async def binance_trades(session, symbols: List[str], on_trade: TradeCB, ws_url: str,
                         chunk: int = 150):
    async def run(part: List[str]):
        backoff = 1
        while True:
            try:
                async with session.ws_connect(ws_url, heartbeat=30, max_msg_size=0) as ws:
                    await ws.send_json({"method": "SUBSCRIBE",
                                        "params": [f"{s.lower()}@aggTrade" for s in part],
                                        "id": int(time.time())})
                    log.info("Binance: подписка на %d монет", len(part))
                    backoff = 1
                    async for msg in ws:
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                break
                            continue
                        d = json.loads(msg.data)
                        d = d.get("data", d)
                        if d.get("e") != "aggTrade":
                            continue
                        p = float(d["p"])
                        # m=True: покупатель — мейкер, значит агрессор продаёт
                        on_trade("binance", d["s"], int(d["T"]), p, p * float(d["q"]), not d["m"])
            except asyncio.CancelledError:
                raise
            except Exception as ex:
                log.warning("Binance WS обрыв: %s", ex)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)

    parts = [symbols[i:i + chunk] for i in range(0, len(symbols), chunk)]
    await asyncio.gather(*(run(p) for p in parts))


async def bybit_trades(session, symbols: List[str], on_trade: TradeCB, ws_url: str,
                       chunk: int = 150):
    async def run(part: List[str]):
        backoff = 1
        while True:
            try:
                async with session.ws_connect(ws_url, max_msg_size=0) as ws:
                    args = [f"publicTrade.{s}" for s in part]
                    for i in range(0, len(args), 10):
                        await ws.send_json({"op": "subscribe", "args": args[i:i + 10]})
                    log.info("Bybit: подписка на %d монет", len(part))
                    backoff = 1

                    async def pinger():
                        while True:
                            await asyncio.sleep(20)
                            await ws.send_json({"op": "ping"})
                    ping_task = asyncio.create_task(pinger())
                    try:
                        async for msg in ws:
                            if msg.type != aiohttp.WSMsgType.TEXT:
                                if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                    break
                                continue
                            d = json.loads(msg.data)
                            if not str(d.get("topic", "")).startswith("publicTrade."):
                                continue
                            for t in d.get("data", []):
                                p = float(t["p"])
                                on_trade("bybit", t["s"], int(t["T"]), p, p * float(t["v"]),
                                         t["S"] == "Buy")
                    finally:
                        ping_task.cancel()
            except asyncio.CancelledError:
                raise
            except Exception as ex:
                log.warning("Bybit WS обрыв: %s", ex)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)

    parts = [symbols[i:i + chunk] for i in range(0, len(symbols), chunk)]
    await asyncio.gather(*(run(p) for p in parts))


TRADES = {"binance": binance_trades, "bybit": bybit_trades}
