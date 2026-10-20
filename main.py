"""
Точка входа для Railway (и любого облака без командной строки).
Режим задаётся переменной окружения MODE:
  MODE=scanner   — живой скринер (по умолчанию)
  MODE=backtest  — бэктест, отчёт приходит в Telegram, потом процесс ждёт
Параметры бэктеста: BT_DAYS (14), BT_TOP (100 монет на биржу), BT_EXCHANGES (binance,bybit)
"""
import asyncio
import logging
import os
import time

from core import load_config, tg_send, tg_send_file

log = logging.getLogger("main")


async def notify(cfg, text=None, files=()):
    import aiohttp
    tok, chat = cfg["telegram"]["bot_token"], cfg["telegram"]["chat_id"]
    async with aiohttp.ClientSession() as s:
        if text:
            await tg_send(s, tok, chat, text)
        for path, caption in files:
            await tg_send_file(s, tok, chat, path, caption)


def run_backtest(cfg):
    import backtest
    days = int(os.getenv("BT_DAYS", "14"))
    top = int(os.getenv("BT_TOP", "100"))
    exs = [x.strip() for x in os.getenv("BT_EXCHANGES", ",".join(cfg["exchanges"])).split(",") if x.strip()]
    asyncio.run(notify(cfg, f"⏳ Бэктест запущен: {days} дн., {', '.join(exs)}, топ-{top} монет. "
                            f"Загрузка истории займёт 10–40 минут."))
    try:
        frames = asyncio.run(backtest.download(cfg, exs, days, top, set()))
        backtest.run(cfg, frames)
    except Exception as e:
        log.exception("Бэктест упал")
        asyncio.run(notify(cfg, f"❌ Бэктест упал: {str(e)[:500]}"))
        return
    out = backtest.OUT_DIR
    rep = os.path.join(out, "report.txt")
    if not os.path.exists(rep):
        asyncio.run(notify(cfg, "Бэктест: сигналов не найдено. Увеличьте BT_DAYS или BT_TOP."))
        return
    rec = next((l for l in open(rep, encoding="utf-8") if l.startswith("Рекомендация")), "")
    asyncio.run(notify(cfg, f"✅ Бэктест готов.\n{rec}", [
        (rep, "Полный отчёт"),
        (os.path.join(out, "equity_best.png"), "Кривая доходности (в R)"),
        (os.path.join(out, "grid.csv"), "Все комбинации настроек"),
        (os.path.join(out, "trades_best.csv"), "Сделки по рекомендованной настройке"),
    ]))


def start_keepalive():
    """Мини-сайт «я жив» для облаков, которые требуют порт (Render, Koyeb).
    Пингуйте его каждые 5 минут через UptimeRobot, чтобы сервис не засыпал."""
    port = os.getenv("PORT")
    if not port:
        return
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(f"ok {os.getenv('MODE', 'scanner')}".encode())

        do_HEAD = do_GET

        def log_message(self, *a):
            pass

    srv = HTTPServer(("0.0.0.0", int(port)), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    log.info("Keep-alive на порту %s", port)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    start_keepalive()
    cfg = load_config()
    mode = os.getenv("MODE", "scanner").lower()
    if mode == "backtest":
        run_backtest(cfg)
        log.info("Бэктест завершён. Поставьте MODE=scanner, чтобы запустить скринер.")
        while True:                      # не даём облаку перезапускать бэктест по кругу
            time.sleep(3600)
    else:
        from scanner import Scanner
        asyncio.run(Scanner(cfg).run())


if __name__ == "__main__":
    main()
