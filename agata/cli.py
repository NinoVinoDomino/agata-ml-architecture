import argparse
import asyncio
import json
import logging
import re
import time
from collections import Counter
from datetime import date, timedelta

import httpx

from agata.config import Settings
from agata.domain import local_date
from agata.generation import ForecastWorkflow
from agata.jobs import publish_day, recover_day, target_date
from agata.model_rpc import open_generator, serve_template_model
from agata.storage import build_store
from agata.telegram import TelegramClient, TelegramError, Update, handle_update, load_credentials

ALLOWED_UPDATES = ["message", "callback_query"]
logger = logging.getLogger(__name__)


async def poll(settings: Settings, telegram: TelegramClient) -> None:
    webhook = await telegram.call("getWebhookInfo")
    if webhook.get("url"):
        raise SystemExit(
            "A webhook is active. Use a separate test bot or remove its webhook first."
        )
    store = build_store(settings)
    offset = 0
    while True:
        try:
            updates = await telegram.call(
                "getUpdates",
                offset=offset,
                timeout=25,
                allowed_updates=ALLOWED_UPDATES,
            )
            for payload in updates:
                update = Update.model_validate(payload)
                try:
                    await handle_update(update, telegram, store, settings.timezone)
                except TelegramError as exc:
                    if exc.retryable or exc.code in {401, 409}:
                        raise
                    logger.warning("%s", exc)
                offset = update.update_id + 1
        except TelegramError as exc:
            if not exc.retryable:
                raise
            logger.warning("%s", exc)
            await asyncio.sleep(max(2, exc.retry_after))


async def run() -> None:
    parser = argparse.ArgumentParser(prog="python -m agata")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="generate and publish a complete edition")
    target = generate.add_mutually_exclusive_group()
    target.add_argument("--date", type=date.fromisoformat)
    target.add_argument("--tomorrow", action="store_true")
    target.add_argument(
        "--scheduled-time", help="prepare the next day from the scheduler timestamp"
    )
    recover = commands.add_parser("recover", help="replace fallback forecasts with model output")
    recover.add_argument("--date", type=date.fromisoformat, required=True)
    commands.add_parser("poll", help="run Telegram long polling")
    webhook = commands.add_parser("set-webhook", help="register the HTTPS Telegram webhook")
    webhook.add_argument("url")
    demo = commands.add_parser("serve-template-model", help="run a local gRPC template model")
    demo.add_argument("--port", type=int, choices=range(1, 65536), metavar="PORT", default=50051)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)

    if args.command == "serve-template-model":
        await serve_template_model(args.port)
        return

    settings = Settings()
    if args.command in {"generate", "recover"}:
        scheduled_time = getattr(args, "scheduled_time", None)
        day = args.date or local_date(settings.timezone)
        if getattr(args, "tomorrow", False):
            day += timedelta(days=1)
        if scheduled_time:
            try:
                day = target_date(settings.timezone, scheduled_time)
            except ValueError as exc:
                parser.error(str(exc))
        if args.command == "recover" and settings.generator == "template":
            parser.error("recover requires a model generator")
        store = build_store(settings)
        async with open_generator(settings) as generator:
            workflow = ForecastWorkflow(generator)
            started = time.perf_counter()
            if args.command == "recover":
                try:
                    edition, status = await recover_day(day, settings, store, workflow)
                except ValueError as exc:
                    parser.error(str(exc))
            else:
                edition, published = await publish_day(day, settings, store, workflow)
                status = "published" if published else "already_published"
        reasons = Counter(
            item.validation_issue or "unknown"
            for item in edition.forecasts
            if item.source == "fallback"
        )
        print(
            json.dumps(
                {
                    "status": status,
                    "health": "degraded" if reasons else "healthy",
                    "date": str(edition.date),
                    "scheduled_time": scheduled_time,
                    "release_id": edition.release_id,
                    "generator_id": edition.generator_id,
                    "forecasts": len(edition.forecasts),
                    "fallbacks": reasons.total(),
                    "fallback_reasons": reasons,
                    "duration_seconds": round(time.perf_counter() - started, 3),
                }
            ),
            flush=True,
        )
        if status == "conflict":
            raise SystemExit(1)
        return

    credentials = await asyncio.to_thread(load_credentials, settings)
    if credentials is None:
        raise SystemExit("Configure AGATA_TELEGRAM_BOT_TOKEN or AGATA_TELEGRAM_SECRET_ARN.")
    async with httpx.AsyncClient(timeout=35 if args.command == "poll" else 8) as client:
        telegram = TelegramClient(credentials.token, client)
        try:
            if args.command == "poll":
                await poll(settings, telegram)
                return
            url = httpx.URL(args.url)
            if url.scheme != "https" or not url.host or url.path != "/v1/telegram/webhook":
                parser.error("provide an HTTPS URL ending in /v1/telegram/webhook")
            secret = credentials.webhook_secret
            if secret is None or not re.fullmatch(r"[A-Za-z0-9_-]{1,256}", secret):
                parser.error("configure a Telegram webhook secret with 1–256 allowed characters")
            await telegram.call(
                "setWebhook",
                url=str(url),
                secret_token=secret,
                allowed_updates=ALLOWED_UPDATES,
            )
            print("Telegram webhook registered.")
        except TelegramError as exc:
            raise SystemExit(str(exc)) from None


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
