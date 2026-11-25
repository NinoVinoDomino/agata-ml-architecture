import asyncio
import logging
from contextlib import asynccontextmanager
from functools import lru_cache
from hmac import compare_digest
from typing import Annotated

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

from agata.config import Settings
from agata.domain import PublicForecast, Sign, Topic, local_date
from agata.storage import EditionStore, build_store
from agata.telegram import TelegramClient, TelegramError, Update, handle_update, load_credentials

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def create_app(
    settings: Settings | None = None,
    store: EditionStore | None = None,
    telegram_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    settings = settings or Settings()
    store = store or build_store(settings)

    @lru_cache(maxsize=1)
    def telegram_credentials():
        return load_credentials(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with httpx.AsyncClient(timeout=4, transport=telegram_transport) as client:
            app.state.telegram_http = client
            yield

    app = FastAPI(title="Agata Horoscope API", version="1.0.0", lifespan=lifespan)

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def ready():
        day = local_date(settings.timezone)
        if await store.read(day) is None:
            raise HTTPException(503, "today's edition is unavailable")
        return {"status": "ready", "date": str(day)}

    @app.get("/v1/horoscopes/today", response_model=PublicForecast)
    async def today(sign: Sign, topic: Topic):
        day = local_date(settings.timezone)
        edition = await store.read(day)
        if edition is None:
            raise HTTPException(
                503, "today's edition is unavailable", headers={"Retry-After": "60"}
            )
        forecast = edition.get(sign, topic)
        return PublicForecast(
            date=day,
            timezone=edition.timezone,
            sign=sign,
            topic=topic,
            text=forecast.text,
            release_id=(
                forecast.recovered_with.release_id
                if forecast.recovered_with
                else edition.release_id
            ),
        )

    @app.post("/v1/telegram/webhook", include_in_schema=False)
    async def telegram_webhook(
        update: Update,
        secret: Annotated[str | None, Header(alias="X-Telegram-Bot-Api-Secret-Token")] = None,
    ):
        credentials = await asyncio.to_thread(telegram_credentials)
        if credentials is None or not credentials.webhook_secret:
            raise HTTPException(404)
        if secret is None or not compare_digest(
            secret.encode(), credentials.webhook_secret.encode()
        ):
            raise HTTPException(403)
        telegram = TelegramClient(credentials.token, app.state.telegram_http)
        try:
            await handle_update(update, telegram, store, settings.timezone)
        except TelegramError as exc:
            logger.warning("%s", exc)
            if exc.retryable:
                return JSONResponse({"ok": False}, 503)
            # A permanent rejection (e.g. user blocked bot) cannot improve on redelivery.
        return {"ok": True}

    return app


app = create_app()
