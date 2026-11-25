import json
from dataclasses import dataclass
from itertools import batched

import boto3
import httpx
from pydantic import BaseModel

from agata.config import Settings
from agata.domain import SIGN_NAMES, TOPIC_NAMES, Sign, Topic, local_date
from agata.storage import EditionStore


class Chat(BaseModel):
    id: int
    type: str


class Message(BaseModel):
    message_id: int
    chat: Chat
    text: str | None = None


class Callback(BaseModel):
    id: str
    message: Message | None = None
    data: str | None = None


class Update(BaseModel):
    update_id: int
    message: Message | None = None
    callback_query: Callback | None = None


@dataclass(frozen=True, repr=False)
class TelegramCredentials:
    token: str
    webhook_secret: str | None


def load_credentials(settings: Settings) -> TelegramCredentials | None:
    if settings.telegram_secret_arn:
        response = boto3.client("secretsmanager").get_secret_value(
            SecretId=settings.telegram_secret_arn
        )
        values = json.loads(response["SecretString"])
        return TelegramCredentials(values["bot_token"], values["webhook_secret"])
    if settings.telegram_bot_token:
        secret = settings.telegram_webhook_secret
        return TelegramCredentials(
            settings.telegram_bot_token.get_secret_value(),
            secret.get_secret_value() if secret else None,
        )
    return None


class TelegramError(RuntimeError):
    def __init__(self, method: str, code: int = 0, retry_after: int = 0):
        super().__init__(f"Telegram {method} failed ({code or 'transport'})")
        self.code = code
        self.retry_after = retry_after

    @property
    def retryable(self) -> bool:
        return self.code == 0 or self.code == 429 or self.code >= 500


class TelegramClient:
    def __init__(self, token: str, client: httpx.AsyncClient):
        self._base = f"https://api.telegram.org/bot{token}"
        self._client = client

    async def call(self, method: str, **payload):
        try:
            response = await self._client.post(f"{self._base}/{method}", json=payload)
        except httpx.HTTPError:
            raise TelegramError(method) from None
        try:
            body = response.json()
        except ValueError:
            raise TelegramError(method, response.status_code) from None
        if response.is_success and body.get("ok"):
            return body.get("result")
        description = body.get("description", "").lower()
        if method == "editMessageText" and "message is not modified" in description:
            return None
        if method == "answerCallbackQuery" and (
            "query is too old" in description or "query id is invalid" in description
        ):
            return None
        raise TelegramError(
            method,
            body.get("error_code", response.status_code),
            body.get("parameters", {}).get("retry_after", 0),
        )


def keyboard(buttons, columns: int = 2) -> dict:
    return {
        "inline_keyboard": [
            [{"text": label, "callback_data": value} for label, value in row]
            for row in batched(buttons, columns)
        ]
    }


def sign_menu() -> dict:
    return keyboard(
        ((label, f"sign:{sign}") for sign, label in SIGN_NAMES.items()),
        columns=3,
    )


def topic_menu(sign: Sign) -> dict:
    return keyboard(
        [
            *((label, f"topic:{sign}:{topic}") for topic, label in TOPIC_NAMES.items()),
            ("← Знаки", "menu"),
        ]
    )


async def callback_view(data: str, store: EditionStore, timezone: str) -> tuple[str, dict] | None:
    if data == "menu":
        return "Выберите знак зодиака.", sign_menu()
    try:
        match data.split(":"):
            case ["sign", raw_sign]:
                sign = Sign(raw_sign)
                return f"{SIGN_NAMES[sign]} · выберите тему.", topic_menu(sign)
            case ["topic", raw_sign, raw_topic]:
                sign, topic = Sign(raw_sign), Topic(raw_topic)
            case _:
                return None
    except ValueError:
        return None
    day = local_date(timezone)
    edition = await store.read(day)
    if edition is None:
        return "Гороскоп на сегодня пока недоступен. Попробуйте позже.", topic_menu(sign)
    forecast = edition.get(sign, topic)
    return (
        (
            f"{SIGN_NAMES[sign]} · {TOPIC_NAMES[topic]} · {day:%d.%m.%Y}\n\n"
            f"{forecast.text}\n\nРазвлекательный гороскоп"
        ),
        topic_menu(sign),
    )


async def handle_update(
    update: Update,
    telegram: TelegramClient,
    store: EditionStore,
    timezone: str,
) -> None:
    if callback := update.callback_query:
        await telegram.call("answerCallbackQuery", callback_query_id=callback.id)
        message = callback.message
        if message and message.chat.type == "private":
            view = await callback_view(callback.data or "", store, timezone)
            if view:
                text, markup = view
                await telegram.call(
                    "editMessageText",
                    chat_id=message.chat.id,
                    message_id=message.message_id,
                    text=text,
                    reply_markup=markup,
                )
        return
    if message := update.message:
        command = (message.text or "").split(maxsplit=1)
        if (
            message.chat.type == "private"
            and command
            and command[0].split("@")[0]
            in {
                "/start",
                "/horoscope",
            }
        ):
            await telegram.call(
                "sendMessage",
                chat_id=message.chat.id,
                text="Агата на связи. Выберите знак зодиака.",
                reply_markup=sign_menu(),
            )
