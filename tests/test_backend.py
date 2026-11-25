import asyncio
import json
from datetime import date
from io import BytesIO

import boto3
import httpx
import pytest
from botocore.response import StreamingBody
from botocore.stub import Stubber
from fastapi.testclient import TestClient

from agata.api import create_app
from agata.config import Settings
from agata.domain import COMBINATIONS, Sign, Topic, local_date
from agata.generation import (
    ForecastWorkflow,
    GeneratedText,
    OpenAICompatibleGenerator,
    TemplateGenerator,
    validate_text,
)
from agata.jobs import generate_edition, publish_day, recover_day, target_date
from agata.storage import FileEditionStore, S3EditionStore
from agata.templates import render_template

WEBHOOK = "/v1/telegram/webhook"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_release_and_rest_contract(tmp_path):
    settings = Settings(edition_dir=tmp_path)
    store = FileEditionStore(tmp_path)
    app = create_app(settings, store)
    params = {"sign": "leo", "topic": "work"}
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 503
        assert client.get("/v1/horoscopes/today", params=params).status_code == 503
        day = local_date(settings.timezone)
        graph = ForecastWorkflow(TemplateGenerator())
        edition, published = await publish_day(day, settings, store, graph)
        assert published and len(edition.forecasts) == 48
        assert all(item.source == "template" for item in edition.forecasts)
        assert await publish_day(day, settings, store, graph) == (edition, False)
        assert client.get("/readyz").status_code == 200
        assert client.get("/v1/horoscopes/today", params=params).json() == {
            "date": str(day),
            "timezone": settings.timezone,
            "release_id": "template-v1",
            "sign": "leo",
            "topic": "work",
            "text": edition.get(Sign.LEO, Topic.WORK).text,
        }
        assert (
            client.get("/v1/horoscopes/today", params={**params, "sign": "invalid"}).status_code
            == 422
        )


@pytest.mark.anyio
async def test_only_one_writer_publishes(tmp_path):
    store = FileEditionStore(tmp_path)
    day = date(2026, 1, 1)
    edition = await generate_edition(day, Settings(), ForecastWorkflow(TemplateGenerator()))
    other = edition.model_copy(update={"release_id": "rival"})
    published = await asyncio.gather(store.publish(edition), store.publish(other))
    assert sum(published) == 1
    assert await store.read(day) == (edition if published[0] else other)
    assert len(list(tmp_path.iterdir())) == 1


@pytest.mark.anyio
async def test_s3_conditional_write(monkeypatch):
    s3 = boto3.client(
        "s3", region_name="us-east-1", aws_access_key_id="test", aws_secret_access_key="test"
    )
    monkeypatch.setattr("agata.storage.boto3.client", lambda service: s3)
    store = S3EditionStore("agata-test")
    day = date(2026, 1, 1)
    edition = await generate_edition(day, Settings(), ForecastWorkflow(TemplateGenerator()))
    parameters = {
        "Bucket": "agata-test",
        "Key": "daily/2026-01-01.json",
        "Body": edition.model_dump_json().encode(),
        "ContentType": "application/json",
        "IfNoneMatch": "*",
    }
    with Stubber(s3) as stub:
        stub.add_client_error(
            "get_object",
            "NoSuchKey",
            expected_params={
                "Bucket": "agata-test",
                "Key": "daily/2026-01-01.json",
            },
        )
        stub.add_response("put_object", {}, parameters)
        stub.add_client_error("put_object", "PreconditionFailed", expected_params=parameters)
        payload = parameters["Body"]
        stub.add_response(
            "get_object",
            {"Body": StreamingBody(BytesIO(payload), len(payload)), "ETag": '"revision"'},
            {"Bucket": parameters["Bucket"], "Key": parameters["Key"]},
        )
        match_parameters = {key: value for key, value in parameters.items() if key != "IfNoneMatch"}
        match_parameters["IfMatch"] = '"revision"'
        stub.add_response("put_object", {}, match_parameters)
        stub.add_client_error("put_object", "PreconditionFailed", expected_params=match_parameters)
        assert await store.read(day) is None
        assert await store.publish(edition)
        assert not await store.publish(edition)
        stored = await store.read_version(day)
        assert stored is not None and stored.edition == edition
        assert await store.replace(edition, stored.revision)
        assert not await store.replace(edition, stored.revision)


def test_schedule_date_from_original_invocation():
    assert target_date("Europe/Moscow", "2025-12-31T20:00:00Z") == date(2026, 1, 1)


@pytest.mark.parametrize(
    ("text", "issue"),
    [
        (
            (
                "Сегодня стоит поговорить с партнёром о ваших чувствах. "
                "Выберите спокойный момент, чтобы лучше услышать друг друга."
            ),
            "другой теме",
        ),
        (
            (
                "Сегодня Скорпиону стоит закончить важную рабочую задачу. "
                "Обсудите с коллегами детали, которые давно откладывали."
            ),
            "другой знак",
        ),
        (
            "Сегодня стоит спокойно разобраться с несколькими важными рабочими задачами.",
            "2–3 предложения",
        ),
        (
            (
                "Подготовьте рабочие материалы: таблицы, записи и т. д. до встречи. "
                "Чтобы проверить вес оборудования, попросите коллег принести весы."
            ),
            None,
        ),
    ],
)
def test_model_validation(text, issue):
    result = validate_text(text, Sign.LEO, Topic.WORK)
    if issue is None:
        assert result is None
    else:
        assert result is not None and issue in result


def test_all_editorial_variants_pass_validation():
    for sign, topic in COMBINATIONS:
        for attempt in range(1, 4):
            text = render_template("2026-01-01", sign, topic, attempt)
            assert validate_text(text, sign, topic) is None


@pytest.mark.anyio
async def test_recover_published_edition_after_full_outage(tmp_path):
    store = FileEditionStore(tmp_path)
    day = date(2026, 1, 1)
    original_settings = Settings(release_id="before")
    recovery_settings = Settings(release_id="after")
    calls = 0
    availability = "outage"

    def endpoint(request):
        nonlocal calls
        calls += 1
        prompt = json.loads(request.content)["messages"][1]["content"]
        if availability == "outage" or (
            availability == "partial" and "Тема: Работа." not in prompt
        ):
            return httpx.Response(503)
        text = (
            "Начните день с небольшой паузы и выберите удобный темп. "
            "К вечеру оставьте время для того, что давно хотелось сделать."
        )
        return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(endpoint)) as client:
        workflow = ForecastWorkflow(
            OpenAICompatibleGenerator(client, "https://model.test", "model")
        )
        original, published = await publish_day(day, original_settings, store, workflow)
        assert published and calls == 96
        assert all(item.source == "fallback" for item in original.forecasts)
        assert await publish_day(day, original_settings, store, workflow) == (original, False)
        assert calls == 96

        assert await recover_day(day, recovery_settings, store, workflow) == (
            original,
            "no_improvement",
        )
        availability = "partial"
        partial, status = await recover_day(day, recovery_settings, store, workflow)
        assert status == "recovered"
        assert sum(item.source == "model" for item in partial.forecasts) == 12
        assert partial.get(Sign.LEO, Topic.WORK).recovered_with.release_id == "after"
        assert partial.get(Sign.LEO, Topic.FINANCE) == original.get(Sign.LEO, Topic.FINANCE)

        availability = "healthy"
        calls = 0
        healthy, status = await recover_day(day, recovery_settings, store, workflow)
        assert status == "recovered" and calls == 36
        assert all(item.source == "model" for item in healthy.forecasts)
        assert await store.read(day) == healthy
        assert await recover_day(day, recovery_settings, store, workflow) == (
            healthy,
            "already_healthy",
        )
        assert calls == 36


@pytest.mark.anyio
async def test_only_one_recovery_can_replace_a_revision(tmp_path):
    store = FileEditionStore(tmp_path)
    day = date(2026, 1, 1)
    edition = await generate_edition(day, Settings(), ForecastWorkflow(TemplateGenerator()))
    assert await store.publish(edition)
    stored = await store.read_version(day)
    assert stored is not None
    first = edition.model_copy(update={"release_id": "first"})
    second = edition.model_copy(update={"release_id": "second"})
    results = await asyncio.gather(
        store.replace(first, stored.revision), store.replace(second, stored.revision)
    )
    assert sum(results) == 1
    assert await store.read(day) == (first if results[0] else second)


@pytest.mark.anyio
async def test_model_retries_with_reason_and_accepts_second_candidate():
    calls = []

    def model(request):
        payload = json.loads(request.content)
        calls.append(payload)
        text = (
            "too short"
            if len(calls) == 1
            else (
                "Сегодня получится спокойно закончить важную рабочую задачу. "
                "Обсудите с коллегами детали, которые давно откладывали."
            )
        )
        return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(model)) as client:
        graph = ForecastWorkflow(OpenAICompatibleGenerator(client, "https://model.test", "model"))
        forecast = await graph.run(date(2026, 1, 1), Sign.LEO, Topic.WORK)

    assert forecast.source == "model"
    assert forecast.attempts == 2
    assert "Лев" in calls[0]["messages"][1]["content"]
    assert "Работа" in calls[0]["messages"][1]["content"]
    assert "длина вне диапазона" in calls[1]["messages"][1]["content"]


@pytest.mark.anyio
async def test_model_falls_back_after_two_failed_requests():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(502))
    ) as client:
        graph = ForecastWorkflow(OpenAICompatibleGenerator(client, "https://model.test", "model"))
        forecast = await graph.run(date(2026, 1, 1), Sign.LEO, Topic.FINANCE)
    assert forecast.source == "fallback"
    assert forecast.attempts == 2
    assert forecast.validation_issue == "модель временно недоступна"


@pytest.mark.anyio
async def test_telegram_webhook_menu_and_forecast(tmp_path):
    settings = Settings(
        edition_dir=tmp_path,
        telegram_bot_token="test-token",
        telegram_webhook_secret="test-secret",
    )
    store = FileEditionStore(tmp_path)
    await publish_day(
        local_date(settings.timezone), settings, store, ForecastWorkflow(TemplateGenerator())
    )
    calls = []

    def telegram(request):
        assert request.url.host == "api.telegram.org"
        method = request.url.path.rsplit("/", 1)[-1]
        calls.append((method, json.loads(request.content)))
        if method == "editMessageText" and len(calls) == 7:
            return httpx.Response(
                400,
                json={
                    "ok": False,
                    "error_code": 400,
                    "description": "Bad Request: message is not modified",
                },
            )
        return httpx.Response(200, json={"ok": True, "result": True})

    app = create_app(settings, store, httpx.MockTransport(telegram))
    message = {"message_id": 3, "chat": {"id": 17, "type": "private"}, "text": "/start"}
    update = {"update_id": 1, "message": message}
    headers = {"X-Telegram-Bot-Api-Secret-Token": "test-secret"}

    def callback(value):
        return {"update_id": 2, "callback_query": {"id": "c", "message": message, "data": value}}

    with TestClient(app) as client:
        assert client.post(WEBHOOK, json=update).status_code == 403
        assert client.post(WEBHOOK, json=update, headers=headers).status_code == 200
        assert client.post(WEBHOOK, json=callback("sign:leo"), headers=headers).status_code == 200
        for _ in range(2):
            response = client.post(WEBHOOK, json=callback("topic:leo:work"), headers=headers)
            assert response.status_code == 200

    assert [method for method, _ in calls] == [
        "sendMessage",
        "answerCallbackQuery",
        "editMessageText",
        "answerCallbackQuery",
        "editMessageText",
        "answerCallbackQuery",
        "editMessageText",
    ]
    assert calls[0][1]["reply_markup"]["inline_keyboard"][0][0]["callback_data"] == "sign:aries"
    assert calls[2][1]["reply_markup"]["inline_keyboard"][1][0]["callback_data"] == "topic:leo:work"
    assert "Развлекательный гороскоп" in calls[-1][1]["text"]


@pytest.mark.anyio
async def test_generation_respects_concurrency_limit():
    class CountingGenerator:
        id = "counting"
        active = 0
        maximum_active = 0

        async def generate(self, request):
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
            try:
                await asyncio.sleep(0.001)
                output = await TemplateGenerator().generate(request)
                return GeneratedText(output.text, "model")
            finally:
                self.active -= 1

    generator = CountingGenerator()
    edition = await generate_edition(
        date(2026, 1, 1),
        Settings(generation_concurrency=3),
        ForecastWorkflow(generator),
    )
    assert generator.maximum_active == 3
    assert len(edition.forecasts) == 48
    assert all(item.source == "model" for item in edition.forecasts)
