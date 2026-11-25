import asyncio
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from typing import Literal

from agata.config import Settings
from agata.domain import (
    COMBINATIONS,
    DailyEdition,
    Forecast,
    ReleaseProvenance,
    Sign,
    Topic,
    local_date,
)
from agata.generation import PROMPT_VERSION, ForecastWorkflow
from agata.storage import EditionStore


def target_date(timezone: str, scheduled_time: str) -> date:
    timestamp = datetime.fromisoformat(scheduled_time)
    if timestamp.tzinfo is None:
        raise ValueError("scheduled time must include its UTC offset")
    return local_date(timezone, timestamp) + timedelta(days=1)


async def generate_edition(
    day: date,
    settings: Settings,
    workflow: ForecastWorkflow,
) -> DailyEdition:
    return DailyEdition(
        date=day,
        timezone=settings.timezone,
        release_id=settings.release_id,
        prompt_version=PROMPT_VERSION,
        generator_id=workflow.generator.id,
        forecasts=await generate_forecasts(day, settings, workflow, COMBINATIONS),
    )


async def generate_forecasts(
    day: date,
    settings: Settings,
    workflow: ForecastWorkflow,
    combinations: Sequence[tuple[Sign, Topic]],
) -> list[Forecast]:
    slots = asyncio.Semaphore(settings.generation_concurrency)

    async def generate_one(sign: Sign, topic: Topic) -> Forecast:
        async with slots:
            return await workflow.run(day, sign, topic)

    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(generate_one(sign, topic)) for sign, topic in combinations]
    return [task.result() for task in tasks]


async def publish_day(
    day: date,
    settings: Settings,
    store: EditionStore,
    workflow: ForecastWorkflow,
) -> tuple[DailyEdition, bool]:
    existing = await store.read(day)
    if existing is not None:
        return existing, False
    edition = await generate_edition(day, settings, workflow)
    if not await store.publish(edition):
        existing = await store.read(day)
        if existing is None:
            raise RuntimeError("edition publication conflicted and no edition was visible")
        return existing, False
    return edition, True


async def recover_day(
    day: date,
    settings: Settings,
    store: EditionStore,
    workflow: ForecastWorkflow,
) -> tuple[DailyEdition, Literal["recovered", "already_healthy", "no_improvement", "conflict"]]:
    stored = await store.read_version(day)
    if stored is None:
        raise ValueError(f"no published edition for {day}")
    edition = stored.edition
    missing = [(item.sign, item.topic) for item in edition.forecasts if item.source == "fallback"]
    if not missing:
        return edition, "already_healthy"

    candidates = await generate_forecasts(day, settings, workflow, missing)
    provenance = ReleaseProvenance(
        release_id=settings.release_id,
        prompt_version=PROMPT_VERSION,
        generator_id=workflow.generator.id,
    )
    replacements = {
        (item.sign, item.topic): item.model_copy(update={"recovered_with": provenance})
        for item in candidates
        if item.source == "model"
    }
    if not replacements:
        return edition, "no_improvement"

    updated = DailyEdition(
        date=edition.date,
        timezone=edition.timezone,
        release_id=edition.release_id,
        prompt_version=edition.prompt_version,
        generator_id=edition.generator_id,
        forecasts=[replacements.get((item.sign, item.topic), item) for item in edition.forecasts],
    )
    if not await store.replace(updated, stored.revision):
        return await store.read(day) or edition, "conflict"
    return updated, "recovered"
