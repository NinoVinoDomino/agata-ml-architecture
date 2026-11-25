from datetime import date, datetime
from enum import StrEnum
from itertools import product
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, model_validator


class Sign(StrEnum):
    ARIES = "aries"
    TAURUS = "taurus"
    GEMINI = "gemini"
    CANCER = "cancer"
    LEO = "leo"
    VIRGO = "virgo"
    LIBRA = "libra"
    SCORPIO = "scorpio"
    SAGITTARIUS = "sagittarius"
    CAPRICORN = "capricorn"
    AQUARIUS = "aquarius"
    PISCES = "pisces"


class Topic(StrEnum):
    GENERAL = "general"
    RELATIONSHIPS = "relationships"
    WORK = "work"
    FINANCE = "finance"


SIGN_NAMES = dict(
    zip(
        Sign,
        (
            "Овен",
            "Телец",
            "Близнецы",
            "Рак",
            "Лев",
            "Дева",
            "Весы",
            "Скорпион",
            "Стрелец",
            "Козерог",
            "Водолей",
            "Рыбы",
        ),
        strict=True,
    )
)

TOPIC_NAMES = {
    Topic.GENERAL: "Общее",
    Topic.RELATIONSHIPS: "Отношения",
    Topic.WORK: "Работа",
    Topic.FINANCE: "Финансы",
}
COMBINATIONS = tuple(product(Sign, Topic))


def local_date(timezone: str, now: datetime | None = None) -> date:
    zone = ZoneInfo(timezone)
    return datetime.now(zone).date() if now is None else now.astimezone(zone).date()


class ReleaseProvenance(BaseModel):
    release_id: str
    prompt_version: str
    generator_id: str


class Forecast(BaseModel):
    sign: Sign
    topic: Topic
    text: str = Field(min_length=60, max_length=500)
    source: Literal["template", "model", "fallback"]
    attempts: int = Field(ge=1, le=2)
    validation_issue: str | None = None
    recovered_with: ReleaseProvenance | None = None


class DailyEdition(BaseModel):
    schema_version: Literal[1] = 1
    date: date
    timezone: str
    release_id: str
    prompt_version: str
    generator_id: str
    forecasts: list[Forecast]

    @model_validator(mode="after")
    def complete_matrix(self) -> "DailyEdition":
        keys = {(item.sign, item.topic) for item in self.forecasts}
        if len(self.forecasts) != len(COMBINATIONS) or keys != set(COMBINATIONS):
            raise ValueError("edition must contain each sign and topic exactly once")
        return self

    def get(self, sign: Sign, topic: Topic) -> Forecast:
        return next(item for item in self.forecasts if (item.sign, item.topic) == (sign, topic))


class PublicForecast(BaseModel):
    date: date
    timezone: str
    sign: Sign
    topic: Topic
    text: str
    release_id: str
