import re
from dataclasses import dataclass
from datetime import date
from typing import Literal, Protocol

import httpx

from agata.domain import SIGN_NAMES, TOPIC_NAMES, Forecast, Sign, Topic
from agata.templates import render_template

PROMPT_VERSION = "ru-daily-v2"
SYSTEM_PROMPT = (
    "Ты пишешь короткие развлекательные гороскопы на русском. "
    "Верни только текст из 2–3 предложений для заданной темы, без заголовка и названий знаков. "
    "Избегай категоричных обещаний и конкретных инвестиционных рекомендаций."
)
RUSSIAN = re.compile(r"[а-яё]", re.IGNORECASE)
FINANCE_ADVICE = re.compile(
    r"гарантированн\w+\s+(прибыл\w*|доход\w*)|"
    r"(купите|продавайте|продайте)\s+(акци\w+|криптовалют\w+|биткоин\w*)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GenerationRequest:
    date: date
    sign: Sign
    topic: Topic
    attempt: int
    previous_issue: str | None = None


class GenerationUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class GeneratedText:
    text: str
    source: Literal["template", "model"]


class Generator(Protocol):
    id: str

    async def generate(self, request: GenerationRequest) -> GeneratedText: ...


def format_prompt(request: GenerationRequest) -> tuple[str, str]:
    correction = (
        f" Исправь проблему предыдущего варианта: {request.previous_issue}."
        if request.previous_issue
        else ""
    )
    user = f"Дата: {request.date}. Знак: {SIGN_NAMES[request.sign]}. Тема: {TOPIC_NAMES[request.topic]}."
    return SYSTEM_PROMPT, user + correction


class TemplateGenerator:
    id = "editorial-v1"

    async def generate(self, request: GenerationRequest) -> GeneratedText:
        text = render_template(str(request.date), request.sign, request.topic, request.attempt)
        return GeneratedText(text, "template")


class OpenAICompatibleGenerator:
    def __init__(self, client: httpx.AsyncClient, endpoint: str, model: str):
        self.client = client
        self.endpoint = endpoint
        self.id = model

    async def generate(self, request: GenerationRequest) -> GeneratedText:
        system, user = format_prompt(request)
        payload = {
            "model": self.id,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": 240,
            "temperature": 0.7,
        }
        try:
            response = await self.client.post(self.endpoint, json=payload)
            response.raise_for_status()
            text = response.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, ValueError, TypeError, KeyError, IndexError):
            raise GenerationUnavailable("модель временно недоступна") from None
        if not isinstance(text, str):
            raise GenerationUnavailable("модель не вернула текст")
        return GeneratedText(text.strip(), "model")


# Match capitalized forms to avoid treating common nouns such as "весы" as signs.
SIGN_FORMS = {
    Sign.ARIES: r"Ов(?:ен|н(?:а|у|ом|е|ы|ов|ам|ами|ах))",
    Sign.TAURUS: r"Тел(?:ец|ьц(?:а|у|ом|е|ы|ов|ам|ами|ах))",
    Sign.GEMINI: r"Близнец(?:ы|а|у|ом|е|ов|ам|ами|ах)",
    Sign.CANCER: r"Рак(?:а|у|ом|е|и|ов|ам|ами|ах)?",
    Sign.LEO: r"Л(?:ев|ьв(?:а|у|ом|е|ы|ов|ам|ами|ах))",
    Sign.VIRGO: r"Дев(?:а|е|у|ой|ою|ы|ам|ами|ах)?",
    Sign.LIBRA: r"Вес(?:ы|ов|ам|ами|ах)",
    Sign.SCORPIO: r"Скорпион(?:а|у|ом|е|ы|ов|ам|ами|ах)?",
    Sign.SAGITTARIUS: r"Стрел(?:ец|ьц(?:а|у|ом|е|ы|ов|ам|ами|ах))",
    Sign.CAPRICORN: r"Козерог(?:а|у|ом|е|и|ов|ам|ами|ах)?",
    Sign.AQUARIUS: r"Водоле(?:й|я|ю|ем|е|и|ев|ям|ями|ях)",
    Sign.PISCES: r"Рыб(?:а|ы|е|у|ой|ою|ам|ами|ах)?",
}
SIGN_PATTERNS = {sign: re.compile(rf"(?<!\w){form}(?!\w)") for sign, form in SIGN_FORMS.items()}
TOPIC_PATTERNS = {
    Topic.WORK: re.compile(
        r"\b(?:рабоч\w+|работ(?:а|ы|е|у|ой)|коллег\w*|дедлайн\w*|совещани\w+|руководител\w+)\b",
        re.IGNORECASE,
    ),
    Topic.RELATIONSHIPS: re.compile(
        r"\b(?:любим(?:ый|ая|ым|ой|ого|ому)|влюбл\w+|"
        r"(?:личн\w+|романтическ\w+)\s+отношени\w+|"
        r"(?:близк\w+|любим\w+|романтическ\w+)\s+человек\w*|"
        r"с\s+партн[её]ром\s+о\s+(?:ваших\s+)?чувствах)\b",
        re.IGNORECASE,
    ),
    Topic.FINANCE: re.compile(
        r"\b(?:бюджет\w*|расход\w*|трат\w*|финанс\w*|сбережени\w+|"
        r"доход\w*|инвестици\w+)\b",
        re.IGNORECASE,
    ),
}
ABBREVIATION = re.compile(
    r"(?<!\w)(?:т\.\s*[депк]\.|и\s+т\.\s*(?:д|п)\.|"
    r"(?:руб|коп|тыс|млн|им|рис|стр)\.|[А-ЯЁ]\.(?=\s*[А-ЯЁ]\.))"
)
SENTENCE_END = re.compile(r"[.!?…]+[»”\")\]]*(?=\s|$)")


def editorial_body(text: str, sign: Sign, topic: Topic) -> str:
    heading = f"{SIGN_NAMES[sign]} · {TOPIC_NAMES[topic]}. "
    return text.removeprefix(heading)


def validate_text(text: str, sign: Sign, topic: Topic) -> str | None:
    if not 60 <= len(text) <= 500:
        return "длина вне диапазона 60–500 символов"
    letters = sum(char.isalpha() for char in text)
    if len(RUSSIAN.findall(text)) < max(30, int(letters * 0.75)):
        return "слишком мало русского текста"
    if any(other != sign and pattern.search(text) for other, pattern in SIGN_PATTERNS.items()):
        return "упомянут другой знак"
    body = editorial_body(text, sign, topic)
    body_without_abbreviations = ABBREVIATION.sub(lambda match: match[0].replace(".", ""), body)
    sentences = [part for part in SENTENCE_END.split(body_without_abbreviations) if part.strip()]
    if not 2 <= len(sentences) <= 3:
        return "нужно 2–3 предложения"
    if (
        topic in TOPIC_PATTERNS
        and not TOPIC_PATTERNS[topic].search(body)
        and any(pattern.search(body) for other, pattern in TOPIC_PATTERNS.items() if other != topic)
    ):
        return f"текст явно относится к другой теме, нужна тема «{TOPIC_NAMES[topic]}»"
    if FINANCE_ADVICE.search(body):
        return "конкретная инвестиционная рекомендация или обещание доходности"
    return None


class ForecastWorkflow:
    def __init__(self, generator: Generator):
        self.generator = generator

    async def run(self, day: date, sign: Sign, topic: Topic) -> Forecast:
        issue = None
        for attempt in (1, 2):
            request = GenerationRequest(day, sign, topic, attempt, issue)
            try:
                candidate = await self.generator.generate(request)
            except GenerationUnavailable as exc:
                issue = str(exc)
                continue
            issue = validate_text(candidate.text, sign, topic)
            if issue is None:
                return Forecast(
                    sign=sign,
                    topic=topic,
                    text=candidate.text,
                    source=candidate.source,
                    attempts=attempt,
                )
        return Forecast(
            sign=sign,
            topic=topic,
            text=render_template(str(day), sign, topic),
            source="fallback",
            attempts=2,
            validation_issue=issue,
        )
