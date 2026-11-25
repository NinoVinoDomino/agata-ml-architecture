import json
from collections.abc import Iterator
from datetime import date
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from agata.domain import Sign, Topic
from agata.generation import GenerationRequest, editorial_body, format_prompt, validate_text

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class TrainingExample(BaseModel):
    date: date
    sign: Sign
    topic: Topic
    text: str = Field(min_length=60, max_length=500)
    source: str = Field(min_length=1)

    @field_validator("text", "source", mode="before")
    @classmethod
    def strip_whitespace(cls, value):
        return value.strip() if isinstance(value, str) else value

    def request(self) -> GenerationRequest:
        return GenerationRequest(self.date, self.sign, self.topic, attempt=1)


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{number}: expected a JSON object")
            yield row


def read_examples(path: Path) -> list[TrainingExample]:
    examples = []
    for number, row in enumerate(read_jsonl(path), 1):
        example = TrainingExample.model_validate(row)
        if editorial_body(example.text, example.sign, example.topic) != example.text:
            raise ValueError(f"{path} record {number}: provide only the body without a heading")
        issue = validate_text(example.text, example.sign, example.topic)
        if issue:
            raise ValueError(f"{path} record {number}: {issue}")
        examples.append(example)
    if not examples:
        raise ValueError(f"{path}: no examples")
    return examples


def normalized(text: str) -> str:
    return " ".join("".join(char if char.isalnum() else " " for char in text.casefold()).split())


def chat_messages(request: GenerationRequest) -> list[dict[str, str]]:
    system, user = format_prompt(request)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
