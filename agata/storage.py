import asyncio
import fcntl
import os
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol

import boto3
from botocore.exceptions import ClientError

from agata.config import Settings
from agata.domain import DailyEdition


@dataclass(frozen=True)
class StoredEdition:
    edition: DailyEdition
    revision: str


class EditionStore(Protocol):
    async def read(self, day: date) -> DailyEdition | None: ...
    async def read_version(self, day: date) -> StoredEdition | None: ...
    async def publish(self, edition: DailyEdition) -> bool: ...
    async def replace(self, edition: DailyEdition, revision: str) -> bool: ...


def decode_edition(data: bytes, day: date) -> DailyEdition:
    edition = DailyEdition.model_validate_json(data)
    if edition.date != day:
        raise ValueError("edition date does not match its storage key")
    return edition


class FileEditionStore:
    def __init__(self, directory: Path):
        self.directory = directory

    async def read(self, day: date) -> DailyEdition | None:
        stored = await self.read_version(day)
        return stored.edition if stored else None

    async def read_version(self, day: date) -> StoredEdition | None:
        return await asyncio.to_thread(self._read_version, day)

    def _read_version(self, day: date) -> StoredEdition | None:
        try:
            payload = (self.directory / f"{day}.json").read_bytes()
        except FileNotFoundError:
            return None
        return StoredEdition(decode_edition(payload, day), sha256(payload).hexdigest())

    async def publish(self, edition: DailyEdition) -> bool:
        return await asyncio.to_thread(self._publish, edition)

    def _publish(self, edition: DailyEdition) -> bool:
        self.directory.mkdir(parents=True, exist_ok=True)
        destination = self.directory / f"{edition.date}.json"
        with NamedTemporaryFile(dir=self.directory) as temporary:
            temporary.write(edition.model_dump_json().encode())
            temporary.flush()
            os.fsync(temporary.fileno())
            try:
                os.link(temporary.name, destination)
            except FileExistsError:
                return False
        return True

    async def replace(self, edition: DailyEdition, revision: str) -> bool:
        return await asyncio.to_thread(self._replace, edition, revision)

    def _replace(self, edition: DailyEdition, revision: str) -> bool:
        locks = self.directory / ".locks"
        locks.mkdir(exist_ok=True)
        destination = self.directory / f"{edition.date}.json"
        with (locks / f"{edition.date}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            current = self._read_version(edition.date)
            if current is None or current.revision != revision:
                return False
            with NamedTemporaryFile(dir=self.directory, delete=False) as temporary:
                temporary_path = Path(temporary.name)
                try:
                    temporary.write(edition.model_dump_json().encode())
                    temporary.flush()
                    os.fsync(temporary.fileno())
                    os.replace(temporary_path, destination)
                finally:
                    temporary_path.unlink(missing_ok=True)
        return True


class S3EditionStore:
    def __init__(self, bucket: str):
        self.bucket = bucket
        self.client = boto3.client("s3")

    async def read(self, day: date) -> DailyEdition | None:
        stored = await self.read_version(day)
        return stored.edition if stored else None

    async def read_version(self, day: date) -> StoredEdition | None:
        return await asyncio.to_thread(self._read_version, day)

    def _read_version(self, day: date) -> StoredEdition | None:
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=f"daily/{day}.json")
        except ClientError as exc:
            if exc.response["Error"]["Code"] not in {"NoSuchKey", "404", "NotFound"}:
                raise
            return None
        with response["Body"] as body:
            return StoredEdition(decode_edition(body.read(), day), response["ETag"])

    async def publish(self, edition: DailyEdition) -> bool:
        return await asyncio.to_thread(self._write, edition, {"IfNoneMatch": "*"})

    async def replace(self, edition: DailyEdition, revision: str) -> bool:
        return await asyncio.to_thread(self._write, edition, {"IfMatch": revision})

    def _write(self, edition: DailyEdition, condition: dict[str, str]) -> bool:
        try:
            self.client.put_object(
                Bucket=self.bucket,
                Key=f"daily/{edition.date}.json",
                Body=edition.model_dump_json().encode(),
                ContentType="application/json",
                **condition,
            )
        except ClientError as exc:
            if exc.response["Error"]["Code"] not in {
                "PreconditionFailed",
                "412",
                "ConditionalRequestConflict",
                "409",
            }:
                raise
            return False
        return True


def build_store(settings: Settings) -> EditionStore:
    if settings.s3_bucket:
        return S3EditionStore(settings.s3_bucket)
    return FileEditionStore(settings.edition_dir)
