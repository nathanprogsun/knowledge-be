from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class SystemInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    version: str
    edition: str
    commit_id: str
    build_time: datetime
    go_version: str
    keyword_index_engine: str
    vector_store_engine: str
    graph_database_engine: str
    minio_enabled: bool
    db_version: str


class ParserEngine(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    label: str
    description: str
    available: bool


class ParserEnginesList(BaseModel):
    model_config = ConfigDict(frozen=True)

    engines: list[ParserEngine]
    connected: bool | None = Field(default=None)


class CheckParserEnginesRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    addr: str


class DocreaderReconnectRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    addr: str


class StorageEngineInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    available: bool
    description: str


class StorageEngineStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    engines: list[StorageEngineInfo]
    minio_env_available: bool


class StorageEngineCheckRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    minio: dict[str, object] | None = Field(default=None)


class StorageEngineCheckResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    ok: bool
    message: str
    bucket_created: bool = False


__all__ = [
    "CheckParserEnginesRequest",
    "DocreaderReconnectRequest",
    "ParserEngine",
    "ParserEnginesList",
    "StorageEngineCheckRequest",
    "StorageEngineCheckResult",
    "StorageEngineInfo",
    "StorageEngineStatus",
    "SystemInfo",
]
