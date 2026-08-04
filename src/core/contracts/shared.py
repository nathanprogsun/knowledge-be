from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.common.json import JsonObject


class ErrorDetail(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str | None = Field(default=None)
    message: str
    details: str | None = Field(default=None)


class ErrorResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    success: bool = False
    error: ErrorDetail | None = Field(default=None)


class ValidationError(BaseModel):
    model_config = ConfigDict(frozen=True)

    field: str
    message: str


class FileInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    file_name: str
    file_type: str
    file_size: int
    file_hash: str
    file_path: str


class BatchOperationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    success: bool
    total: int
    processed: int
    failed: int = 0
    skipped: int = 0
    message: str | None = Field(default=None)


class AsyncTaskResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    status: str
    message: str | None = Field(default=None)


class TaskProgressResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    status: str
    progress: int
    total: int = 0
    processed: int = 0
    failed: int = 0
    message: str | None = Field(default=None)
    error: str | None = Field(default=None)
    created_at: int | None = Field(default=None)
    updated_at: int | None = Field(default=None)


class SSEResponseChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    response_type: str
    content: str | None = Field(default=None)
    done: bool
    data: JsonObject | None = Field(default=None)
    knowledge_references: list[JsonObject] | None = Field(default=None)


__all__ = [
    "AsyncTaskResponse",
    "BatchOperationResult",
    "ErrorDetail",
    "ErrorResponse",
    "FileInfo",
    "SSEResponseChunk",
    "TaskProgressResponse",
    "ValidationError",
]
