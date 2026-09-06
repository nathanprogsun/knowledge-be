"""Wire shapes for ``GET/PUT /initialization/config/{kb_id}``.

Field names follow the SPA ``KBModelConfigRequest`` and Go's
``UpdateKBConfig`` body (camelCase model ids and documentSplitting).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.common.json import JsonObject
from src.core.infra.models.types import ModelInfo
from src.core.knowledge.knowledge_bases.types import KnowledgeBaseInfo


class VLMConfigBody(BaseModel):
    """Optional VLM slot on the save-and-close body."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    model_id: str = ""
    description_language: str = ""
    custom_instructions: str = ""


class ASRConfigBody(BaseModel):
    """Optional ASR slot on the save-and-close body."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    model_id: str = ""
    language: str = ""


class DocumentSplittingBody(BaseModel):
    """Chunking fields the editor sends as ``documentSplitting``."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    chunk_size: int = Field(default=0, alias="chunkSize")
    chunk_overlap: int = Field(default=0, alias="chunkOverlap")
    separators: list[str] = Field(default_factory=list)
    parser_engine_rules: list[JsonObject] | None = Field(default=None, alias="parserEngineRules")
    enable_parent_child: bool = Field(default=False, alias="enableParentChild")
    parent_chunk_size: int = Field(default=0, alias="parentChunkSize")
    child_chunk_size: int = Field(default=0, alias="childChunkSize")
    strategy: str | None = None
    token_limit: int | None = Field(default=None, alias="tokenLimit")
    languages: list[str] | None = None
    table_metadata_instructions: str | None = Field(default=None, alias="tableMetadataInstructions")


class MultimodalToggleBody(BaseModel):
    """Whether multimodal processing is on."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = False


class NodeExtractBody(BaseModel):
    """Graph-extract fields the editor sends as ``nodeExtract``."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    enabled: bool = False
    text: str = ""
    tags: list[str] = Field(default_factory=list)
    nodes: list[JsonObject] = Field(default_factory=list)
    relations: list[JsonObject] = Field(default_factory=list)
    custom_instructions: str = Field(default="", alias="customInstructions")


class QuestionGenerationBody(BaseModel):
    """Question-generation fields the editor sends as ``questionGeneration``."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    enabled: bool = False
    question_count: int = Field(default=3, alias="questionCount")
    custom_instructions: str = Field(default="", alias="customInstructions")


class KBModelConfigRequest(BaseModel):
    """PUT body for ``/initialization/config/{kb_id}``."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    llm_model_id: str = Field(alias="llmModelId")
    embedding_model_id: str = Field(default="", alias="embeddingModelId")
    vlm_config: VLMConfigBody | None = None
    asr_config: ASRConfigBody | None = None
    document_splitting: DocumentSplittingBody = Field(
        default_factory=DocumentSplittingBody, alias="documentSplitting"
    )
    multimodal: MultimodalToggleBody = Field(default_factory=MultimodalToggleBody)
    storage_backend_id: str = Field(default="", alias="storageBackendId")
    storage_provider: str = Field(default="local", alias="storageProvider")
    node_extract: NodeExtractBody = Field(default_factory=NodeExtractBody, alias="nodeExtract")
    question_generation: QuestionGenerationBody = Field(
        default_factory=QuestionGenerationBody, alias="questionGeneration"
    )


class KBConfigUpdateEnvelope(BaseModel):
    """``{"success": true, "message": "..."}`` after a config save."""

    model_config = ConfigDict(frozen=True)

    success: bool
    message: str


class KBConfigReadEnvelope(BaseModel):
    """``{"success": true, "data": {...}}`` for the current KB config."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: JsonObject


def chunking_from_request(body: KBModelConfigRequest) -> JsonObject:
    """Project ``documentSplitting`` onto the stored chunking_config keys."""
    split = body.document_splitting
    cfg: JsonObject = {
        "chunk_size": split.chunk_size,
        "chunk_overlap": split.chunk_overlap,
        "separators": list(split.separators),
        "enable_parent_child": split.enable_parent_child,
        "parent_chunk_size": split.parent_chunk_size,
        "child_chunk_size": split.child_chunk_size,
    }
    if split.parser_engine_rules is not None:
        cfg["parser_engine_rules"] = list(split.parser_engine_rules)
    if split.strategy is not None:
        cfg["strategy"] = split.strategy
    if split.token_limit is not None:
        cfg["token_limit"] = split.token_limit
    if split.languages is not None:
        cfg["languages"] = list(split.languages)
    if split.table_metadata_instructions is not None:
        cfg["table_metadata_instructions"] = split.table_metadata_instructions.strip()
    return cfg


def extract_from_request(body: KBModelConfigRequest) -> JsonObject:
    """Project ``nodeExtract`` onto the stored extract_config keys."""
    node = body.node_extract
    return {
        "enabled": node.enabled,
        "text": node.text,
        "tags": list(node.tags),
        "nodes": list(node.nodes),
        "relations": list(node.relations),
        "custom_instructions": node.custom_instructions.strip(),
    }


def question_generation_from_request(body: KBModelConfigRequest) -> JsonObject:
    """Project ``questionGeneration`` onto the stored JSON column."""
    gen = body.question_generation
    count = gen.question_count
    if count <= 0:
        count = 3
    if count > 10:
        count = 10
    return {
        "enabled": gen.enabled,
        "question_count": count,
        "custom_instructions": gen.custom_instructions.strip(),
    }


def vlm_from_request(body: KBModelConfigRequest, *, model_ok: bool) -> JsonObject:
    """Build the stored VLM config; drop the model id when unused."""
    vlm = body.vlm_config
    enabled = bool(vlm and vlm.enabled and body.multimodal.enabled and model_ok)
    return {
        "enabled": enabled,
        "model_id": (vlm.model_id if vlm and enabled else ""),
        "description_language": (vlm.description_language.strip() if vlm else ""),
        "custom_instructions": (vlm.custom_instructions.strip() if vlm else ""),
    }


def asr_from_request(body: KBModelConfigRequest, *, model_ok: bool) -> JsonObject:
    """Build the stored ASR config; drop the model id when unused."""
    asr = body.asr_config
    enabled = bool(asr and asr.enabled and model_ok)
    return {
        "enabled": enabled,
        "model_id": (asr.model_id if asr and enabled else ""),
        "language": (asr.language if asr and enabled else ""),
    }


def config_read_payload(
    info: KnowledgeBaseInfo,
    *,
    models: list[ModelInfo],
    has_files: bool,
) -> JsonObject:
    """Build the GET ``data`` object the SPA already consumes."""
    by_id = {model.id: model for model in models}
    llm = by_id.get(info.summary_model_id)
    embedding = by_id.get(info.embedding_model_id)
    chunking = info.chunking_config or {}
    extract = info.extract_config or {"enabled": False}
    question = info.question_generation_config or {"enabled": False}
    return {
        "hasFiles": has_files,
        "llm": _slot(llm),
        "embedding": _embedding_slot(embedding),
        "rerank": {"enabled": False, "modelName": "", "baseUrl": ""},
        "multimodal": {"enabled": _vlm_enabled(info.vlm_config)},
        "documentSplitting": {
            "chunkSize": chunking.get("chunk_size", 0),
            "chunkOverlap": chunking.get("chunk_overlap", 0),
            "separators": chunking.get("separators", []),
            "strategy": chunking.get("strategy", ""),
            "tokenLimit": chunking.get("token_limit", 0),
            "languages": chunking.get("languages", []),
        },
        "nodeExtract": extract,
        "questionGeneration": question,
    }


def _slot(model: ModelInfo | None) -> JsonObject:
    if model is None:
        return {"source": "", "modelName": "", "baseUrl": ""}
    return {"source": model.source, "modelName": model.name, "baseUrl": ""}


def _embedding_slot(model: ModelInfo | None) -> JsonObject:
    slot = _slot(model)
    dimension = 0
    params = None if model is None else model.parameters.embedding_parameters
    if params is not None and params.dimension is not None:
        dimension = params.dimension
    slot["dimension"] = dimension
    return slot


def _vlm_enabled(vlm_config: JsonObject | None) -> bool:
    return bool(isinstance(vlm_config, dict) and vlm_config.get("enabled") is True)


__all__ = [
    "ASRConfigBody",
    "DocumentSplittingBody",
    "KBConfigReadEnvelope",
    "KBConfigUpdateEnvelope",
    "KBModelConfigRequest",
    "MultimodalToggleBody",
    "NodeExtractBody",
    "QuestionGenerationBody",
    "VLMConfigBody",
    "asr_from_request",
    "chunking_from_request",
    "config_read_payload",
    "extract_from_request",
    "question_generation_from_request",
    "vlm_from_request",
]
