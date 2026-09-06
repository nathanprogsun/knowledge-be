from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from src.common.json import JsonObject


class ChunkingParserEngineRule(BaseModel):
    model_config = ConfigDict(frozen=True)

    file_types: list[str]
    engine: str


class WikiConfig(BaseModel):
    """Wiki-ingest settings for a knowledge base."""

    model_config = ConfigDict(frozen=True)

    synthesis_model_id: str | None = Field(default=None)
    max_pages_per_ingest: int | None = Field(default=0)
    extraction_granularity: str | None = Field(default=None)
    content_instructions: str | None = Field(default=None)
    extraction_instructions: str | None = Field(default=None)


class IndexingStrategy(BaseModel):
    """Flags indicating which knowledge-base pipelines are active."""

    model_config = ConfigDict(frozen=True)

    vector_enabled: bool = False
    keyword_enabled: bool = False
    wiki_enabled: bool = False
    graph_enabled: bool = False


class ChunkingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_size: int
    chunk_overlap: int
    separators: list[str] | None = Field(default=None)
    enable_multimodal: bool = False
    parser_engine_rules: list[ChunkingParserEngineRule] | None = Field(default=None)
    enable_parent_child: bool = False
    parent_chunk_size: int | None = Field(default=None)
    child_chunk_size: int | None = Field(default=None)


class ImageProcessingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str | None = Field(default=None)


class VLMConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool
    model_id: str | None = Field(default=None)


class ASRConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool
    model_id: str | None = Field(default=None)
    language: str | None = Field(default=None)


class StorageProviderConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str


class LegacyStorageConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    secret_id: str | None = Field(default=None)
    secret_key: str | None = Field(default=None)
    region: str | None = Field(default=None)
    bucket_name: str | None = Field(default=None)
    app_id: str | None = Field(default=None)
    path_prefix: str | None = Field(default=None)


class ExtractConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool
    text: str | None = Field(default=None)
    tags: list[str] | None = Field(default=None)
    nodes: list[str] | None = Field(default=None)
    relations: list[str] | None = Field(default=None)


class FAQConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    default_question_count: int | None = Field(default=None)


class QuestionGenerationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool
    question_count: int | None = Field(default=None)


class KnowledgeBase(BaseModel):
    """Knowledge-base representation with persisted data and response enrichments.

    List responses may include counts, creator details, permissions, and
    vector-store status. ``is_processing`` and ``share_count`` are runtime-only
    response fields.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    description: str | None = Field(default=None)
    type: str
    is_temporary: bool = False
    tenant_id: int
    creator_id: str | None = Field(default=None)
    creator_name: str | None = Field(default=None)
    chunking_config: ChunkingConfig | None = Field(default=None)
    image_processing_config: ImageProcessingConfig | None = Field(default=None)
    embedding_model_id: str | None = Field(default=None)
    summary_model_id: str | None = Field(default=None)
    vlm_config: VLMConfig | None = Field(default=None)
    asr_config: ASRConfig | None = Field(default=None)
    storage_provider_config: StorageProviderConfig | None = Field(default=None)
    storage_backend_id: str | None = Field(default=None)
    storage_config: LegacyStorageConfig | None = Field(default=None)
    extract_config: ExtractConfig | None = Field(default=None)
    faq_config: FAQConfig | None = Field(default=None)
    question_generation_config: QuestionGenerationConfig | None = Field(default=None)
    wiki_config: WikiConfig | None = Field(default=None)
    indexing_strategy: IndexingStrategy | None = Field(default=None)
    vector_store_id: str | None = Field(default=None)
    vector_store_name: str | None = Field(default=None)
    vector_store_source: str | None = Field(default=None)
    vector_store_engine_type: str | None = Field(default=None)
    vector_store_status: str | None = Field(default=None)
    is_pinned: bool = False
    pinned_at: datetime | None = Field(default=None)
    knowledge_count: int = 0
    chunk_count: int = 0
    processing_count: int = 0
    is_processing: bool = False
    share_count: int = 0
    my_permission: str | None = Field(default=None)
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = Field(default=None)


class CreateKnowledgeBaseRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    description: str | None = Field(default=None)
    type: str | None = Field(default="document")
    is_temporary: bool = False
    chunking_config: ChunkingConfig | None = Field(default=None)
    image_processing_config: ImageProcessingConfig | None = Field(default=None)
    embedding_model_id: str | None = Field(default=None)
    summary_model_id: str | None = Field(default=None)
    vlm_config: VLMConfig | None = Field(default=None)
    asr_config: ASRConfig | None = Field(default=None)
    storage_provider_config: StorageProviderConfig | None = Field(default=None)
    # Binds the knowledge base to a concrete storage backend row at
    # creation; when omitted the service resolves the tenant default.
    storage_backend_id: str | None = Field(default=None)
    storage_config: LegacyStorageConfig | None = Field(default=None)
    extract_config: ExtractConfig | None = Field(default=None)
    faq_config: FAQConfig | None = Field(default=None)
    question_generation_config: QuestionGenerationConfig | None = Field(default=None)
    wiki_config: WikiConfig | None = Field(default=None)
    indexing_strategy: IndexingStrategy | None = Field(default=None)
    vector_store_id: str | None = Field(default=None)


class UpdateKnowledgeBaseRequest(BaseModel):
    """Partial-update body: every field optional so callers can PATCH a subset.

    The service layer skips a field when its value is ``None`` and
    inherits the existing row's value, so the same request shape works
    for both PUT (full body) and PATCH (subset).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str | None = Field(default=None)
    description: str | None = Field(default=None)
    config: JsonObject | None = Field(default=None)


class HybridSearchRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    query_text: str
    vector_threshold: float | None = Field(default=None)
    keyword_threshold: float | None = Field(default=None)
    match_count: int | None = Field(default=None)
    disable_keywords_match: bool = False
    disable_vector_match: bool = False
    knowledge_ids: list[str] | None = Field(default=None)
    tag_ids: list[str] | None = Field(default=None)
    only_recommended: bool = False
    knowledge_base_ids: list[str] | None = Field(default=None)
    skip_context_enrichment: bool = False


class KnowledgeSearchRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    knowledge_base_id: str | None = Field(default=None)
    knowledge_base_ids: list[str] | None = Field(default=None)
    knowledge_ids: list[str] | None = Field(default=None)


class KnowledgeSearchHit(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    content: str
    knowledge_id: str
    chunk_index: int
    knowledge_title: str
    start_at: int | None = Field(default=None)
    end_at: int | None = Field(default=None)
    seq: int
    score: float
    chunk_type: str
    image_info: str | None = Field(default=None)
    metadata: JsonObject | None = Field(default=None)
    knowledge_filename: str | None = Field(default=None)
    knowledge_source: str | None = Field(default=None)


class KnowledgeCopyRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_id: str
    target_id: str | None = Field(default=None)
    task_id: str | None = Field(default=None)


class KnowledgeCopyResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    source_id: str
    target_id: str
    message: str


class KnowledgeCopyProgress(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    source_id: str
    target_id: str | None = Field(default=None)
    status: str
    progress: int
    total: int
    processed: int
    message: str | None = Field(default=None)
    error: str | None = Field(default=None)
    created_at: int
    updated_at: int


class KnowledgeDuplicateResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_id: str
    target_id: str
    message: str
    knowledge_base: KnowledgeBase


class Knowledge(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    tenant_id: int
    knowledge_base_id: str
    type: str
    title: str | None = Field(default=None)
    description: str | None = Field(default=None)
    source: str | None = Field(default=None)
    channel: str | None = Field(default=None)
    tag_id: str | None = Field(default=None)
    summary_status: str | None = Field(default=None)
    parse_status: str
    enable_status: str
    embedding_model_id: str | None = Field(default=None)
    file_name: str | None = Field(default=None)
    file_type: str | None = Field(default=None)
    file_size: int | None = Field(default=None)
    file_hash: str | None = Field(default=None)
    file_path: str | None = Field(default=None)
    storage_size: int | None = Field(default=None)
    metadata: JsonObject | None = Field(default=None)
    created_at: datetime
    updated_at: datetime
    processed_at: datetime | None = Field(default=None)
    error_message: str | None = Field(default=None)
    deleted_at: datetime | None = Field(default=None)
    knowledge_base_name: str | None = Field(default=None)


class CreateKnowledgeFromURLRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    url: str
    file_name: str | None = Field(default=None)
    file_type: str | None = Field(default=None)
    enable_multimodel: bool = False
    title: str | None = Field(default=None)
    tag_id: str | None = Field(default=None)
    channel: str | None = Field(default=None)


class CreateManualKnowledgeRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str
    content: str
    status: str | None = Field(default=None)
    tag_id: str | None = Field(default=None)
    channel: str | None = Field(default=None)


class ListKnowledgeQuery(BaseModel):
    model_config = ConfigDict(frozen=True)

    page: int = 1
    page_size: int = 20
    tag_id: str | None = Field(default=None)
    keyword: str | None = Field(default=None)
    file_type: str | None = Field(default=None)
    parse_status: str | None = Field(default=None)
    source: str | None = Field(default=None)
    start_time: str | None = Field(default=None)
    end_time: str | None = Field(default=None)


class UpdateKnowledgeRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str | None = Field(default=None)
    description: str | None = Field(default=None)
    tag_id: str | None = Field(default=None)
    enable_status: str | None = Field(default=None)
    content: str | None = Field(default=None)
    status: str | None = Field(default=None)
    process_config: JsonObject | None = Field(default=None)


class UpdateKnowledgeImageRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    image_info: str


class BatchUpdateKnowledgeTagsRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    updates: JsonObject
    kb_id: str | None = Field(default=None)


class SearchKnowledgeQuery(BaseModel):
    model_config = ConfigDict(frozen=True)

    keyword: str | None = Field(default=None)
    offset: int = 0
    limit: int = 20
    file_types: str | None = Field(default=None)
    agent_id: str | None = Field(default=None)


class KnowledgeBatchReparseRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    kb_id: str
    ids: list[str]
    process_config: JsonObject | None = Field(default=None)


class KnowledgeBatchReparseResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    reparse_count: int


class KnowledgeBatchDeleteRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    kb_id: str
    ids: list[str]


class KnowledgeBatchDeleteResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    deleted_count: int


class KnowledgeMoveRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    knowledge_ids: list[str]
    source_kb_id: str
    target_kb_id: str
    mode: str


class KnowledgeMoveResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    source_kb_id: str
    target_kb_id: str
    knowledge_count: int
    message: str


class KnowledgeMoveProgress(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    source_kb_id: str
    target_kb_id: str
    status: str
    progress: int
    total: int
    processed: int
    failed: int
    message: str | None = Field(default=None)
    error: str | None = Field(default=None)
    created_at: int
    updated_at: int


class KnowledgeClearResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    deleted_count: int


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    tenant_id: int
    knowledge_id: str
    knowledge_base_id: str
    tag_id: str | None = Field(default=None)
    content: str
    chunk_index: int
    is_enabled: bool
    status: int
    start_at: int | None = Field(default=None)
    end_at: int | None = Field(default=None)
    pre_chunk_id: str | None = Field(default=None)
    next_chunk_id: str | None = Field(default=None)
    chunk_type: str
    parent_chunk_id: str | None = Field(default=None)
    relation_chunks: list[str] | None = Field(default=None)
    indirect_relation_chunks: list[str] | None = Field(default=None)
    metadata: JsonObject | None = Field(default=None)
    content_hash: str | None = Field(default=None)
    image_info: str | None = Field(default=None)
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = Field(default=None)


class UpdateChunkRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    content: str | None = Field(default=None)
    chunk_index: int | None = Field(default=None)
    is_enabled: bool | None = Field(default=None)
    start_at: int | None = Field(default=None)
    end_at: int | None = Field(default=None)
    image_info: str | None = Field(default=None)


class DeleteChunkQuestionRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    question_id: str


class Tag(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    tenant_id: int
    knowledge_base_id: str
    name: str
    color: str | None = Field(default=None)
    sort_order: int | None = Field(default=None)
    knowledge_count: int = 0
    chunk_count: int = 0
    created_at: datetime
    updated_at: datetime


class TagList(BaseModel):
    model_config = ConfigDict(frozen=True)

    total: int
    page: int
    page_size: int
    data: list[Tag]


class CreateTagRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    color: str | None = Field(default=None)
    sort_order: int | None = Field(default=None)


class UpdateTagRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str | None = Field(default=None)
    color: str | None = Field(default=None)
    sort_order: int | None = Field(default=None)


class ListTagsQuery(BaseModel):
    model_config = ConfigDict(frozen=True)

    page: int = 1
    page_size: int = 20
    keyword: str | None = Field(default=None)


class FAQEntryPayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int | None = Field(default=None)
    standard_question: str
    similar_questions: list[str] | None = Field(default=None)
    negative_questions: list[str] | None = Field(default=None)
    answers: list[str] | None = Field(default=None)
    answer_strategy: str | None = Field(default=None)
    tag_id: int | None = Field(default=None)
    tag_name: str | None = Field(default=None)
    is_enabled: bool | None = Field(default=None)
    is_recommended: bool | None = Field(default=None)


class FAQBatchUpsertPayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    entries: list[FAQEntryPayload]
    mode: str
    knowledge_id: str | None = Field(default=None)
    task_id: str | None = Field(default=None)
    dry_run: bool = False


class FAQEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int
    chunk_id: str
    knowledge_id: str
    knowledge_base_id: str
    tag_id: int | None = Field(default=None)
    tag_name: str | None = Field(default=None)
    is_enabled: bool
    is_recommended: bool
    standard_question: str
    similar_questions: list[str]
    negative_questions: list[str]
    answers: list[str]
    answer_strategy: str
    index_mode: str | None = Field(default=None)
    chunk_type: str
    score: float | None = Field(default=None)
    match_type: str | None = Field(default=None)
    matched_question: str | None = Field(default=None)
    created_at: datetime
    updated_at: datetime


class FAQEntryListResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    total: int
    page: int
    page_size: int
    data: list[FAQEntry]


class ListFAQEntriesQuery(BaseModel):
    model_config = ConfigDict(frozen=True)

    page: int = 1
    page_size: int = 20
    tag_id: int | None = Field(default=None)
    keyword: str | None = Field(default=None)
    search_field: str | None = Field(default=None)
    sort_order: str | None = Field(default=None)


class FAQSearchRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    query_text: str
    vector_threshold: float | None = Field(default=None)
    match_count: int | None = Field(default=None)
    first_priority_tag_ids: list[int] | None = Field(default=None)
    second_priority_tag_ids: list[int] | None = Field(default=None)
    only_recommended: bool = False


class FAQEntryFieldsUpdate(BaseModel):
    model_config = ConfigDict(frozen=True)

    is_enabled: bool | None = Field(default=None)
    is_recommended: bool | None = Field(default=None)
    tag_id: int | None = Field(default=None)


class FAQEntryFieldsBatchUpdate(BaseModel):
    model_config = ConfigDict(frozen=True)

    by_id: dict[int, FAQEntryFieldsUpdate] | None = Field(default=None)
    by_tag: dict[int, FAQEntryFieldsUpdate] | None = Field(default=None)
    exclude_ids: list[int] | None = Field(default=None)


class FAQEntryTagsBatchUpdate(BaseModel):
    model_config = ConfigDict(frozen=True)

    updates: dict[int, int]


class FAQBatchDeleteRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    ids: list[int]


class FAQImportDisplayStatusRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    display_status: str


class FAQSimilarQuestionsRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    similar_questions: list[str]


class FAQImportTaskProgress(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    kb_id: str | None = Field(default=None)
    knowledge_id: str | None = Field(default=None)
    status: str
    progress: int
    total: int
    processed: int
    success_count: int | None = Field(default=None)
    failed_count: int | None = Field(default=None)
    partial_failed_count: int | None = Field(default=None)
    skipped_count: int | None = Field(default=None)
    failed_entries: list[JsonObject] | None = Field(default=None)
    failed_entries_url: str | None = Field(default=None)
    success_entries: list[JsonObject] | None = Field(default=None)
    message: str | None = Field(default=None)
    error: str | None = Field(default=None)
    created_at: int
    updated_at: int
    dry_run: bool = False
    import_mode: str | None = Field(default=None)
    imported_at: datetime | None = Field(default=None)
    display_status: str | None = Field(default=None)
    processing_time: int | None = Field(default=None)


__all__ = [
    "ASRConfig",
    "BatchUpdateKnowledgeTagsRequest",
    "Chunk",
    "ChunkingConfig",
    "ChunkingParserEngineRule",
    "CreateKnowledgeBaseRequest",
    "CreateKnowledgeFromURLRequest",
    "CreateManualKnowledgeRequest",
    "CreateTagRequest",
    "DeleteChunkQuestionRequest",
    "ExtractConfig",
    "FAQBatchDeleteRequest",
    "FAQBatchUpsertPayload",
    "FAQConfig",
    "FAQEntry",
    "FAQEntryFieldsBatchUpdate",
    "FAQEntryFieldsUpdate",
    "FAQEntryListResponse",
    "FAQEntryPayload",
    "FAQEntryTagsBatchUpdate",
    "FAQImportDisplayStatusRequest",
    "FAQImportTaskProgress",
    "FAQSearchRequest",
    "FAQSimilarQuestionsRequest",
    "HybridSearchRequest",
    "ImageProcessingConfig",
    "IndexingStrategy",
    "Knowledge",
    "KnowledgeBase",
    "KnowledgeBatchDeleteRequest",
    "KnowledgeBatchDeleteResponse",
    "KnowledgeBatchReparseRequest",
    "KnowledgeBatchReparseResponse",
    "KnowledgeClearResponse",
    "KnowledgeCopyProgress",
    "KnowledgeCopyRequest",
    "KnowledgeCopyResponse",
    "KnowledgeDuplicateResponse",
    "KnowledgeMoveProgress",
    "KnowledgeMoveRequest",
    "KnowledgeMoveResponse",
    "KnowledgeSearchHit",
    "KnowledgeSearchRequest",
    "LegacyStorageConfig",
    "ListFAQEntriesQuery",
    "ListKnowledgeQuery",
    "ListTagsQuery",
    "QuestionGenerationConfig",
    "SearchKnowledgeQuery",
    "StorageProviderConfig",
    "Tag",
    "TagList",
    "UpdateChunkRequest",
    "UpdateKnowledgeBaseRequest",
    "UpdateKnowledgeImageRequest",
    "UpdateKnowledgeRequest",
    "UpdateTagRequest",
    "VLMConfig",
    "WikiConfig",
]
