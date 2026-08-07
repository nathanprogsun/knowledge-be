"""Elasticsearch v8 retrieve engine repository.

Maps the upstream v8 contract: index management (create-if-not-exists +
field-type detection), keyword retrieval (BM25 match), vector retrieval
(script-score cosine similarity with source filtering), bulk save, delete-
by-query, copy-indices, and batch update-by-query. Uses the
``elasticsearch`` (v8+) sync client; SDK calls are wrapped in
``asyncio.to_thread`` so the async event loop is not blocked.

The v8 driver advertises both keyword and vector retrieval (``support``
returns ``[Keywords, Vector]``); ``retrieve`` dispatches to the
appropriate path based on ``RetrieverType``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, cast

from elasticsearch import Elasticsearch

from src.ai.retrieval._es_common import (
    build_base_conds,
    calculate_storage_size,
    id_field,
    parse_search_hits,
    resolve_index_name,
    to_db_vector_embedding,
)
from src.ai.retrieval.base import AppConfig, Context, RetrieveEngineRepository
from src.ai.retrieval.elasticsearch_v7 import _process_source_batch
from src.ai.retrieval.types import (
    IndexConfig,
    IndexInfo,
    IndexSaveParams,
    MatchType,
    RetrieveParams,
    RetrieverEngineType,
    RetrieveResult,
    RetrieverType,
)
from src.app_logging import logger

_ENV_INDEX_KEY: str = "ELASTICSEARCH_INDEX"
_DEFAULT_INDEX_NAME: str = "xwrag_default"
_COPY_BATCH_SIZE: int = 500
_EMBEDDING_FIELD: str = "embedding"


def new_elasticsearch_v8_client(
    addr: str, username: str, password: str
) -> Elasticsearch:
    """Build an ``elasticsearch`` (v8+) sync client from connection params."""
    basic_auth = (username, password) if username else None
    return Elasticsearch(hosts=addr if addr else None, basic_auth=basic_auth)


class ElasticsearchV8Repository(RetrieveEngineRepository):
    """Elasticsearch v8 engine repository."""

    def __init__(
        self,
        client: Elasticsearch,
        index_config: IndexConfig | None = None,
    ) -> None:
        self._client = client
        self._index = resolve_index_name(index_config, _ENV_INDEX_KEY, _DEFAULT_INDEX_NAME)
        self._number_of_shards = index_config.number_of_shards if index_config else 0
        self._number_of_replicas = index_config.number_of_replicas if index_config else -1
        self._use_keyword_suffix = True
        self._init_index()

    def _init_index(self) -> None:
        try:
            self._create_index_if_not_exists()
        except Exception as exc:
            logger.error("[Elasticsearch] Failed to create index: {}", exc)
        try:
            self._detect_field_types()
        except Exception as exc:
            logger.warning("[Elasticsearch] Field type detection failed: {}", exc)

    def _create_index_if_not_exists(self) -> None:
        if self._client.indices.exists(index=self._index):
            return
        settings: dict[str, Any] = {}
        if self._number_of_shards > 0:
            settings["number_of_shards"] = str(self._number_of_shards)
        if self._number_of_replicas >= 0:
            settings["number_of_replicas"] = str(self._number_of_replicas)
        self._client.indices.create(index=self._index, settings=settings or None)

    def _detect_field_types(self) -> None:
        resp = self._client.indices.get_mapping(index=self._index)
        index_data = resp.get(self._index, {})
        mappings = index_data.get("mappings", {})
        properties = mappings.get("properties", {})
        chunk_id_prop = properties.get("chunk_id", {})
        field_type = chunk_id_prop.get("type", "")
        self._use_keyword_suffix = field_type != "keyword"

    def _id_field(self, name: str) -> str:
        return id_field(name, self._use_keyword_suffix)

    # ── protocol: engine_type / support ───────────────────────────────

    def engine_type(self) -> RetrieverEngineType:
        return RetrieverEngineType.ELASTICSEARCH

    def support(self) -> list[RetrieverType]:
        return [RetrieverType.KEYWORDS, RetrieverType.VECTOR]

    # ── protocol: estimate_storage_size ───────────────────────────────

    def estimate_storage_size(
        self,
        ctx: Context,
        index_info_list: list[IndexInfo],
        params: IndexSaveParams,
    ) -> int:
        del ctx
        total = 0
        for info in index_info_list:
            doc = to_db_vector_embedding(info, params)
            total += calculate_storage_size(doc)
        return total

    # ── protocol: save / batch_save ───────────────────────────────────

    async def save(
        self, ctx: Context, index_info: IndexInfo, params: IndexSaveParams
    ) -> None:
        del ctx
        doc = to_db_vector_embedding(index_info, params)
        if not doc.get("embedding"):
            raise ValueError(f"empty embedding vector for chunk ID: {index_info.chunk_id}")
        await asyncio.to_thread(
            self._client.index, index=self._index, document=doc
        )

    async def batch_save(
        self,
        ctx: Context,
        index_info_list: list[IndexInfo],
        params: IndexSaveParams,
    ) -> None:
        del ctx
        if not index_info_list:
            return
        operations: list[dict[str, Any]] = []
        for info in index_info_list:
            doc = to_db_vector_embedding(info, params)
            operations.append({"index": {"_index": self._index}})
            operations.append(doc)
        await asyncio.to_thread(
            self._client.bulk, operations=operations, index=self._index
        )

    # ── protocol: delete_by_* ─────────────────────────────────────────

    async def delete_by_chunk_id_list(
        self, ctx: Context, index_id_list: list[str], dimension: int, knowledge_type: str
    ) -> None:
        del ctx, dimension, knowledge_type
        await self._delete_by_field_list(self._id_field("chunk_id"), index_id_list)

    async def delete_by_source_id_list(
        self, ctx: Context, source_id_list: list[str], dimension: int, knowledge_type: str
    ) -> None:
        del ctx, dimension, knowledge_type
        await self._delete_by_field_list(self._id_field("source_id"), source_id_list)

    async def delete_by_knowledge_id_list(
        self, ctx: Context, knowledge_id_list: list[str], dimension: int, knowledge_type: str
    ) -> None:
        del ctx, dimension, knowledge_type
        await self._delete_by_field_list(self._id_field("knowledge_id"), knowledge_id_list)

    async def _delete_by_field_list(self, field: str, value_list: list[str]) -> None:
        if not value_list:
            return
        query = {"terms": {field: value_list}}
        await asyncio.to_thread(
            self._client.delete_by_query, index=self._index, query=query
        )

    # ── protocol: retrieve ───────────────────────────────────────────

    async def retrieve(
        self, ctx: Context, params: RetrieveParams
    ) -> list[RetrieveResult]:
        del ctx
        if params.retriever_type == RetrieverType.VECTOR:
            return await self._vector_retrieve(params)
        if params.retriever_type == RetrieverType.KEYWORDS:
            return await self._keywords_retrieve(params)
        raise ValueError(f"invalid retriever type: {params.retriever_type}")

    async def _vector_retrieve(self, params: RetrieveParams) -> list[RetrieveResult]:
        filter_clauses = build_base_conds(params, self._id_field)
        body: dict[str, Any] = {
            "query": {
                "script_score": {
                    "query": {"bool": {"filter": filter_clauses}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding')",
                        "params": {"query_vector": list(params.embedding)},
                    },
                    "min_score": params.threshold,
                }
            },
            "size": params.top_k,
            "_source": {"excludes": [_EMBEDDING_FIELD]},
        }
        response = await asyncio.to_thread(
            self._client.search, index=self._index, body=body
        )
        results = parse_search_hits(cast(dict[str, Any], response), MatchType.EMBEDDING)
        return [RetrieveResult(
            results=results,
            retriever_engine_type=RetrieverEngineType.ELASTICSEARCH,
            retriever_type=RetrieverType.VECTOR,
        )]

    async def _keywords_retrieve(self, params: RetrieveParams) -> list[RetrieveResult]:
        filter_clauses = build_base_conds(params, self._id_field)
        body: dict[str, Any] = {
            "query": {
                "bool": {
                    "must": [{"match": {"content": params.query}}],
                    "filter": filter_clauses,
                }
            },
            "size": params.top_k,
            "_source": {"excludes": [_EMBEDDING_FIELD]},
        }
        response = await asyncio.to_thread(
            self._client.search, index=self._index, body=body
        )
        results = parse_search_hits(cast(dict[str, Any], response), MatchType.KEYWORDS)
        return [RetrieveResult(
            results=results,
            retriever_engine_type=RetrieverEngineType.ELASTICSEARCH,
            retriever_type=RetrieverType.KEYWORDS,
        )]

    # ── protocol: copy_indices ───────────────────────────────────────

    async def copy_indices(
        self,
        ctx: Context,
        source_knowledge_base_id: str,
        source_to_target_kb_id_map: Mapping[str, str],
        source_to_target_chunk_id_map: Mapping[str, str],
        target_knowledge_base_id: str,
        dimension: int,
        knowledge_type: str,
    ) -> None:
        del dimension, knowledge_type
        if not source_to_target_chunk_id_map:
            return
        retrieve_params = RetrieveParams(knowledge_base_ids=[source_knowledge_base_id])
        from_val = 0
        while True:
            hits = await self._query_source_batch(retrieve_params, from_val, _COPY_BATCH_SIZE)
            if not hits:
                break
            index_info_list, embedding_map = _process_source_batch(
                hits, source_to_target_kb_id_map, source_to_target_chunk_id_map,
                target_knowledge_base_id,
            )
            if index_info_list:
                params: IndexSaveParams = {}
                if embedding_map:
                    params["embedding"] = embedding_map
                await self.batch_save(ctx, index_info_list, params)
            from_val += len(hits)
            if len(hits) < _COPY_BATCH_SIZE:
                break

    async def _query_source_batch(
        self, retrieve_params: RetrieveParams, from_val: int, batch_size: int
    ) -> list[dict[str, Any]]:
        filter_clauses = build_base_conds(retrieve_params, self._id_field)
        query = filter_clauses[0] if filter_clauses else {}
        body = {"query": query, "from": from_val, "size": batch_size}
        response = await asyncio.to_thread(
            self._client.search, index=self._index, body=body
        )
        hits_obj = cast(dict[str, Any], response).get("hits", {})
        hits_value = hits_obj.get("hits", [])
        return cast(list[dict[str, Any]], hits_value)

    # ── protocol: batch_update_chunk_* ───────────────────────────────

    async def batch_update_chunk_enabled_status(
        self, ctx: Context, chunk_status_map: Mapping[str, bool]
    ) -> None:
        del ctx
        if not chunk_status_map:
            return
        enabled_ids = [k for k, v in chunk_status_map.items() if v]
        disabled_ids = [k for k, v in chunk_status_map.items() if not v]
        if enabled_ids:
            await self._update_by_query(
                self._id_field("chunk_id"), enabled_ids,
                "ctx._source.is_enabled = true",
            )
        if disabled_ids:
            await self._update_by_query(
                self._id_field("chunk_id"), disabled_ids,
                "ctx._source.is_enabled = false",
            )

    async def batch_update_chunk_tag_id(
        self, ctx: Context, chunk_tag_map: Mapping[str, str]
    ) -> None:
        del ctx
        if not chunk_tag_map:
            return
        tag_groups: dict[str, list[str]] = {}
        for chunk_id, tag_id in chunk_tag_map.items():
            tag_groups.setdefault(tag_id, []).append(chunk_id)
        for tag_id, chunk_ids in tag_groups.items():
            await self._update_by_query(
                self._id_field("chunk_id"), chunk_ids,
                "ctx._source.tag_id = params.tag_id",
                {"tag_id": tag_id},
            )

    async def _update_by_query(
        self,
        field: str,
        values: list[str],
        script_source: str,
        params: dict[str, Any] | None = None,
    ) -> None:
        query = {"terms": {field: values}}
        script: dict[str, Any] = {"source": script_source, "lang": "painless"}
        if params:
            script["params"] = params
        await asyncio.to_thread(
            self._client.update_by_query,
            index=self._index, query=query, script=script,
        )


async def new_elasticsearch_v8_repository(
    addr: str,
    username: str,
    password: str,
    cfg: AppConfig,
    index_config: IndexConfig,
) -> ElasticsearchV8Repository:
    """Construct an Elasticsearch v8 repository from connection params."""
    del cfg
    client = new_elasticsearch_v8_client(addr, username, password)
    return ElasticsearchV8Repository(client, index_config)


__all__ = [
    "ElasticsearchV8Repository",
    "new_elasticsearch_v8_client",
    "new_elasticsearch_v8_repository",
]
