"""Pipeline step: replace web-search snippets with fetched full-page text.

Runs after rerank and before merge. For the top ``web_fetch_top_n`` web
results (those whose knowledge source is ``web_search``) it fetches the
full page content — the result id doubles as the URL — and replaces the
snippet ``content``, truncated to a bounded size for the LLM context.
Fetches run concurrently and a failure on one page never fails the run.
"""

from __future__ import annotations

import asyncio

from src.core.agents.tools.web_fetch import (
    FetchError,
    WebContentFetcher,
    WebPageFetcher,
)
from src.core.chat.pipeline.common import pipeline_info, pipeline_warn
from src.core.chat.pipeline.context import PipelineContext
from src.core.chat.pipeline.engine import Next, PluginError
from src.core.chat.pipeline.types import Context, EventType, SearchResult

#: Default page cap when the request did not configure ``web_fetch_top_n``.
_DEFAULT_TOP_N = 3
#: Maximum characters of fetched page text kept per result.
_MAX_FETCH_CONTENT = 8000


class WebFetchPlugin:
    """Pipeline step that fetches full content for reranked web results."""

    def __init__(self, fetcher: WebContentFetcher | None = None) -> None:
        self._fetcher = fetcher if fetcher is not None else WebPageFetcher()

    def activation_events(self) -> list[EventType]:
        return [EventType.WEB_FETCH]

    async def on_event(
        self,
        ctx: Context,
        event_type: EventType | str,
        pipeline_ctx: PipelineContext,
        next: Next,
    ) -> PluginError | None:
        if not pipeline_ctx.web_fetch_enabled or not pipeline_ctx.web_search_enabled:
            pipeline_info("WebFetch", "skip", {"reason": "disabled"})
            return await next()

        top_n = pipeline_ctx.web_fetch_top_n if pipeline_ctx.web_fetch_top_n > 0 else _DEFAULT_TOP_N
        web_results: list[tuple[int, SearchResult]] = []
        for index, result in enumerate(pipeline_ctx.rerank_result):
            if result.knowledge_source.lower() == "web_search":
                web_results.append((index, result))
                if len(web_results) >= top_n:
                    break

        if not web_results:
            pipeline_info("WebFetch", "skip", {"reason": "no_web_results"})
            return await next()

        fetched = await asyncio.gather(*(self._fetch_one(result) for _index, result in web_results))

        replacements: dict[int, SearchResult] = {}
        fetched_count = 0
        for (index, result), content in zip(web_results, fetched, strict=True):
            if content is None or not content:
                continue
            truncated = content
            if len(truncated) > _MAX_FETCH_CONTENT:
                truncated = truncated[:_MAX_FETCH_CONTENT] + "\n...(truncated)"
            replacements[index] = result.model_copy(update={"content": truncated})
            fetched_count += 1

        if replacements:
            rerank_result = list(pipeline_ctx.rerank_result)
            for index, updated in replacements.items():
                rerank_result[index] = updated
            pipeline_ctx.rerank_result = rerank_result

        pipeline_info(
            "WebFetch",
            "complete",
            {"fetched": fetched_count, "total": len(web_results)},
        )
        return await next()

    async def _fetch_one(self, result: SearchResult) -> str | None:
        """Fetch one page; return ``None`` (and log) on any failure."""
        fetch_url = result.id  # web search results use the URL as the id
        if not fetch_url:
            return None
        try:
            return await self._fetcher.fetch(fetch_url)
        except FetchError as exc:
            pipeline_warn(
                "WebFetch",
                "fetch_failed",
                {"url": fetch_url, "error": exc.message},
            )
            return None
        except Exception as exc:
            pipeline_warn(
                "WebFetch",
                "fetch_failed",
                {"url": fetch_url, "error": str(exc)},
            )
            return None


__all__ = ["WebFetchPlugin"]
