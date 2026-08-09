"""Chat pipeline step plugins.

The step modules implement the pipeline ``Plugin`` protocol over the shared
``PipelineContext`` carrier:

- ``search`` — the ``CHUNK_SEARCH`` retrieval step (KB + web, with low-recall
  query expansion);
- ``extract_entity`` — the ``QUERY_UNDERSTAND`` entity-extraction step;
- ``query_expansion`` — pure local query-variant generation helpers.
"""

from __future__ import annotations

from src.core.chat.pipeline.steps.extract_entity import ExtractEntityStep
from src.core.chat.pipeline.steps.search import SearchStep

__all__ = ["ExtractEntityStep", "SearchStep"]
