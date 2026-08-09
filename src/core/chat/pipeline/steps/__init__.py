"""Pipeline step implementations.

Each module ports one pipeline stage onto the merged ``Plugin`` protocol
from ``engine.py``. Steps consume the shared ``PipelineContext`` carrier
(``context.py``) and are wired into an ``EventManager`` at the
composition root; the step modules stay standalone so a run can be
assembled from whichever stages the request mode selects.
"""

from __future__ import annotations

from src.core.chat.pipeline.steps.query_understand import (
    QueryUnderstandPlugin,
    parse_structured_query_output,
)
from src.core.chat.pipeline.steps.search_entity import SearchEntityPlugin
from src.core.chat.pipeline.steps.search_parallel import SearchParallelPlugin
from src.core.chat.pipeline.steps.extract_entity import ExtractEntityStep
from src.core.chat.pipeline.steps.search import SearchStep

__all__ = [
    "ExtractEntityStep",
    "QueryUnderstandPlugin",
    "SearchEntityPlugin",
    "SearchParallelPlugin",
    "SearchStep",
    "parse_structured_query_output",
]
