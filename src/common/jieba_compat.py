"""Load jieba on CPython builds that reject its regex literals.

jieba 0.42.1 ships unescaped ``\\.`` / ``\\-`` sequences. Some CPython
builds treat those as a parse-time SyntaxError, which blocks retrieval
import. The package is unmaintained, so the installed sources are
rewritten to raw strings before the import runs.
"""

from __future__ import annotations

import importlib
import importlib.util
import warnings
from pathlib import Path
from types import ModuleType


def _rewrite_compile_literals(source: str) -> str:
    return source.replace('re.compile("', 're.compile(r"').replace("re.compile('", "re.compile(r'")


def _patch_installed_jieba() -> None:
    spec = importlib.util.find_spec("jieba")
    if spec is None or spec.origin is None:
        return
    root = Path(spec.origin).parent
    for path in root.rglob("*.py"):
        original: str = path.read_text(encoding="utf-8")
        rewritten: str = _rewrite_compile_literals(original)
        if rewritten == original:
            continue
        try:
            path.write_text(rewritten, encoding="utf-8")
        except OSError:
            return


_patch_installed_jieba()
warnings.filterwarnings("ignore", category=SyntaxWarning)
jieba: ModuleType = importlib.import_module("jieba")
