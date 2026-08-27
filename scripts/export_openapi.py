#!/usr/bin/env python3
"""Export the FastAPI OpenAPI schema to a JSON file for frontend codegen.

Imports the app without running the lifespan, so no database or Redis
connection is required. The output feeds ``openapi-typescript`` in
``frontend/`` (see the Makefile ``openapi`` target).

Usage::

    python scripts/export_openapi.py [--output PATH]

Exit codes:
    0 = schema written
    1 = export failed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="docs/api/openapi.json",
        help="Destination JSON path (default: docs/api/openapi.json)",
    )
    args = parser.parse_args()

    from src.app_context.lifespan import app

    schema = app.openapi()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(schema, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    paths = len(schema.get("paths", {}))
    schemas = len(schema.get("components", {}).get("schemas", {}))
    print(f"wrote {output} ({paths} paths, {schemas} schemas)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
