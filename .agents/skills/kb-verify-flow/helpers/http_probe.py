"""GET probe used by kb-verify-flow Doctor and Drive."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Mapping


def get_json_or_text(
    url: str,
    headers: Mapping[str, str] | None = None,
    timeout: float = 3.0,
) -> dict[str, object]:
    req = urllib.request.Request(url, headers=dict(headers or {}), method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            payload: object
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                payload = body[:500]
            return {
                "ok": True,
                "url": url,
                "status": resp.status,
                "body": payload,
            }
    except urllib.error.HTTPError as exc:
        return {
            "ok": False,
            "url": url,
            "status": exc.code,
            "error": exc.reason,
        }
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {
            "ok": False,
            "url": url,
            "status": None,
            "error": str(exc),
        }
