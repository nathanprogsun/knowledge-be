"""Auxiliary ``ai`` layer - external service clients.

Holds HTTP/gRPC clients that call out to external providers (OIDC IdPs,
AI backends, vector stores, storage). Per AGENTS.md §1 the ``ai`` layer
may import only ``common`` and ``util`` - never ``core`` or ``db``.
"""
