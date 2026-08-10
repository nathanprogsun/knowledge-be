"""Background worker layer (ARQ).

Workers consume tasks from a Redis-backed queue and dispatch them to
handlers registered in :mod:`src.workers.registry`. The layer depends
only on ``core`` / ``common`` / ``util`` — never on ``db`` or ``ai``
directly.
"""
