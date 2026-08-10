"""ARQ worker task handlers.

Each task module in this package implements a single task type from
the upstream async queue. Importing a task module is what registers
the handler with the worker registry (see :mod:`src.workers.registry`).

The :mod:`src.workers.main` entry point imports every task module for
its side effect so that the default worker serves the full set.
"""
