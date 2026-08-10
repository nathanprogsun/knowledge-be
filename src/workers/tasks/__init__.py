"""Background-task handlers for the worker layer.

Each module here registers a single ARQ task that maps an upstream
asynq task name. Importing a module is what wires its handler into
:func:`src.workers.registry.all_functions`; :mod:`src.workers.main`
imports the package so every task lands on the running worker.
"""
