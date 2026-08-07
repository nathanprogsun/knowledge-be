"""Conftest for the ``tests.unit.util`` package.

Importing :mod:`tests.util.factories` runs the ``@register_fixture``
decorators at collection time. The decorators store the resulting pytest
fixture functions in the factories module's namespace; pytest only
discovers fixtures from conftest modules or registered plugins, so we
register the module as a session plugin to expose the factory fixtures
(``user_factory``, ``tenant_factory``, …) globally.
"""

from __future__ import annotations

import pytest

from tests.util import factories

_PLUGIN_NAME = "tests.util.factories"


def pytest_collectstart(collector: pytest.Collector) -> None:
    """Register the factories module as a pytest plugin on first collection.

    Runs once per session - the ``hasplugin`` guard prevents duplicate
    registration. Registering the module (rather than relying on plain
    import) makes the fixture functions discoverable by pytest's
    FixtureManager for every test in the session.
    """
    pm = collector.config.pluginmanager
    if not pm.hasplugin(_PLUGIN_NAME):
        pm.register(factories, name=_PLUGIN_NAME)
