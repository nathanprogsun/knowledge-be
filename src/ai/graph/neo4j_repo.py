"""Neo4j-backed graph repository.

Implements the ``RetrieveGraphRepository`` protocol over the Neo4j
async driver. The repository is environment-gated: when ``NEO4J_ENABLE``
is not ``true`` the built repository is a *disabled* no-op — writes and
deletes are skipped and ``search_node`` returns ``None``.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncManagedTransaction

from src.ai.graph.types import (
    GraphData,
    GraphNode,
    GraphRelation,
    NameSpace,
    RetrieveGraphRepository,
)
from src.app_logging import logger
from src.common.exception import ExternalServiceError

# Prefix applied to every namespace label so entity nodes never collide
# with unrelated labels in the shared graph store.
_NODE_PREFIX = "ENTITY"

_ENABLE_ENV = "NEO4J_ENABLE"
_URI_ENV = "NEO4J_URI"
_USERNAME_ENV = "NEO4J_USERNAME"
_PASSWORD_ENV = "NEO4J_PASSWORD"

# Connect/verify retry budget.
_MAX_CONNECT_RETRIES = 30
_RETRY_INTERVAL_SECONDS = 2.0

_NODE_IMPORT_QUERY = """
UNWIND $data AS row
CALL apoc.merge.node(row.labels, {name: row.name, kg: row.knowledge_id}, row.props, {}) YIELD node
SET node.chunks = apoc.coll.union(node.chunks, row.chunks)
RETURN distinct 'done' AS result
"""

_REL_IMPORT_QUERY = """
UNWIND $data AS row
CALL apoc.merge.node(row.source_labels, {name: row.source, kg: row.knowledge_id}, {}, {}) YIELD node as source
CALL apoc.merge.node(row.target_labels, {name: row.target, kg: row.knowledge_id}, {}, {}) YIELD node as target
CALL apoc.merge.relationship(source, row.type, {}, row.attributes, target) YIELD rel
RETURN distinct 'done'
"""


def graph_enabled() -> bool:
    """Return whether the graph repository is enabled via ``NEO4J_ENABLE``."""
    return os.getenv(_ENABLE_ENV, "").strip().lower() == "true"


def remove_hyphen(value: str) -> str:
    """Replace ``-`` with ``_`` — hyphens are not valid in Neo4j labels."""
    return value.replace("-", "_")


def _to_strings(values: Any) -> list[str]:
    """Coerce an arbitrary property sequence into strings."""
    return [str(item) for item in (values or [])]


def _delete_rels_query(label_expr: str) -> str:
    """Cypher deleting every relationship among the scoped nodes."""
    match_expr = (
        f"MATCH (n:{label_expr} {{kg: $knowledge_id}})-[r]-"
        f"(m:{label_expr} {{kg: $knowledge_id}}) RETURN r"
    )
    return (
        "CALL apoc.periodic.iterate("
        f'"{match_expr}",'
        '"DELETE r",'
        "{batchSize: 1000, parallel: true, params: {knowledge_id: $knowledge_id}}"
        ") YIELD batches, total RETURN total"
    )


def _delete_nodes_query(label_expr: str) -> str:
    """Cypher deleting every node scoped to the namespace."""
    match_expr = f"MATCH (n:{label_expr} {{kg: $knowledge_id}}) RETURN n"
    return (
        "CALL apoc.periodic.iterate("
        f'"{match_expr}",'
        '"DELETE n",'
        "{batchSize: 1000, parallel: true, params: {knowledge_id: $knowledge_id}}"
        ") YIELD batches, total RETURN total"
    )


def _search_node_query(label_expr: str) -> str:
    """Cypher matching nodes (and their neighbours) by substring."""
    return (
        f"MATCH (n:{label_expr})-[r]-(m:{label_expr}) "
        "WHERE ANY(nodeText IN $nodes WHERE n.name CONTAINS nodeText) "
        "RETURN n, r, m"
    )


class Neo4jConnectionError(ExternalServiceError):
    """Raised when the Neo4j driver cannot be created or verified."""


async def connect_driver(
    uri: str,
    username: str,
    password: str,
    *,
    max_retries: int = _MAX_CONNECT_RETRIES,
    retry_interval: float = _RETRY_INTERVAL_SECONDS,
) -> AsyncDriver:
    """Create and verify an async Neo4j driver, retrying on failure.

    Retries ``max_retries`` times with a ``retry_interval`` pause between
    attempts, then raises ``Neo4jConnectionError`` when every attempt has
    failed.
    """
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        driver: AsyncDriver | None = None
        try:
            driver = AsyncGraphDatabase.driver(uri, auth=(username, password))
            await driver.verify_authentication()
        except Exception as exc:
            last_error = exc
            if driver is not None:
                await driver.close()
            if attempt < max_retries:
                logger.warning("neo4j connect attempt {}/{} failed: {}", attempt, max_retries, exc)
                await asyncio.sleep(retry_interval)
            continue
        return driver
    raise Neo4jConnectionError(
        f"failed to connect to Neo4j after {max_retries} attempts: {last_error}"
    )


async def build_graph_repository() -> Neo4jRepository:
    """Build the graph repository from the environment.

    Returns a no-op repository when ``NEO4J_ENABLE`` is not ``true``.
    Otherwise connects with retry (raising ``Neo4jConnectionError`` on
    failure) and wraps the verified driver.
    """
    if not graph_enabled():
        return Neo4jRepository(driver=None)
    driver = await connect_driver(
        uri=os.getenv(_URI_ENV, ""),
        username=os.getenv(_USERNAME_ENV, ""),
        password=os.getenv(_PASSWORD_ENV, ""),
    )
    logger.info("neo4j graph repository connected")
    return Neo4jRepository(driver=driver)


class Neo4jRepository:
    """Async adapter over the Neo4j async driver.

    ``driver`` may be ``None`` for a deployment without the graph store;
    every method then no-ops / returns ``None``.
    """

    def __init__(self, driver: AsyncDriver | None, *, node_prefix: str = _NODE_PREFIX) -> None:
        self._driver = driver
        self._node_prefix = node_prefix

    def labels(self, namespace: NameSpace) -> list[str]:
        """Namespace labels, prefixed and with hyphens normalized."""
        return [self._node_prefix + remove_hyphen(label) for label in namespace.labels()]

    def _label_expr(self, namespace: NameSpace) -> str:
        return ":".join(self.labels(namespace))

    async def close(self) -> None:
        """Close the underlying driver (no-op when disabled)."""
        if self._driver is not None:
            await self._driver.close()

    async def add_graph(self, namespace: NameSpace, graphs: list[GraphData]) -> None:
        """Persist every graph under ``namespace``."""
        if self._driver is None:
            logger.warning("graph repository disabled — skipping add_graph")
            return
        for graph in graphs:
            await self._add_graph(self._driver, namespace, graph)

    async def _add_graph(self, driver: AsyncDriver, namespace: NameSpace, graph: GraphData) -> None:
        session = driver.session()
        try:
            await session.execute_write(self._write_graph, namespace, graph)
        finally:
            await session.close()

    async def _write_graph(
        self, tx: AsyncManagedTransaction, namespace: NameSpace, graph: GraphData
    ) -> None:
        labels = self.labels(namespace)
        node_data = [
            {
                "name": node.name,
                "knowledge_id": namespace.knowledge,
                "props": {"attributes": node.attributes},
                "chunks": node.chunks,
                "labels": labels,
            }
            for node in graph.node
        ]
        node_result = await tx.run(_NODE_IMPORT_QUERY, {"data": node_data})
        await node_result.consume()

        rel_data = [
            {
                "source": rel.node1,
                "target": rel.node2,
                "knowledge_id": namespace.knowledge,
                "type": rel.type,
                "source_labels": labels,
                "target_labels": labels,
            }
            for rel in graph.relation
        ]
        rel_result = await tx.run(_REL_IMPORT_QUERY, {"data": rel_data})
        await rel_result.consume()

    async def del_graph(self, namespaces: list[NameSpace]) -> None:
        """Delete the graphs under every given namespace."""
        if self._driver is None:
            logger.warning("graph repository disabled — skipping del_graph")
            return
        session = self._driver.session()
        try:
            await session.execute_write(self._delete_graphs, namespaces)
        finally:
            await session.close()

    async def _delete_graphs(
        self, tx: AsyncManagedTransaction, namespaces: list[NameSpace]
    ) -> None:
        for namespace in namespaces:
            label_expr = self._label_expr(namespace)
            params = {"knowledge_id": namespace.knowledge}
            rels_result = await tx.run(_delete_rels_query(label_expr), params)
            await rels_result.consume()
            nodes_result = await tx.run(_delete_nodes_query(label_expr), params)
            await nodes_result.consume()

    async def search_node(self, namespace: NameSpace, nodes: list[str]) -> GraphData | None:
        """Return the subgraph matching any of ``nodes``, or ``None`` when disabled."""
        if self._driver is None:
            logger.warning("graph repository disabled — search_node returns None")
            return None
        session = self._driver.session()
        try:
            return await session.execute_read(self._search_nodes, namespace, nodes)
        finally:
            await session.close()

    async def _search_nodes(
        self, tx: AsyncManagedTransaction, namespace: NameSpace, nodes: list[str]
    ) -> GraphData:
        label_expr = self._label_expr(namespace)
        result = await tx.run(_search_node_query(label_expr), {"nodes": nodes})
        graph = GraphData()
        node_seen: set[str] = set()
        async for record in result:
            node = record["n"]
            rel = record["r"]
            target = record["m"]
            for current in (node, target):
                name = str(current["name"])
                if name not in node_seen:
                    node_seen.add(name)
                    graph.node.append(
                        GraphNode(
                            name=name,
                            chunks=_to_strings(current.get("chunks")),
                            attributes=_to_strings(current.get("attributes")),
                        )
                    )
            graph.relation.append(
                GraphRelation(
                    node1=str(node["name"]),
                    node2=str(target["name"]),
                    type=rel.type,
                )
            )
        return graph


__all__ = [
    "Neo4jConnectionError",
    "Neo4jRepository",
    "RetrieveGraphRepository",
    "build_graph_repository",
    "connect_driver",
    "graph_enabled",
    "remove_hyphen",
]
