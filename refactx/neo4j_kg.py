"""Neo4j-backed knowledge-graph connectors for ReFactX.

This module mirrors the three-level, rule-backed API of the synthetic
``ReFactX-with-Rules/api.py`` demo, but reads from a Neo4j property graph
instead of an in-process reasoner:

    level 1  ``entity_index()``        -- which entities exist (index the LLM may generate)
    level 2  ``property_strings(e)``   -- which relations can answer for ``e``
    level 3  ``get(e, rel)``           -- the objects that answer ``rel`` for ``e``

``property_strings`` and ``get`` are the exact callbacks expected by
``refactx.generate.KnowledgeGraphGeneration`` (``get_relations`` and
``get_objects``).

Graph model (see ``neo4j.md``)::

    (:Entity {id: '/m/02mjmr', name: 'Barack Obama'})
    (:Entity)-[:PEOPLE_PERSON_PLACE_OF_BIRTH]->(:Entity)

i.e. Freebase predicates are sanitized (``/people/person/place_of_birth`` ->
``PEOPLE_PERSON_PLACE_OF_BIRTH``) and used directly as Neo4j relationship
types.  Literal values are not stored as separate nodes here.

The module exposes both a :class:`Neo4jKnowledgeGraph` class and module-level
convenience functions that share a lazily-created default connection, so a
notebook can simply do::

    from refactx.neo4j_kg import entity_index, property_strings, get
"""
from __future__ import annotations

import os
import re
from typing import Dict, Iterable, List, Optional, Sequence

from neo4j import GraphDatabase

# ---------------------------------------------------------------------------
# Connection defaults (overridable through environment variables)
# ---------------------------------------------------------------------------

DEFAULT_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
DEFAULT_USER = os.getenv("NEO4J_USER", "neo4j")
DEFAULT_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
# None lets the driver use the server's default database.
DEFAULT_DATABASE = os.getenv("NEO4J_DATABASE") or None

# Relationship types that carry no useful factual content (Freebase
# meta/authority/common-topic relations).  They are filtered out so the
# generated paths stay meaningful.
DEFAULT_SKIP_TYPES = {
    "TYPE_OBJECT_TYPE", "COMMON_TOPIC_NOTABLE_FOR", "COMMON_TOPIC_NOTABLE_TYPES",
    "COMMON_TOPIC_ALIAS", "COMMON_TOPIC_DESCRIPTION", "TYPE_OBJECT_NAME",
    "FREEBASE_VALUENOTATION_HAS_VALUE", "FREEBASE_VALUENOTATION_IS_REVIEWED",
    "TYPE_CONTENT_MEDIA_TYPE",
    "COMMON_IMAGE", "COMMON_RESOURCE", "COMMON_TOPIC_TOPIC_EQUIVALENT_WEBPAGE",
    "KG_OBJECT_PROFILE_PROMINENT_TYPE", "BASE_WORDNET_WORD_WORD",
    "BUSINESS_CIK", "SOFT_ISBN", "BOOK_BOOK_EDITION_ISBN",
}

MAX_NAME_LEN = 80

# Relationship-type prefixes that are Freebase meta/schema noise.
DEFAULT_SKIP_PREFIXES = ("TYPE_", "COMMON_", "FREEBASE_", "AUTHORITY_",
                         "USER_", "META_", "BASE_")

# Reject names that are clearly identifiers rather than human-readable names.
_BAD_NAME_RE = re.compile(r"^\s*$|^[0-9]|^/|^g\.|:")
_HEX_ID_RE = re.compile(r"^[0-9a-fA-F-]{20,}$")


def is_good_name(name: Optional[str]) -> bool:
    """Whether ``name`` is a usable entity name for the prefix index."""
    if not name or not isinstance(name, str):
        return False
    if len(name) > MAX_NAME_LEN:
        return False
    if _BAD_NAME_RE.search(name):
        return False
    if _HEX_ID_RE.match(name):
        return False
    return True


class Neo4jKnowledgeGraph:
    """Thin Cypher connector exposing the three interaction levels.

    All arguments of the module-level functions are forwarded here.
    """

    def __init__(self, uri: str = DEFAULT_URI, user: str = DEFAULT_USER,
                 password: str = DEFAULT_PASSWORD, database: Optional[str] = DEFAULT_DATABASE,
                 skip_types: Optional[Iterable[str]] = None,
                 skip_prefixes: Optional[Iterable[str]] = None, verify: bool = True):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.skip_types = sorted(set(skip_types) if skip_types is not None else DEFAULT_SKIP_TYPES)
        self.skip_prefixes = tuple(DEFAULT_SKIP_PREFIXES if skip_prefixes is None else skip_prefixes)
        if verify:
            self.driver.verify_connectivity()

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        self.driver.close()

    def __enter__(self) -> "Neo4jKnowledgeGraph":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _run(self, query: str, **params):
        with self.driver.session(database=self.database) as session:
            return list(session.run(query, **params))

    # -- level 1: which entities exist ------------------------------------
    def entity_index(self, limit: int = 20000, seeds: Optional[Sequence[str]] = None,
                     hops: int = 1, max_relations_per_node: int = 50) -> List[str]:
        """Entity names to load into the prefix index.

        If ``seeds`` is given, the names are collected by expanding the
        neighbourhood of those entities up to ``hops`` hops (breadth-first),
        which keeps the index coherent and the queries cheap.  Otherwise a
        simple scan of entity nodes is used as a fallback.
        """
        names: Dict[str, None] = {}
        if seeds:
            frontier = [s for s in seeds if s]
            for s in frontier:
                names.setdefault(s, None)
            for _ in range(max(0, hops)):
                next_frontier: List[str] = []
                for name in frontier:
                    for neighbour in self._neighbour_names(name, max_relations_per_node):
                        if neighbour not in names:
                            names[neighbour] = None
                            next_frontier.append(neighbour)
                        if len(names) >= limit:
                            break
                    if len(names) >= limit:
                        break
                frontier = next_frontier
                if not frontier or len(names) >= limit:
                    break
        else:
            rows = self._run(
                "MATCH (n:Entity) "
                "WHERE n.name IS NOT NULL AND n.name <> '' "
                "RETURN n.name AS name LIMIT $limit",
                limit=limit,
            )
            for row in rows:
                names.setdefault(row["name"], None)

        return sorted(name for name in names if is_good_name(name))[:limit]

    def _keep_relation(self, relation: str) -> bool:
        return not relation.startswith(self.skip_prefixes)

    def _neighbour_names(self, name: str, limit: int) -> List[str]:
        rows = self._run(
            "MATCH (n:Entity {name: $name})-[r]->(m:Entity) "
            "WHERE m.name IS NOT NULL AND m.name <> '' "
            "AND NOT type(r) IN $skip "
            "RETURN DISTINCT type(r) AS relation, m.name AS name",
            name=name, skip=self.skip_types,
        )
        names = []
        for row in rows:
            if self._keep_relation(row["relation"]):
                names.append(row["name"])
            if len(names) >= limit:
                break
        return names

    # -- level 2: what can answer for an entity ---------------------------
    def property_strings(self, entity: str, limit: Optional[int] = None) -> List[str]:
        """Sorted relation types with at least one object for ``entity``."""
        rows = self._run(
            "MATCH (n:Entity {name: $name})-[r]->(m:Entity) "
            "WHERE m.name IS NOT NULL AND m.name <> '' "
            "AND NOT type(r) IN $skip "
            "RETURN DISTINCT type(r) AS relation ORDER BY relation",
            name=entity, skip=self.skip_types,
        )
        relations = [row["relation"] for row in rows if self._keep_relation(row["relation"])]
        return relations[:limit] if limit else relations

    # -- level 3: run the relation ----------------------------------------
    def get(self, subject: str, relation: str, limit: Optional[int] = None) -> List[str]:
        """Distinct object names for ``(subject)-[:relation]->(...)``."""
        query = (
            "MATCH (n:Entity {name: $name})-[r]->(m:Entity) "
            "WHERE type(r) = $relation "
            "AND m.name IS NOT NULL AND m.name <> '' "
            "RETURN DISTINCT m.name AS name"
        )
        if limit:
            query += " LIMIT $limit"
            rows = self._run(query, name=subject, relation=relation, limit=limit)
        else:
            rows = self._run(query, name=subject, relation=relation)
        return [row["name"] for row in rows]

    def count(self, subject: str, relation: str) -> int:
        """Number of distinct objects answering ``relation`` for ``subject``."""
        return len(self.get(subject, relation))

    def ask(self, subject: str, relation: str, obj: str) -> bool:
        """Whether the single fact ``(subject, relation, obj)`` holds."""
        rows = self._run(
            "MATCH (n:Entity {name: $name})-[r]->(m:Entity {name: $obj}) "
            "WHERE type(r) = $relation "
            "RETURN 1 AS ok LIMIT 1",
            name=subject, relation=relation, obj=obj,
        )
        return len(rows) > 0


# ---------------------------------------------------------------------------
# Module-level convenience API with a shared lazy connection
# ---------------------------------------------------------------------------

_DEFAULT_KG: Optional[Neo4jKnowledgeGraph] = None


def get_kg() -> Neo4jKnowledgeGraph:
    """Return (creating if needed) the shared default :class:`Neo4jKnowledgeGraph`."""
    global _DEFAULT_KG
    if _DEFAULT_KG is None:
        _DEFAULT_KG = Neo4jKnowledgeGraph()
    return _DEFAULT_KG


def close_kg() -> None:
    """Close the shared default connection (if one was created)."""
    global _DEFAULT_KG
    if _DEFAULT_KG is not None:
        _DEFAULT_KG.close()
        _DEFAULT_KG = None


def entity_index(*args, **kwargs) -> List[str]:
    return get_kg().entity_index(*args, **kwargs)


def property_strings(*args, **kwargs) -> List[str]:
    return get_kg().property_strings(*args, **kwargs)


def get(*args, **kwargs) -> List[str]:
    return get_kg().get(*args, **kwargs)


def count(*args, **kwargs) -> int:
    return get_kg().count(*args, **kwargs)


def ask(*args, **kwargs) -> bool:
    return get_kg().ask(*args, **kwargs)
