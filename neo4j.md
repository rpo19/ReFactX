# Freebase -> Neo4j Import Guidelines

Goal:
Import the Freebase RDF dump into Neo4j using a simple property graph model, without preserving RDF semantics or ontology information.

## Graph Model

### Nodes
- Every Freebase entity (MID or GUID) becomes a node with label `Entity`.
- Use the Freebase identifier as the primary key.

Example:

(:Entity {
    id: "/m/02mjmr"
})

Additional literal values are stored as node properties.

Example:

(:Entity {
    id: "/m/02mjmr",
    name: "Barack Obama",
    date_of_birth: "1961-08-04"
})

If a property has multiple literal values, store them as arrays.

### Relationships

For triples whose object is another entity:

(subject, predicate, object)

create

(:Entity {id: subject})-[:PREDICATE]->(:Entity {id: object})

Predicate names must be sanitized to valid Neo4j relationship types.

Example mapping:

/people/person/place_of_birth
    ->
PEOPLE_PERSON_PLACE_OF_BIRTH

Suggested sanitization:
- remove leading '/'
- replace '/' with '_'
- replace '.' with '_'
- convert to upper case
- prepend 'FB_' if necessary to avoid naming conflicts

Examples:

/type/object/type
    ->
TYPE_OBJECT_TYPE

/film/film/director
    ->
FILM_FILM_DIRECTOR

### Literal Objects

If the object is a literal, do not create a node.

Instead, add/update a property on the subject node.

Example:

<BarackObama> <type.object.name> "Barack Obama"

becomes

(:Entity {
    id: "...",
    type_object_name: "Barack Obama"
})

If multiple literals exist for the same predicate, store them as arrays.

## Import Strategy

For large dumps:

1. Parse RDF sequentially.
2. Assign each entity a unique integer ID internally.
3. Produce Neo4j bulk-import CSV files.
4. Import using `neo4j-admin database import`.

Recommended CSVs:

nodes.csv
-----------
id:ID(Entity),fbid,name,...
0,/m/02mjmr,Barack Obama,...

relationships.csv
-----------------
:START_ID(Entity),:END_ID(Entity),:TYPE
0,12345,PEOPLE_PERSON_PLACE_OF_BIRTH

Integer IDs should be used for relationships because they make the bulk import significantly faster than repeatedly matching string identifiers.

## Constraints

After import:

CREATE CONSTRAINT entity_id IF NOT EXISTS
FOR (e:Entity)
REQUIRE e.fbid IS UNIQUE;

## Notes

- Ignore RDF-specific constructs (blank nodes, reification, RDF collections) unless explicitly needed.
- No ontology or schema reasoning is required.
- Preserve every entity-to-entity triple as a Neo4j relationship.
- Preserve every literal as a node property.
- Relationship types correspond directly to Freebase predicates after sanitization.
- This representation is intended for efficient graph traversal and Cypher queries rather than RDF/SPARQL compatibility.
