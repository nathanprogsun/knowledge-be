"""Default graph-extraction prompt templates.

These are the default templates the graph builder fills the ``{{language}}``
placeholder into; callers that configure their own prompt templates pass
them to the builder constructor instead. The templates mirror the upstream
default entity / relationship extraction prompts.
"""

from __future__ import annotations

DEFAULT_EXTRACT_ENTITIES_PROMPT = """## Task
Extract all entities from the user-provided text that match the following entity types:
EntityTypes: [Person, Organization, Location, Product, Event, Date, Work, Concept, Resource, Category, Operation]

## Requirements
1. Output must be in JSON array format
2. Each entity must contain title and type fields; the description field is optional but strongly recommended
3. The type field value must be strictly selected from the EntityTypes list; do not create new types
4. If the entity type cannot be determined, do not force a classification; it is better to skip that entity
5. Do not output any explanation or additional content; output only the JSON array
6. All field values must not contain HTML tags or other code
7. If an entity is ambiguous, specify the reference in the description
8. If no entities are found, return an empty array []

## Entity Extraction Rules
- Person: Real or fictional characters, including historical figures, modern figures, literary characters, etc.
- Organization: Companies, government agencies, teams, schools, and other organizational entities
- Location: Geographic locations, landmarks, countries, cities, etc.
- Product: Goods, services, brands, and other commercial products
- Event: Events, conferences, festivals, historical events, etc.
- Date: Dates, time periods, eras, and other time-related information
- Work: Books, movies, music, artworks, and other creative works
- Concept: Abstract concepts, ideas, theories, etc.
- Resource: Natural resources, information resources, tools, etc.
- Category: Classifications, categories, fields, etc.
- Operation: Operations, actions, methods, processes, etc.

## Extraction Steps
1. Carefully read the text and identify potential entities
2. For each identified entity, determine the most appropriate entity type (must be selected from EntityTypes)
3. Create a JSON object for each entity with the following fields:
   - title: The standard name of the entity, without modifiers such as quotation marks
   - type: The entity type selected from EntityTypes
   - description: A brief description of the entity, based on the text content, in the same language as the source text
4. Verify that all fields of each entity are correct and properly formatted
5. Merge all entity objects into a single JSON array
6. Check that the final JSON is valid and meets requirements

## CRITICAL: Language Rule
- Extract entity titles exactly as they appear in the source text
- Write descriptions in {{language}}"""

DEFAULT_EXTRACT_RELATIONSHIPS_PROMPT = """## Task
From the user-provided entity array, extract explicit relationships between entities to form a structured relationship network.

## Requirements
1. Relationship extraction must be based on the provided text content; do not fabricate non-existent relationships
2. Output must be in JSON array format, with each relationship as an object in the array
3. Each relationship object must contain source, target, description, and strength fields
4. Do not output any explanation or additional content; output only the JSON array
5. If no relationships are found, return an empty array []

## Relationship Extraction Rules
- Only relationships explicitly present in the text should be extracted
- Source entity and target entity must be entities already in the entity array
- Relationship description should concisely explain the specific relationship between the two entities
- Relationship strength should be determined based on the following criteria:
  * 10: Direct creation/subordination relationship (e.g., author and work, inventor and invention, parent company and subsidiary)
  * 9: Different manifestations of the same entity (e.g., alias, former name)
  * 8: Closely related and mutually influential relationships (e.g., close partners, family members)
  * 7: Clear but indirect relationships (e.g., characters in a work, members of an organization)
  * 6: Indirect association with clear connection (e.g., colleague relationship, similar products)
  * 5: Related but loosely connected (e.g., different concepts in the same field)

## Extraction Steps
1. Carefully analyze the text content to determine which entities have explicit relationships
2. Only consider relationships explicitly mentioned in the text; do not fabricate
3. For each relationship found, determine:
   - source: The title of the source entity (must be an entity already in the entity list)
   - target: The title of the target entity (must be an entity already in the entity list)
   - description: A concise and accurate relationship description
   - strength: Relationship strength based on the above criteria (integer between 5-10)
4. Check whether each relationship is bidirectional:
   - If the relationship is bidirectional (e.g., "A is B's friend" implies "B is also A's friend"), consider whether a reverse relationship should be created
   - If the relationship is unidirectional (e.g., "A created B"), keep only the unidirectional relationship
5. Verify the consistency and reasonableness of all relationships:
   - Ensure there are no contradictory relationships (e.g., A is simultaneously B's father and brother)
   - Ensure relationship descriptions match relationship strengths
6. Organize all valid relationships into a JSON array

## CRITICAL: Language Rule
- Write relationship descriptions in {{language}}"""

__all__ = [
    "DEFAULT_EXTRACT_ENTITIES_PROMPT",
    "DEFAULT_EXTRACT_RELATIONSHIPS_PROMPT",
]
