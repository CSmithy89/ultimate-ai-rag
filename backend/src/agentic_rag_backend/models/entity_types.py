"""Custom entity types for Graphiti temporal knowledge graph.

Defines domain-specific entity types that extend Graphiti's EntityModel
for technical documentation and code-related knowledge extraction.
"""

from typing import Optional, TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from graphiti_core.nodes import EntityNode as EntityNodeBase
else:
    try:
        from graphiti_core.nodes import EntityNode as EntityNodeBase
    except ImportError:
        # Fallback for when graphiti-core is not installed
        class EntityNodeBase(BaseModel):
            """Fallback EntityNode when graphiti-core is not available."""

            name: str
            uuid: str = ""
            group_id: str = ""
            created_at: Optional[str] = None
            summary: str = ""

EntityNode = EntityNodeBase


class TechnicalConcept(EntityNode):
    """Technical concept extracted from documentation.

    Represents frameworks, patterns, algorithms, or other technical
    knowledge that appears in documentation.
    """

    domain: str = Field(
        default="general",
        description="Technical domain (e.g., 'frontend', 'backend', 'database', 'devops')",
    )
    complexity: str = Field(
        default="intermediate",
        description="Complexity level ('beginner', 'intermediate', 'advanced', 'expert')",
    )
    category: Optional[str] = Field(
        default=None,
        description="Category within the domain (e.g., 'authentication', 'caching')",
    )


class CodePattern(EntityNode):
    """Code pattern or programming construct.

    Represents reusable code patterns, design patterns, or
    implementation approaches found in documentation or code.
    """

    language: str = Field(
        default="python",
        description="Programming language (e.g., 'python', 'typescript', 'sql')",
    )
    pattern_type: str = Field(
        default="implementation",
        description="Pattern type ('design', 'implementation', 'architectural', 'idiom')",
    )
    use_case: Optional[str] = Field(
        default=None,
        description="Primary use case for this pattern",
    )


class APIEndpoint(EntityNode):
    """API endpoint definition.

    Represents REST API endpoints, GraphQL operations, or other
    API interface definitions.
    """

    method: str = Field(
        default="GET",
        description="HTTP method ('GET', 'POST', 'PUT', 'DELETE', 'PATCH')",
    )
    path: str = Field(
        default="/",
        description="URL path template (e.g., '/api/v1/users/{id}')",
    )
    version: Optional[str] = Field(
        default=None,
        description="API version (e.g., 'v1', 'v2')",
    )
    auth_required: bool = Field(
        default=True,
        description="Whether authentication is required",
    )


class ConfigurationOption(EntityNode):
    """Configuration option or setting.

    Represents environment variables, config file options,
    or other configurable parameters.
    """

    config_type: str = Field(
        default="environment",
        description="Configuration type ('environment', 'file', 'runtime', 'build')",
    )
    default_value: Optional[str] = Field(
        default=None,
        description="Default value if not specified",
    )
    required: bool = Field(
        default=False,
        description="Whether this configuration is required",
    )
    sensitive: bool = Field(
        default=False,
        description="Whether this contains sensitive data (secrets, keys)",
    )


# Entity type registry for Graphiti configuration
ENTITY_TYPES: list[type] = [
    TechnicalConcept,
    CodePattern,
    APIEndpoint,
    ConfigurationOption,
]

# Edge type mappings define relationship types between entity pairs
EDGE_TYPE_MAPPINGS: dict[tuple[str, str], list[str]] = {
    # TechnicalConcept relationships
    ("TechnicalConcept", "TechnicalConcept"): [
        "DEPENDS_ON",
        "RELATED_TO",
        "EXTENDS",
        "IMPLEMENTS",
        "ALTERNATIVE_TO",
    ],
    ("TechnicalConcept", "CodePattern"): [
        "IMPLEMENTED_BY",
        "USES_PATTERN",
        "DEMONSTRATED_IN",
    ],
    # CodePattern relationships
    ("CodePattern", "CodePattern"): [
        "COMPOSES_WITH",
        "VARIANT_OF",
        "REPLACES",
        "CONFLICTS_WITH",
    ],
    ("CodePattern", "TechnicalConcept"): [
        "IMPLEMENTS",
        "APPLIES_TO",
        "DEMONSTRATES",
    ],
    # APIEndpoint relationships
    ("APIEndpoint", "TechnicalConcept"): [
        "EXPOSES",
        "REQUIRES",
        "RETURNS",
    ],
    ("APIEndpoint", "APIEndpoint"): [
        "CALLS",
        "REDIRECTS_TO",
        "SUPERSEDES",
    ],
    # ConfigurationOption relationships
    ("ConfigurationOption", "TechnicalConcept"): [
        "CONFIGURES",
        "ENABLES",
        "CONTROLS",
    ],
    ("ConfigurationOption", "ConfigurationOption"): [
        "DEPENDS_ON",
        "OVERRIDES",
        "CONFLICTS_WITH",
    ],
    ("ConfigurationOption", "APIEndpoint"): [
        "CONFIGURES",
        "ENABLES",
    ],
}


def get_edge_types(source_type: str, target_type: str) -> list[str]:
    """Get valid edge types for a source-target entity pair.

    Args:
        source_type: Name of the source entity type
        target_type: Name of the target entity type

    Returns:
        List of valid edge type names, or default types if not mapped
    """
    key = (source_type, target_type)
    if key in EDGE_TYPE_MAPPINGS:
        return EDGE_TYPE_MAPPINGS[key]

    # Default edge types for unmapped pairs
    return ["RELATED_TO", "REFERENCES"]
