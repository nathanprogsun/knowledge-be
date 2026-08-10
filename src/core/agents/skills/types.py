"""Skill domain types, validation, and ``SKILL.md`` parsing.

Implements the progressive-disclosure skill model: Level 1 metadata
(name/description), Level 2 instructions (the ``SKILL.md`` body), and
Level 3 resources (additional files under the skill directory). Parsing
handles YAML frontmatter delimited by ``---`` lines.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Final

import yaml  # type: ignore[import-untyped]

from src.common.exception import ApplicationError

#: Maximum allowed skill name length (characters).
MAX_NAME_LENGTH: Final[int] = 64
#: Maximum allowed skill description length (characters).
MAX_DESCRIPTION_LENGTH: Final[int] = 1024
#: Canonical skill instruction file name.
SKILL_FILE_NAME: Final[str] = "SKILL.md"

#: Reserved words that cannot appear in a skill name.
_RESERVED_WORDS: Final[tuple[str, ...]] = ("anthropic", "claude")
#: Detects XML tags in metadata.
_XML_TAG_PATTERN = re.compile(r"<[^>]+>")
#: File extensions treated as executable scripts.
_SCRIPT_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".py", ".sh", ".bash", ".js", ".ts", ".rb", ".pl", ".php"}
)
#: Extension to interpreter/language mapping.
_SCRIPT_LANGUAGES: Final[dict[str, str]] = {
    ".py": "python",
    ".sh": "bash",
    ".bash": "bash",
    ".js": "node",
    ".ts": "ts-node",
    ".rb": "ruby",
    ".pl": "perl",
    ".php": "php",
}


class SkillError(ApplicationError):
    """Base for every skill lifecycle failure."""

    code = "skill_error"
    message = "Skill error"


class SkillDisabledError(SkillError):
    """Raised when skills are not enabled."""

    code = "skill_disabled"
    message = "Skills are not enabled"


class SkillNotAllowedError(SkillError):
    """Raised when a skill is outside the configured allowlist."""

    code = "skill_not_allowed"
    message = "Skill not allowed"


class SkillNotFoundError(SkillError):
    """Raised when a skill cannot be found on disk."""

    code = "skill_not_found"
    message = "Skill not found"


class SkillValidationError(SkillError):
    """Raised when skill metadata fails validation."""

    code = "skill_validation"
    message = "Skill validation failed"


class SkillPathError(SkillError):
    """Raised when a skill file path is invalid or escapes its directory."""

    code = "skill_path"
    message = "Invalid skill file path"


@dataclass(frozen=True, slots=True)
class Skill:
    """A loaded skill: metadata plus on-demand instructions.

    ``loaded`` marks whether the Level 2 instructions are available. The
    field shape mirrors the tool-seam ``Skill`` so a manager instance can
    be consumed through the tool protocol unchanged.
    """

    name: str
    description: str = ""
    base_path: str = ""
    file_path: str = ""
    instructions: str = ""
    loaded: bool = False

    def validate(self) -> None:
        """Validate the skill metadata, raising on the first violation."""
        if self.name == "":
            raise SkillValidationError(message="skill name is required")
        if len(self.name) > MAX_NAME_LENGTH:
            raise SkillValidationError(
                message=(
                    f"skill name exceeds maximum length of {MAX_NAME_LENGTH} characters"
                )
            )
        if not _is_valid_skill_name(self.name):
            raise SkillValidationError(
                message=(
                    "skill name must contain only lowercase letters, numbers, "
                    "and hyphens"
                )
            )
        for reserved in _RESERVED_WORDS:
            if reserved in self.name:
                raise SkillValidationError(
                    message=f"skill name cannot contain reserved word: {reserved}"
                )
        if _XML_TAG_PATTERN.search(self.name) is not None:
            raise SkillValidationError(message="skill name cannot contain XML tags")
        if self.description == "":
            raise SkillValidationError(message="skill description is required")
        if len(self.description) > MAX_DESCRIPTION_LENGTH:
            raise SkillValidationError(
                message=(
                    "skill description exceeds maximum length of "
                    f"{MAX_DESCRIPTION_LENGTH} characters"
                )
            )
        if _XML_TAG_PATTERN.search(self.description) is not None:
            raise SkillValidationError(message="skill description cannot contain XML tags")

    def to_metadata(self) -> SkillMetadata:
        """Project onto the lightweight Level 1 metadata shape."""
        return SkillMetadata(
            name=self.name,
            description=self.description,
            base_path=self.base_path,
        )


@dataclass(frozen=True, slots=True)
class SkillMetadata:
    """Level 1 metadata used for system-prompt injection."""

    name: str
    description: str
    base_path: str = ""


@dataclass(frozen=True, slots=True)
class SkillFile:
    """An additional file within a skill directory (Level 3)."""

    name: str
    path: str
    content: str
    is_script: bool


@dataclass(frozen=True, slots=True)
class SkillInfo:
    """Detailed skill view returned to the agent."""

    name: str
    description: str
    base_path: str
    instructions: str
    files: list[str]


@dataclass(frozen=True, slots=True)
class ManagerConfig:
    """Configuration for the skills manager."""

    skill_dirs: list[str] = field(default_factory=list)
    allowed_skills: list[str] = field(default_factory=list)
    enabled: bool = False


def parse_skill_file(content: str) -> Skill:
    """Parse ``SKILL.md`` content into a validated ``Skill``.

    Expects YAML frontmatter delimited by ``---`` lines. Raises
    ``SkillValidationError`` when the frontmatter is missing or unclosed,
    or when the parsed metadata fails validation.
    """
    if not content.strip().startswith("---"):
        raise SkillValidationError(
            message="SKILL.md must start with YAML frontmatter (---)"
        )

    frontmatter_lines: list[str] = []
    body_lines: list[str] = []
    in_frontmatter = False
    frontmatter_ended = False
    for line in content.splitlines():
        if not in_frontmatter and not frontmatter_ended and line.strip() == "---":
            in_frontmatter = True
            continue
        if in_frontmatter and line.strip() == "---":
            in_frontmatter = False
            frontmatter_ended = True
            continue
        if in_frontmatter:
            frontmatter_lines.append(line)
        elif frontmatter_ended:
            body_lines.append(line)

    if not frontmatter_ended:
        raise SkillValidationError(
            message="SKILL.md frontmatter is not properly closed with ---"
        )

    frontmatter = "\n".join(frontmatter_lines)
    try:
        data = yaml.safe_load(frontmatter)
    except yaml.YAMLError as exc:
        raise SkillValidationError(
            message=f"failed to parse YAML frontmatter: {exc}"
        ) from exc

    name = ""
    description = ""
    if isinstance(data, dict):
        raw_name = data.get("name")
        raw_description = data.get("description")
        if isinstance(raw_name, str):
            name = raw_name
        if isinstance(raw_description, str):
            description = raw_description

    skill = Skill(
        name=name,
        description=description,
        instructions="\n".join(body_lines).strip(),
        loaded=True,
    )
    skill.validate()
    return skill


def parse_skill_metadata(content: str) -> SkillMetadata:
    """Parse only the Level 1 metadata from ``SKILL.md`` content."""
    return parse_skill_file(content).to_metadata()


def _extension(path: str) -> str:
    """Return the lowercased filename extension (including the dot)."""
    lowered = path.strip().lower()
    dot = lowered.rfind(".")
    if dot < 0:
        return ""
    return lowered[dot:]


def is_script(path: str) -> bool:
    """Return whether ``path`` marks an executable script by extension."""
    return _extension(path) in _SCRIPT_EXTENSIONS


def get_script_language(path: str) -> str:
    """Return the interpreter/language name for a script path."""
    return _SCRIPT_LANGUAGES.get(_extension(path), "unknown")


def _is_valid_skill_name(name: str) -> bool:
    """Return whether ``name`` is unicode letters, numbers, and hyphens only."""
    if not name:
        return False
    return all(ch == "-" or ch.isalpha() or ch.isnumeric() for ch in name)


__all__ = [
    "MAX_DESCRIPTION_LENGTH",
    "MAX_NAME_LENGTH",
    "SKILL_FILE_NAME",
    "ManagerConfig",
    "Skill",
    "SkillDisabledError",
    "SkillError",
    "SkillFile",
    "SkillInfo",
    "SkillMetadata",
    "SkillNotAllowedError",
    "SkillNotFoundError",
    "SkillPathError",
    "SkillValidationError",
    "get_script_language",
    "is_script",
    "parse_skill_file",
    "parse_skill_metadata",
]
