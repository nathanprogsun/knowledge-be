"""Agent skills: discovery, loading, matching, and sandboxed execution.

Implements progressive disclosure: Level 1 metadata, Level 2 instructions,
and Level 3 resources. Exports the manager, the loader, and the skill types.
"""

from __future__ import annotations

from src.core.agents.skills.manager import Loader, Manager
from src.core.agents.skills.types import (
    MAX_DESCRIPTION_LENGTH,
    MAX_NAME_LENGTH,
    SKILL_FILE_NAME,
    ManagerConfig,
    Skill,
    SkillDisabledError,
    SkillError,
    SkillFile,
    SkillInfo,
    SkillMetadata,
    SkillNotAllowedError,
    SkillNotFoundError,
    SkillPathError,
    SkillValidationError,
    get_script_language,
    is_script,
    parse_skill_file,
    parse_skill_metadata,
)

__all__ = [
    "MAX_DESCRIPTION_LENGTH",
    "MAX_NAME_LENGTH",
    "SKILL_FILE_NAME",
    "Loader",
    "Manager",
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
