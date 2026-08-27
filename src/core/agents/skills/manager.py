"""Skills manager: discovery, loading, matching, and sandboxed execution.

Implements progressive disclosure: Level 1 metadata for the system prompt,
Level 2 full instructions loaded on demand, and Level 3 resources and
scripts. The manager coordinates the filesystem loader and the sandbox
backend, and enforces the enabled flag and the allowlist policy.
"""

from __future__ import annotations

import logging
import os
from dataclasses import replace
from pathlib import Path

from src.ai.embedding.base import Context
from src.core.agents.engine.sandbox.types import (
    ExecuteConfig,
    ExecuteResult,
)
from src.core.agents.engine.sandbox.types import (
    Manager as SandboxManager,
)
from src.core.agents.skills.types import (
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
    is_script,
    parse_skill_file,
)

logger = logging.getLogger(__name__)


class Loader:
    """Filesystem discovery and loading for skills.

    Separates lightweight metadata discovery (Level 1) from on-demand
    instruction and resource loading (Level 2/3).
    """

    def __init__(self, skill_dirs: list[str]) -> None:
        self._skill_dirs = skill_dirs
        self._discovered: dict[str, Skill] = {}

    def discover_skills(self) -> list[SkillMetadata]:
        """Scan all configured directories for ``SKILL.md`` files (Level 1)."""
        all_metadata: list[SkillMetadata] = []
        for dir_path in self._skill_dirs:
            all_metadata.extend(self._discover_in_directory(dir_path))
        return all_metadata

    def _discover_in_directory(self, dir_path: str) -> list[SkillMetadata]:
        """Scan one directory for skill subdirectories, skipping bad entries."""
        directory = Path(dir_path)
        if not directory.is_dir():
            return []
        try:
            entries = sorted(directory.iterdir())
        except OSError as exc:
            logger.warning("failed to read skill directory %s: %s", dir_path, exc)
            return []
        metadata: list[SkillMetadata] = []
        for entry in entries:
            if not entry.is_dir():
                continue
            skill_file = entry / SKILL_FILE_NAME
            if not skill_file.is_file():
                continue
            try:
                skill = parse_skill_file(skill_file.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, SkillValidationError):
                continue
            skill = replace(skill, base_path=str(entry), file_path=str(skill_file))
            self._discovered[skill.name] = skill
            metadata.append(skill.to_metadata())
        return metadata

    def load_skill_instructions(self, skill_name: str) -> Skill:
        """Load the full instructions of a skill (Level 2)."""
        cached = self._discovered.get(skill_name)
        if cached is not None and cached.loaded:
            return cached
        for dir_path in self._skill_dirs:
            skill = self._load_skill_from_directory(dir_path, skill_name)
            if skill is not None:
                self._discovered[skill_name] = skill
                return skill
        raise SkillNotFoundError(message=f"skill not found: {skill_name}")

    def _load_skill_from_directory(self, dir_path: str, skill_name: str) -> Skill | None:
        """Attempt to load a skill from one directory; ``None`` when absent."""
        directory = Path(dir_path)
        skill_path = directory / skill_name
        skill_file = skill_path / SKILL_FILE_NAME
        if skill_file.is_file():
            return self._load_skill_file(skill_path, skill_file)
        if not directory.is_dir():
            return None
        try:
            entries = sorted(directory.iterdir())
        except OSError:
            return None
        for entry in entries:
            if not entry.is_dir():
                continue
            candidate_file = entry / SKILL_FILE_NAME
            if not candidate_file.is_file():
                continue
            try:
                skill = parse_skill_file(candidate_file.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, SkillValidationError):
                continue
            if skill.name == skill_name:
                return replace(skill, base_path=str(entry), file_path=str(candidate_file))
        return None

    def _load_skill_file(self, base_path: Path, file_path: Path) -> Skill:
        """Read and parse a ``SKILL.md`` file, binding its filesystem paths."""
        try:
            content = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise SkillError(message=f"failed to read skill file: {exc}") from exc
        skill = parse_skill_file(content)
        return replace(skill, base_path=str(base_path), file_path=str(file_path))

    def load_skill_file(self, skill_name: str, relative_path: str) -> SkillFile:
        """Load an additional file from a skill directory (Level 3).

        ``relative_path`` must stay within the skill directory; path
        traversal and directory escapes are rejected.
        """
        skill = self._get_or_load_skill(skill_name)
        clean_path = os.path.normpath(relative_path)
        if clean_path.startswith("..") or os.path.isabs(clean_path):
            raise SkillPathError(message=f"invalid file path: {relative_path}")

        base = Path(skill.base_path)
        full_path = base / clean_path
        abs_skill = base.resolve()
        abs_file = full_path.resolve()
        if abs_file != abs_skill and not str(abs_file).startswith(str(abs_skill) + os.sep):
            raise SkillPathError(message=f"file path outside skill directory: {relative_path}")

        try:
            content = full_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise SkillNotFoundError(message=f"failed to read file: {exc}") from exc

        return SkillFile(
            name=relative_path,
            path=str(abs_file),
            content=content,
            is_script=is_script(relative_path),
        )

    def list_skill_files(self, skill_name: str) -> list[str]:
        """List all files under a skill directory as relative paths."""
        skill = self._get_or_load_skill(skill_name)
        base = Path(skill.base_path)
        if not base.is_dir():
            return []
        try:
            entries = sorted(base.rglob("*"))
        except OSError as exc:
            raise SkillError(message=f"failed to list skill files: {exc}") from exc
        return [str(path.relative_to(base)) for path in entries if path.is_file()]

    def get_skill_base_path(self, skill_name: str) -> str:
        """Return the absolute base path for a skill."""
        skill = self._get_or_load_skill(skill_name)
        return str(Path(skill.base_path).resolve())

    def reload(self) -> list[SkillMetadata]:
        """Clear the cache and rediscover all skills."""
        self._discovered = {}
        return self.discover_skills()

    def _get_or_load_skill(self, skill_name: str) -> Skill:
        """Return a cached skill, loading it on demand when absent."""
        skill = self._discovered.get(skill_name)
        if skill is not None:
            return skill
        return self.load_skill_instructions(skill_name)


class Manager:
    """Coordinates skills discovery, loading, and sandboxed execution.

    The manager enforces the enabled flag and the allowlist before any
    loader or sandbox call, and implements the skill-tool manager seam
    (``is_enabled`` / ``load_skill`` / ``read_skill_file`` /
    ``list_skill_files`` / ``execute_script``).
    """

    def __init__(
        self,
        config: ManagerConfig | None = None,
        sandbox_manager: SandboxManager | None = None,
    ) -> None:
        resolved = config if config is not None else ManagerConfig()
        self._config = resolved
        self._loader = Loader(resolved.skill_dirs)
        self._sandbox_manager = sandbox_manager
        self._metadata_cache: list[SkillMetadata] = []

    def is_enabled(self) -> bool:
        """Return whether skills are enabled."""
        return self._config.enabled

    def initialize(self) -> None:
        """Discover all skills and cache their metadata (startup call)."""
        if not self._config.enabled:
            return
        metadata = self._loader.discover_skills()
        self._metadata_cache = self._filter_allowed(metadata)

    def get_all_metadata(self) -> list[SkillMetadata]:
        """Return metadata for all discovered skills (Level 1)."""
        if not self._config.enabled:
            return []
        return list(self._metadata_cache)

    async def load_skill(self, ctx: Context, skill_name: str) -> Skill:
        """Load the full instructions of a skill (Level 2)."""
        self._require_enabled()
        self._require_allowed(skill_name)
        return self._loader.load_skill_instructions(skill_name)

    async def read_skill_file(
        self,
        ctx: Context,
        skill_name: str,
        file_path: str,
    ) -> str:
        """Read an additional file from a skill directory (Level 3)."""
        self._require_enabled()
        self._require_allowed(skill_name)
        return self._loader.load_skill_file(skill_name, file_path).content

    async def list_skill_files(self, ctx: Context, skill_name: str) -> list[str]:
        """List all files in a skill directory."""
        self._require_enabled()
        self._require_allowed(skill_name)
        return self._loader.list_skill_files(skill_name)

    async def execute_script(
        self,
        ctx: Context,
        skill_name: str,
        script_path: str,
        args: list[str],
        stdin: str,
    ) -> ExecuteResult:
        """Execute a skill script in the sandbox."""
        self._require_enabled()
        self._require_allowed(skill_name)
        if self._sandbox_manager is None:
            raise SkillError(message="sandbox is not configured")
        base_path = self._loader.get_skill_base_path(skill_name)
        skill_file = self._loader.load_skill_file(skill_name, script_path)
        if not skill_file.is_script:
            raise SkillValidationError(message=f"file is not an executable script: {script_path}")
        config = ExecuteConfig(
            script=skill_file.path,
            args=args,
            work_dir=base_path,
            stdin=stdin,
        )
        return await self._sandbox_manager.execute(config)

    async def get_skill_info(self, ctx: Context, skill_name: str) -> SkillInfo:
        """Return detailed information about a skill."""
        self._require_enabled()
        self._require_allowed(skill_name)
        skill = self._loader.load_skill_instructions(skill_name)
        try:
            files = self._loader.list_skill_files(skill_name)
        except SkillError:
            files = []
        return SkillInfo(
            name=skill.name,
            description=skill.description,
            base_path=skill.base_path,
            instructions=skill.instructions,
            files=files,
        )

    async def reload(self, ctx: Context) -> None:
        """Refresh the skill cache by rediscovering all skills."""
        if not self._config.enabled:
            return
        self._metadata_cache = self._filter_allowed(self._loader.reload())

    async def cleanup(self, ctx: Context) -> None:
        """Release sandbox resources."""
        if self._sandbox_manager is not None:
            await self._sandbox_manager.cleanup()

    def _require_enabled(self) -> None:
        if not self._config.enabled:
            raise SkillDisabledError()

    def _require_allowed(self, skill_name: str) -> None:
        if not self._is_skill_allowed(skill_name):
            raise SkillNotAllowedError(message=f"skill not allowed: {skill_name}")

    def _is_skill_allowed(self, skill_name: str) -> bool:
        if not self._config.allowed_skills:
            return True
        return skill_name in self._config.allowed_skills

    def _filter_allowed(self, metadata: list[SkillMetadata]) -> list[SkillMetadata]:
        if not self._config.allowed_skills:
            return metadata
        allowed = set(self._config.allowed_skills)
        return [meta for meta in metadata if meta.name in allowed]


__all__ = [
    "Loader",
    "Manager",
]
