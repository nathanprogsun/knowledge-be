"""Unit tests for the agent skills manager.

Filesystem-backed discovery and loading tests use real temporary
directories (pytest ``tmp_path``); the sandbox is injected as a fake seam
so execution is asserted without a real backend. Every test follows the
Arrange-Act-Assert structure.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.core.agents.engine.sandbox.types import ExecuteConfig, ExecuteResult
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

# ── Test doubles ─────────────────────────────────────────────────────────


class _Context:
    """Minimal task context satisfying the ``Context`` protocol."""

    is_background_task: bool = False


class _FakeSandbox:
    """Sandbox seam that records executions and returns a canned result."""

    def __init__(self, result: ExecuteResult | None = None) -> None:
        self.result = result if result is not None else ExecuteResult(stdout="ok")
        self.calls: list[ExecuteConfig] = []
        self.cleaned_up = False

    async def execute(self, config: ExecuteConfig) -> ExecuteResult:
        self.calls.append(config)
        return self.result

    async def cleanup(self) -> None:
        self.cleaned_up = True


def _write_skill(
    base: Path, name: str, *, description: str = "A test skill"
) -> Path:
    """Write a valid ``SKILL.md`` for ``name`` under ``base``."""
    skill_dir = base / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / SKILL_FILE_NAME
    skill_file.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n"
        f"# {name}\n\nBody.\n",
        encoding="utf-8",
    )
    return skill_dir


def _make_manager(
    tmp_path: Path,
    *,
    sandbox: _FakeSandbox | None = None,
    allowed_skills: list[str] | None = None,
    enabled: bool = True,
) -> Manager:
    """Build a manager over a temp skill dir, optionally allowing a subset."""
    return Manager(
        ManagerConfig(
            skill_dirs=[str(tmp_path)],
            allowed_skills=allowed_skills or [],
            enabled=enabled,
        ),
        sandbox_manager=sandbox,
    )


VALID_SKILL = """\
---
name: test-skill
description: A test skill.
---
# Test Skill

This is the content.
"""

# ── Parsing and validation ───────────────────────────────────────────────


def test_parse_skill_file() -> None:
    skill = parse_skill_file(VALID_SKILL)
    assert skill.name == "test-skill"
    assert skill.description == "A test skill."
    assert skill.instructions == "# Test Skill\n\nThis is the content."
    assert skill.loaded is True
    assert skill.base_path == ""
    assert skill.file_path == ""


def test_parse_skill_metadata() -> None:
    metadata = parse_skill_metadata(VALID_SKILL)
    assert isinstance(metadata, SkillMetadata)
    assert metadata.name == "test-skill"
    assert metadata.description == "A test skill."
    assert metadata.base_path == ""


def test_parse_skill_file_missing_frontmatter() -> None:
    with pytest.raises(SkillValidationError, match="must start with YAML frontmatter"):
        parse_skill_file("# no frontmatter\n")


def test_parse_skill_file_unclosed_frontmatter() -> None:
    content = "---\nname: test-skill\ndescription: A skill\n"
    with pytest.raises(SkillValidationError, match="not properly closed"):
        parse_skill_file(content)


def test_parse_skill_file_invalid_yaml() -> None:
    content = "---\nname: [unclosed\ndescription: A skill\n---\n"
    with pytest.raises(SkillValidationError, match="failed to parse YAML frontmatter"):
        parse_skill_file(content)


def test_parse_skill_file_missing_description_fails_validation() -> None:
    content = "---\nname: test-skill\n---\nbody\n"
    with pytest.raises(SkillValidationError, match="description is required"):
        parse_skill_file(content)


def test_validate_name_required() -> None:
    skill = Skill(name="", description="A skill")
    with pytest.raises(SkillValidationError, match="name is required"):
        skill.validate()


def test_validate_name_too_long() -> None:
    skill = Skill(name="x" * (MAX_NAME_LENGTH + 1), description="A skill")
    with pytest.raises(SkillValidationError, match="exceeds maximum length"):
        skill.validate()


def test_validate_name_invalid_characters() -> None:
    skill = Skill(name="My Skill", description="A skill")
    with pytest.raises(SkillValidationError, match="lowercase letters"):
        skill.validate()


def test_validate_name_reserved_word() -> None:
    skill = Skill(name="my-claude-skill", description="A skill")
    with pytest.raises(SkillValidationError, match="reserved word"):
        skill.validate()


def test_validate_description_required() -> None:
    skill = Skill(name="my-skill", description="")
    with pytest.raises(SkillValidationError, match="description is required"):
        skill.validate()


def test_validate_description_too_long() -> None:
    skill = Skill(name="my-skill", description="x" * (MAX_DESCRIPTION_LENGTH + 1))
    with pytest.raises(SkillValidationError, match="exceeds maximum length"):
        skill.validate()


def test_validate_description_xml_tags() -> None:
    skill = Skill(name="my-skill", description="A <b>bold</b> description")
    with pytest.raises(SkillValidationError, match="cannot contain XML tags"):
        skill.validate()


def test_validate_valid_skill_passes() -> None:
    skill = Skill(name="my-skill", description="A valid skill")
    skill.validate()


def test_to_metadata() -> None:
    skill = Skill(name="my-skill", description="A skill", base_path="/tmp/skills/my-skill")
    metadata = skill.to_metadata()
    assert metadata.name == "my-skill"
    assert metadata.description == "A skill"
    assert metadata.base_path == "/tmp/skills/my-skill"


# ── Script helpers ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("script.py", True),
        ("script.sh", True),
        ("script.bash", True),
        ("script.js", True),
        ("script.ts", True),
        ("script.rb", True),
        ("script.pl", True),
        ("script.php", True),
        ("dir/script.py", True),
        ("SCRIPT.PY", True),
        ("README.md", False),
        ("data.json", False),
        ("config.yaml", False),
        ("noextension", False),
    ],
)
def test_is_script(path: str, expected: bool) -> None:
    assert is_script(path) is expected


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("script.py", "python"),
        ("script.sh", "bash"),
        ("script.bash", "bash"),
        ("script.js", "node"),
        ("script.ts", "ts-node"),
        ("script.rb", "ruby"),
        ("script.pl", "perl"),
        ("script.php", "php"),
        ("README.md", "unknown"),
    ],
)
def test_get_script_language(path: str, expected: str) -> None:
    assert get_script_language(path) == expected


# ── Loader: discovery ────────────────────────────────────────────────────


def test_loader_discover_skills(tmp_path: Path) -> None:
    _write_skill(tmp_path, "test-skill")
    _write_skill(tmp_path, "another-skill")

    loader = Loader([str(tmp_path)])

    metadata = loader.discover_skills()
    assert [m.name for m in metadata] == ["another-skill", "test-skill"]


def test_loader_discover_skips_invalid_entries(tmp_path: Path) -> None:
    _write_skill(tmp_path, "good-skill")
    (tmp_path / "no-skill").mkdir()
    bad_dir = tmp_path / "bad-skill"
    bad_dir.mkdir()
    (bad_dir / SKILL_FILE_NAME).write_text("# no frontmatter\n", encoding="utf-8")
    (tmp_path / "plain.txt").write_text("hi", encoding="utf-8")

    loader = Loader([str(tmp_path)])

    metadata = loader.discover_skills()
    assert [m.name for m in metadata] == ["good-skill"]


def test_loader_discover_missing_directory_is_silent(tmp_path: Path) -> None:
    loader = Loader([str(tmp_path / "missing")])
    assert loader.discover_skills() == []


def test_loader_discover_multiple_directories(tmp_path: Path) -> None:
    first = tmp_path / "one"
    second = tmp_path / "two"
    _write_skill(first, "skill-a")
    _write_skill(second, "skill-b")

    loader = Loader([str(first), str(second)])

    names = [m.name for m in loader.discover_skills()]
    assert names == ["skill-a", "skill-b"]


# ── Loader: instructions ─────────────────────────────────────────────────


def test_loader_load_skill_instructions(tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path, "test-skill")

    loader = Loader([str(tmp_path)])
    skill = loader.load_skill_instructions("test-skill")

    assert skill.name == "test-skill"
    assert skill.base_path == str(skill_dir)
    assert skill.file_path == str(skill_dir / SKILL_FILE_NAME)
    assert skill.loaded is True
    assert "Body." in skill.instructions


def test_loader_load_skill_instructions_caches(tmp_path: Path) -> None:
    _write_skill(tmp_path, "test-skill")

    loader = Loader([str(tmp_path)])
    first = loader.load_skill_instructions("test-skill")
    second = loader.load_skill_instructions("test-skill")

    assert first is second


def test_loader_load_skill_by_scanning(tmp_path: Path) -> None:
    skill_dir = tmp_path / "zzz"
    skill_dir.mkdir()
    (skill_dir / SKILL_FILE_NAME).write_text(
        "---\nname: real-name\ndescription: A skill\n---\nBody\n",
        encoding="utf-8",
    )

    loader = Loader([str(tmp_path)])
    skill = loader.load_skill_instructions("real-name")

    assert skill.name == "real-name"
    assert skill.base_path == str(skill_dir)


def test_loader_load_skill_not_found(tmp_path: Path) -> None:
    loader = Loader([str(tmp_path)])

    with pytest.raises(SkillNotFoundError, match="skill not found"):
        loader.load_skill_instructions("nope")


# ── Loader: skill files (Level 3) ────────────────────────────────────────


def test_loader_load_skill_file(tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path, "test-skill")
    (skill_dir / "GUIDE.md").write_text("# Guide\n\nContent.", encoding="utf-8")
    script_dir = skill_dir / "scripts"
    script_dir.mkdir()
    (script_dir / "hello.py").write_text("print('hi')", encoding="utf-8")

    loader = Loader([str(tmp_path)])
    loader.discover_skills()

    guide = loader.load_skill_file("test-skill", "GUIDE.md")
    assert isinstance(guide, SkillFile)
    assert guide.name == "GUIDE.md"
    assert guide.content == "# Guide\n\nContent."
    assert guide.is_script is False
    assert guide.path == str((skill_dir / "GUIDE.md").resolve())

    script = loader.load_skill_file("test-skill", "scripts/hello.py")
    assert script.name == "scripts/hello.py"
    assert script.content == "print('hi')"
    assert script.is_script is True


def test_loader_load_skill_file_rejects_traversal(tmp_path: Path) -> None:
    _write_skill(tmp_path, "test-skill")
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")

    loader = Loader([str(tmp_path)])
    loader.discover_skills()

    for bad in ("../secret.txt", "..", "..foo"):
        with pytest.raises(SkillPathError):
            loader.load_skill_file("test-skill", bad)


def test_loader_load_skill_file_rejects_absolute_path(tmp_path: Path) -> None:
    _write_skill(tmp_path, "test-skill")
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")

    loader = Loader([str(tmp_path)])
    loader.discover_skills()

    with pytest.raises(SkillPathError, match="invalid file path"):
        loader.load_skill_file("test-skill", str(outside))


def test_loader_load_skill_file_rejects_symlink_escape(tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path, "test-skill")
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (skill_dir / "escape.txt").symlink_to(outside)

    loader = Loader([str(tmp_path)])
    loader.discover_skills()

    with pytest.raises(SkillPathError, match="outside skill directory"):
        loader.load_skill_file("test-skill", "escape.txt")


def test_loader_load_skill_file_missing_file(tmp_path: Path) -> None:
    _write_skill(tmp_path, "test-skill")

    loader = Loader([str(tmp_path)])
    loader.discover_skills()

    with pytest.raises(SkillNotFoundError):
        loader.load_skill_file("test-skill", "missing.md")


def test_loader_load_skill_file_unknown_skill(tmp_path: Path) -> None:
    loader = Loader([str(tmp_path)])

    with pytest.raises(SkillNotFoundError, match="skill not found"):
        loader.load_skill_file("nope", "GUIDE.md")


# ── Loader: listing, base path, reload ───────────────────────────────────


def test_loader_list_skill_files(tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path, "test-skill")
    (skill_dir / "GUIDE.md").write_text("# Guide", encoding="utf-8")
    (skill_dir / "scripts").mkdir()
    (skill_dir / "scripts" / "run.py").write_text("print(1)", encoding="utf-8")

    loader = Loader([str(tmp_path)])
    loader.discover_skills()

    files = loader.list_skill_files("test-skill")
    assert sorted(files) == ["GUIDE.md", SKILL_FILE_NAME, "scripts/run.py"]


def test_loader_get_skill_base_path(tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path, "test-skill")

    loader = Loader([str(tmp_path)])
    loader.discover_skills()

    assert loader.get_skill_base_path("test-skill") == str(skill_dir.resolve())


def test_loader_get_skill_base_path_unknown(tmp_path: Path) -> None:
    loader = Loader([str(tmp_path)])

    with pytest.raises(SkillNotFoundError):
        loader.get_skill_base_path("nope")


def test_loader_reload_clears_cache(tmp_path: Path) -> None:
    _write_skill(tmp_path, "test-skill")

    loader = Loader([str(tmp_path)])
    loader.discover_skills()
    assert loader.load_skill_instructions("test-skill").name == "test-skill"

    shutil.rmtree(tmp_path / "test-skill")
    loader.reload()

    with pytest.raises(SkillNotFoundError):
        loader.load_skill_instructions("test-skill")


# ── Manager: enabled / disabled policy ───────────────────────────────────


async def test_manager_disabled_is_inert(tmp_path: Path) -> None:
    _write_skill(tmp_path, "test-skill")
    manager = _make_manager(tmp_path, enabled=False)

    assert manager.is_enabled() is False
    manager.initialize()
    assert manager.get_all_metadata() == []

    with pytest.raises(SkillDisabledError):
        await manager.load_skill(_Context(), "test-skill")
    with pytest.raises(SkillDisabledError):
        await manager.read_skill_file(_Context(), "test-skill", "GUIDE.md")
    with pytest.raises(SkillDisabledError):
        await manager.list_skill_files(_Context(), "test-skill")
    with pytest.raises(SkillDisabledError):
        await manager.execute_script(_Context(), "test-skill", "run.py", [], "")
    with pytest.raises(SkillDisabledError):
        await manager.get_skill_info(_Context(), "test-skill")


def test_manager_initialize_caches_metadata(tmp_path: Path) -> None:
    _write_skill(tmp_path, "test-skill")
    manager = _make_manager(tmp_path)

    manager.initialize()

    assert [m.name for m in manager.get_all_metadata()] == ["test-skill"]


def test_manager_initialize_filters_allowed(tmp_path: Path) -> None:
    _write_skill(tmp_path, "keep-skill")
    _write_skill(tmp_path, "drop-skill")
    manager = _make_manager(tmp_path, allowed_skills=["keep-skill"])

    manager.initialize()

    assert [m.name for m in manager.get_all_metadata()] == ["keep-skill"]


# ── Manager: loading and reading ─────────────────────────────────────────


async def test_manager_load_skill(tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path, "test-skill")
    manager = _make_manager(tmp_path)
    manager.initialize()

    skill = await manager.load_skill(_Context(), "test-skill")

    assert skill.name == "test-skill"
    assert skill.base_path == str(skill_dir)


async def test_manager_load_skill_denied_by_allowlist(tmp_path: Path) -> None:
    _write_skill(tmp_path, "keep-skill")
    _write_skill(tmp_path, "drop-skill")
    manager = _make_manager(tmp_path, allowed_skills=["keep-skill"])
    manager.initialize()

    with pytest.raises(SkillNotAllowedError, match="skill not allowed"):
        await manager.load_skill(_Context(), "drop-skill")


async def test_manager_read_skill_file(tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path, "test-skill")
    (skill_dir / "GUIDE.md").write_text("guide content", encoding="utf-8")
    manager = _make_manager(tmp_path)
    manager.initialize()

    content = await manager.read_skill_file(_Context(), "test-skill", "GUIDE.md")

    assert content == "guide content"


async def test_manager_list_skill_files(tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path, "test-skill")
    (skill_dir / "extra.md").write_text("x", encoding="utf-8")
    manager = _make_manager(tmp_path)
    manager.initialize()

    files = await manager.list_skill_files(_Context(), "test-skill")

    assert sorted(files) == [SKILL_FILE_NAME, "extra.md"]


async def test_manager_get_skill_info(tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path, "test-skill")
    (skill_dir / "GUIDE.md").write_text("guide", encoding="utf-8")
    manager = _make_manager(tmp_path)
    manager.initialize()

    info = await manager.get_skill_info(_Context(), "test-skill")

    assert isinstance(info, SkillInfo)
    assert info.name == "test-skill"
    assert info.description == "A test skill"
    assert info.base_path == str(skill_dir)
    assert sorted(info.files) == sorted([SKILL_FILE_NAME, "GUIDE.md"])


# ── Manager: script execution ────────────────────────────────────────────


async def test_manager_execute_script_builds_config(tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path, "test-skill")
    (skill_dir / "scripts").mkdir()
    (skill_dir / "scripts" / "run.py").write_text("print('hi')", encoding="utf-8")
    sandbox = _FakeSandbox(ExecuteResult(stdout="hi", exit_code=0))
    manager = _make_manager(tmp_path, sandbox=sandbox)
    manager.initialize()

    result = await manager.execute_script(
        _Context(), "test-skill", "scripts/run.py", ["--flag"], "input"
    )

    assert result.stdout == "hi"
    assert len(sandbox.calls) == 1
    config = sandbox.calls[0]
    assert config.script == str((skill_dir / "scripts" / "run.py").resolve())
    assert config.args == ["--flag"]
    assert config.work_dir == str(skill_dir.resolve())
    assert config.stdin == "input"


async def test_manager_execute_script_rejects_non_script(tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path, "test-skill")
    (skill_dir / "README.md").write_text("not a script", encoding="utf-8")
    sandbox = _FakeSandbox()
    manager = _make_manager(tmp_path, sandbox=sandbox)
    manager.initialize()

    with pytest.raises(SkillValidationError, match="not an executable script"):
        await manager.execute_script(_Context(), "test-skill", "README.md", [], "")

    assert sandbox.calls == []


async def test_manager_execute_script_without_sandbox(tmp_path: Path) -> None:
    _write_skill(tmp_path, "test-skill")
    manager = _make_manager(tmp_path)
    manager.initialize()

    with pytest.raises(SkillError, match="sandbox is not configured"):
        await manager.execute_script(_Context(), "test-skill", "run.py", [], "")


async def test_manager_execute_script_denied_by_allowlist(tmp_path: Path) -> None:
    _write_skill(tmp_path, "keep-skill")
    manager = _make_manager(tmp_path, allowed_skills=["keep-skill"], sandbox=_FakeSandbox())
    manager.initialize()

    with pytest.raises(SkillNotAllowedError):
        await manager.execute_script(_Context(), "drop-skill", "run.py", [], "")


# ── Manager: reload and cleanup ──────────────────────────────────────────


async def test_manager_reload_rediscovers(tmp_path: Path) -> None:
    _write_skill(tmp_path, "skill-a")
    manager = _make_manager(tmp_path)
    manager.initialize()
    assert len(manager.get_all_metadata()) == 1

    _write_skill(tmp_path, "skill-b")
    await manager.reload(_Context())

    assert [m.name for m in manager.get_all_metadata()] == ["skill-a", "skill-b"]


async def test_manager_reload_disabled_is_noop(tmp_path: Path) -> None:
    _write_skill(tmp_path, "skill-a")
    manager = _make_manager(tmp_path, enabled=False)
    manager.initialize()

    await manager.reload(_Context())

    assert manager.get_all_metadata() == []


async def test_manager_cleanup_delegates_to_sandbox() -> None:
    sandbox = _FakeSandbox()
    manager = Manager(
        ManagerConfig(enabled=True),
        sandbox_manager=sandbox,
    )

    await manager.cleanup(_Context())

    assert sandbox.cleaned_up is True


async def test_manager_cleanup_without_sandbox_is_noop() -> None:
    manager = Manager(ManagerConfig(enabled=True))

    await manager.cleanup(_Context())
