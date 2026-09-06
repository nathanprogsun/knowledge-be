"""Unit tests for the parser-engine registry and the storage allowlist.

Both modules are pure functions over the local registry / environment,
so no fakes are needed — the allowlist value is passed explicitly where
possible and via ``monkeypatch`` where the env-read path itself is under
test.
"""

from __future__ import annotations

import pytest

from src.core.system.parser_engine import (
    BUILTIN_ENGINE_NAME,
    KB_CLOUD_APP_ID_OVERRIDE,
    KB_CLOUD_ENGINE_NAME,
    LOCAL_PARSER_ENGINES,
    MINERU_API_KEY_OVERRIDE,
    MINERU_CLOUD_ENGINE_NAME,
    MINERU_ENDPOINT_OVERRIDE,
    MINERU_ENGINE_NAME,
    PADDLEOCR_VL_CLOUD_ENGINE_NAME,
    PADDLEOCR_VL_CLOUD_TOKEN_OVERRIDE,
    PADDLEOCR_VL_ENDPOINT_OVERRIDE,
    PADDLEOCR_VL_ENGINE_NAME,
    SIMPLE_ENGINE_NAME,
    ParserEngineInfo,
    list_all_engines,
    local_engine_names,
)
from src.core.system.storage_allowlist import (
    ALLOW_LIST_ENV,
    SUPPORTED_STORAGE_PROVIDERS,
    allowed_provider_map,
    allowed_providers,
    build_storage_provider_statuses,
    first_allowed_provider,
    is_storage_provider_allowed,
    supported_providers,
)

# ── Parser engines: registration ────────────────────────────────────


def test_local_engine_names_match_the_go_registration_order() -> None:
    # Arrange / Act
    names = local_engine_names()

    # Assert — order of ``RegisterEngine`` calls in ``init()``
    assert names == [
        BUILTIN_ENGINE_NAME,
        SIMPLE_ENGINE_NAME,
        KB_CLOUD_ENGINE_NAME,
        MINERU_ENGINE_NAME,
        MINERU_CLOUD_ENGINE_NAME,
        PADDLEOCR_VL_ENGINE_NAME,
        PADDLEOCR_VL_CLOUD_ENGINE_NAME,
    ]


def test_every_local_engine_declares_file_types() -> None:
    # Arrange / Act / Assert
    for spec in LOCAL_PARSER_ENGINES:
        assert spec.file_types, spec.name
        assert spec.description, spec.name


def test_builtin_engine_file_types_match_in_process_handlers() -> None:
    # Arrange
    engines = {e.name: e for e in list_all_engines(docreader_connected=False)}

    # Act
    builtin = engines[BUILTIN_ENGINE_NAME]

    # Assert — only types the in-process reader actually handles
    assert {
        "md",
        "markdown",
        "txt",
        "csv",
        "json",
        "html",
        "htm",
        "jpg",
        "jpeg",
        "png",
        "gif",
        "bmp",
        "tiff",
        "webp",
    } <= set(builtin.file_types)
    assert "docx" not in builtin.file_types
    assert "pdf" not in builtin.file_types


# ── Parser engines: availability ────────────────────────────────────


def test_simple_engine_is_available_without_any_configuration() -> None:
    # Arrange / Act
    engines = {e.name: e for e in list_all_engines(docreader_connected=False)}

    # Assert
    assert engines[SIMPLE_ENGINE_NAME].available is True
    assert engines[SIMPLE_ENGINE_NAME].unavailable_reason == ""


@pytest.mark.parametrize("docreader_connected", [False, True])
def test_builtin_engine_is_available_without_docreader(docreader_connected: bool) -> None:
    # Arrange / Act
    engines = {e.name: e for e in list_all_engines(docreader_connected=docreader_connected)}

    # Assert — builtin is in-process; docreader connectivity does not gate it
    assert engines[BUILTIN_ENGINE_NAME].available is True
    assert engines[BUILTIN_ENGINE_NAME].unavailable_reason == ""


def test_cloud_engine_needs_an_app_id_override() -> None:
    # Arrange / Act
    without = {e.name: e for e in list_all_engines()}
    with_creds = {
        e.name: e for e in list_all_engines(overrides={KB_CLOUD_APP_ID_OVERRIDE: "app-123"})
    }

    # Assert
    assert without[KB_CLOUD_ENGINE_NAME].available is False
    assert "Knowledge Base Cloud credentials not configured" in (
        without[KB_CLOUD_ENGINE_NAME].unavailable_reason
    )
    assert with_creds[KB_CLOUD_ENGINE_NAME].available is True


@pytest.mark.parametrize(
    ("engine_name", "override_key", "expected_reason"),
    [
        (MINERU_ENGINE_NAME, MINERU_ENDPOINT_OVERRIDE, "MinerU service not configured"),
        (MINERU_CLOUD_ENGINE_NAME, MINERU_API_KEY_OVERRIDE, "MinerU API Key not configured"),
        (
            PADDLEOCR_VL_ENGINE_NAME,
            PADDLEOCR_VL_ENDPOINT_OVERRIDE,
            "PaddleOCR-VL service not configured",
        ),
        (
            PADDLEOCR_VL_CLOUD_ENGINE_NAME,
            PADDLEOCR_VL_CLOUD_TOKEN_OVERRIDE,
            "PaddleOCR-VL Cloud Token not configured",
        ),
    ],
)
def test_override_backed_engines_report_their_go_reason_when_unconfigured(
    engine_name: str,
    override_key: str,
    expected_reason: str,
) -> None:
    # Arrange / Act
    unconfigured = {e.name: e for e in list_all_engines()}
    configured = {e.name: e for e in list_all_engines(overrides={override_key: "value"})}

    # Assert
    assert unconfigured[engine_name].available is False
    assert unconfigured[engine_name].unavailable_reason == expected_reason
    assert configured[engine_name].available is True


def test_blank_override_values_do_not_enable_an_engine() -> None:
    # Arrange / Act — Go trims the override before the empty check
    engines = {e.name: e for e in list_all_engines(overrides={MINERU_ENDPOINT_OVERRIDE: "   "})}

    # Assert
    assert engines[MINERU_ENGINE_NAME].available is False


# ── Parser engines: remote merge ────────────────────────────────────


def test_remote_engine_overrides_the_local_description_and_file_types() -> None:
    # Arrange
    remote = ParserEngineInfo(
        name=BUILTIN_ENGINE_NAME,
        description="Remote builtin",
        file_types=["pdf"],
        available=False,
    )

    # Act
    engines = {
        e.name: e for e in list_all_engines(docreader_connected=True, remote_engines=[remote])
    }

    # Assert — remote wins on capabilities, local wins on availability
    assert engines[BUILTIN_ENGINE_NAME].description == "Remote builtin"
    assert engines[BUILTIN_ENGINE_NAME].file_types == ["pdf"]
    assert engines[BUILTIN_ENGINE_NAME].available is True


def test_remote_engine_with_empty_fields_does_not_clobber_the_local_spec() -> None:
    # Arrange
    remote = ParserEngineInfo(name=SIMPLE_ENGINE_NAME, description="", file_types=[])

    # Act
    engines = {e.name: e for e in list_all_engines(remote_engines=[remote])}

    # Assert
    assert engines[SIMPLE_ENGINE_NAME].description.startswith("Simple format")
    assert "txt" in engines[SIMPLE_ENGINE_NAME].file_types


def test_remote_only_engines_are_appended_verbatim() -> None:
    # Arrange
    remote = ParserEngineInfo(
        name="markitdown",
        description="Remote-only engine",
        file_types=["docx"],
        available=True,
    )

    # Act
    engines = list_all_engines(docreader_connected=True, remote_engines=[remote])

    # Assert — appended after every local engine, unchanged
    assert engines[-1] == remote
    assert len(engines) == len(LOCAL_PARSER_ENGINES) + 1


def test_parser_engine_info_serializes_with_pascal_case_keys() -> None:
    # Arrange
    engine = ParserEngineInfo(name="simple", description="d", file_types=["md"], available=True)

    # Act
    payload = engine.model_dump(by_alias=True)

    # Assert — Go's ``ParserEngineInfo`` has no json tags
    assert payload == {
        "Name": "simple",
        "Description": "d",
        "FileTypes": ["md"],
        "Available": True,
        "UnavailableReason": "",
    }


# ── Storage allowlist ───────────────────────────────────────────────


def test_supported_providers_match_the_go_canonical_order() -> None:
    # Arrange / Act / Assert
    assert supported_providers() == [
        "local",
        "minio",
        "cos",
        "tos",
        "s3",
        "oss",
        "ks3",
        "obs",
    ]


def test_a_blank_allow_list_allows_every_provider() -> None:
    # Arrange / Act
    allowed = allowed_provider_map("")

    # Assert
    assert all(allowed[name] for name in SUPPORTED_STORAGE_PROVIDERS)


def test_an_allow_list_restricts_to_the_named_providers() -> None:
    # Arrange / Act
    allowed = allowed_provider_map("minio,cos")

    # Assert
    assert allowed["minio"] is True
    assert allowed["cos"] is True
    assert allowed["local"] is False
    assert allowed["obs"] is False


@pytest.mark.parametrize("raw", ["obs,minio", "obs;minio", "obs|minio", "obs minio", "obs\tminio"])
def test_every_go_separator_is_accepted(raw: str) -> None:
    # Arrange / Act / Assert — canonical order, not input order
    assert allowed_providers(raw) == ["minio", "obs"]


def test_provider_names_are_case_insensitive_and_trimmed() -> None:
    # Arrange / Act / Assert
    assert allowed_providers("  MinIO , COS ") == ["minio", "cos"]


def test_unknown_provider_names_are_dropped() -> None:
    # Arrange / Act / Assert
    assert allowed_providers("minio,notareal") == ["minio"]
    assert is_storage_provider_allowed("notareal", "minio,notareal") is False


def test_an_empty_provider_is_treated_as_allowed() -> None:
    # Arrange / Act / Assert — Go's ``IsAllowed("")``
    assert is_storage_provider_allowed("", "minio") is True
    assert is_storage_provider_allowed("   ", "minio") is True


def test_first_allowed_provider_follows_the_canonical_order() -> None:
    # Arrange / Act / Assert
    assert first_allowed_provider("minio") == "minio"
    assert first_allowed_provider("obs,minio") == "minio"
    assert first_allowed_provider("") == "local"
    assert first_allowed_provider("notareal") == ""


def test_the_allow_list_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setenv(ALLOW_LIST_ENV, "cos")

    # Act / Assert
    assert allowed_providers() == ["cos"]
    assert is_storage_provider_allowed("minio") is False


def test_a_missing_env_var_allows_every_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.delenv(ALLOW_LIST_ENV, raising=False)

    # Act / Assert
    assert allowed_providers() == list(SUPPORTED_STORAGE_PROVIDERS)


# ── Storage provider statuses ───────────────────────────────────────


def test_storage_statuses_cover_every_provider_in_canonical_order() -> None:
    # Arrange / Act
    statuses = build_storage_provider_statuses(raw_allow_list="")

    # Assert
    assert [s.name for s in statuses] == list(SUPPORTED_STORAGE_PROVIDERS)
    assert all(s.description for s in statuses)


def test_local_storage_is_always_available() -> None:
    # Arrange / Act
    statuses = {s.name: s for s in build_storage_provider_statuses(raw_allow_list="")}

    # Assert — Go hard-codes ``Available: true`` for local
    assert statuses["local"].available is True


def test_only_configured_providers_are_reported_available() -> None:
    # Arrange / Act
    statuses = {
        s.name: s
        for s in build_storage_provider_statuses(
            configured_providers={"MinIO"},
            raw_allow_list="",
        )
    }

    # Assert
    assert statuses["minio"].available is True
    assert statuses["cos"].available is False


def test_allowed_flag_is_independent_of_availability() -> None:
    # Arrange / Act
    statuses = {
        s.name: s
        for s in build_storage_provider_statuses(
            configured_providers={"cos"},
            raw_allow_list="minio",
        )
    }

    # Assert — cos is configured but not permitted
    assert statuses["cos"].available is True
    assert statuses["cos"].allowed is False
    assert statuses["minio"].allowed is True
    assert statuses["minio"].available is False
