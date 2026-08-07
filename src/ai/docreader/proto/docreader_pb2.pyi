# mypy: disable-error-code="import-untyped,misc,type-arg"
from collections.abc import Iterable as _Iterable
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf.internal import containers as _containers

DESCRIPTOR: _descriptor.FileDescriptor

class ReadConfig(_message.Message):
    __slots__ = ("parser_engine", "parser_engine_overrides")
    class ParserEngineOverridesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: str | None = ..., value: str | None = ...) -> None: ...
    PARSER_ENGINE_FIELD_NUMBER: _ClassVar[int]
    PARSER_ENGINE_OVERRIDES_FIELD_NUMBER: _ClassVar[int]
    parser_engine: str
    parser_engine_overrides: _containers.ScalarMap[str, str]
    def __init__(self, parser_engine: str | None = ..., parser_engine_overrides: _Mapping[str, str] | None = ...) -> None: ...

class ReadRequest(_message.Message):
    __slots__ = ("config", "file_content", "file_name", "file_type", "request_id", "title", "url")
    FILE_CONTENT_FIELD_NUMBER: _ClassVar[int]
    FILE_NAME_FIELD_NUMBER: _ClassVar[int]
    FILE_TYPE_FIELD_NUMBER: _ClassVar[int]
    URL_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    file_content: bytes
    file_name: str
    file_type: str
    url: str
    title: str
    config: ReadConfig
    request_id: str
    def __init__(self, file_content: bytes | None = ..., file_name: str | None = ..., file_type: str | None = ..., url: str | None = ..., title: str | None = ..., config: ReadConfig | _Mapping | None = ..., request_id: str | None = ...) -> None: ...

class ImageRef(_message.Message):
    __slots__ = ("filename", "image_data", "mime_type", "original_ref", "storage_key")
    FILENAME_FIELD_NUMBER: _ClassVar[int]
    ORIGINAL_REF_FIELD_NUMBER: _ClassVar[int]
    MIME_TYPE_FIELD_NUMBER: _ClassVar[int]
    STORAGE_KEY_FIELD_NUMBER: _ClassVar[int]
    IMAGE_DATA_FIELD_NUMBER: _ClassVar[int]
    filename: str
    original_ref: str
    mime_type: str
    storage_key: str
    image_data: bytes
    def __init__(self, filename: str | None = ..., original_ref: str | None = ..., mime_type: str | None = ..., storage_key: str | None = ..., image_data: bytes | None = ...) -> None: ...

class ReadResponse(_message.Message):
    __slots__ = ("error", "image_dir_path", "image_refs", "markdown_content", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: str | None = ..., value: str | None = ...) -> None: ...
    MARKDOWN_CONTENT_FIELD_NUMBER: _ClassVar[int]
    IMAGE_REFS_FIELD_NUMBER: _ClassVar[int]
    IMAGE_DIR_PATH_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    markdown_content: str
    image_refs: _containers.RepeatedCompositeFieldContainer[ImageRef]
    image_dir_path: str
    metadata: _containers.ScalarMap[str, str]
    error: str
    def __init__(self, markdown_content: str | None = ..., image_refs: _Iterable[ImageRef | _Mapping] | None = ..., image_dir_path: str | None = ..., metadata: _Mapping[str, str] | None = ..., error: str | None = ...) -> None: ...

class ReadStreamMeta(_message.Message):
    __slots__ = ("error", "image_count", "image_dir_path", "markdown_content", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: str | None = ..., value: str | None = ...) -> None: ...
    MARKDOWN_CONTENT_FIELD_NUMBER: _ClassVar[int]
    IMAGE_DIR_PATH_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    IMAGE_COUNT_FIELD_NUMBER: _ClassVar[int]
    markdown_content: str
    image_dir_path: str
    metadata: _containers.ScalarMap[str, str]
    error: str
    image_count: int
    def __init__(self, markdown_content: str | None = ..., image_dir_path: str | None = ..., metadata: _Mapping[str, str] | None = ..., error: str | None = ..., image_count: int | None = ...) -> None: ...

class ReadStreamResponse(_message.Message):
    __slots__ = ("image", "meta")
    META_FIELD_NUMBER: _ClassVar[int]
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    meta: ReadStreamMeta
    image: ImageRef
    def __init__(self, meta: ReadStreamMeta | _Mapping | None = ..., image: ImageRef | _Mapping | None = ...) -> None: ...

class ListEnginesRequest(_message.Message):
    __slots__ = ("config_overrides",)
    class ConfigOverridesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: str | None = ..., value: str | None = ...) -> None: ...
    CONFIG_OVERRIDES_FIELD_NUMBER: _ClassVar[int]
    config_overrides: _containers.ScalarMap[str, str]
    def __init__(self, config_overrides: _Mapping[str, str] | None = ...) -> None: ...

class ParserEngineInfo(_message.Message):
    __slots__ = ("available", "description", "file_types", "name", "unavailable_reason")
    NAME_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    FILE_TYPES_FIELD_NUMBER: _ClassVar[int]
    AVAILABLE_FIELD_NUMBER: _ClassVar[int]
    UNAVAILABLE_REASON_FIELD_NUMBER: _ClassVar[int]
    name: str
    description: str
    file_types: _containers.RepeatedScalarFieldContainer[str]
    available: bool
    unavailable_reason: str
    def __init__(self, name: str | None = ..., description: str | None = ..., file_types: _Iterable[str] | None = ..., available: bool | None = ..., unavailable_reason: str | None = ...) -> None: ...

class ListEnginesResponse(_message.Message):
    __slots__ = ("engines",)
    ENGINES_FIELD_NUMBER: _ClassVar[int]
    engines: _containers.RepeatedCompositeFieldContainer[ParserEngineInfo]
    def __init__(self, engines: _Iterable[ParserEngineInfo | _Mapping] | None = ...) -> None: ...
