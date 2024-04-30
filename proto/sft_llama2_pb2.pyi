from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class FineTuneRequest(_message.Message):
    __slots__ = ("file_name", "data_file")
    FILE_NAME_FIELD_NUMBER: _ClassVar[int]
    DATA_FILE_FIELD_NUMBER: _ClassVar[int]
    file_name: str
    data_file: bytes
    def __init__(self, file_name: _Optional[str] = ..., data_file: _Optional[bytes] = ...) -> None: ...

class FineTuneReply(_message.Message):
    __slots__ = ("status",)
    STATUS_FIELD_NUMBER: _ClassVar[int]
    status: str
    def __init__(self, status: _Optional[str] = ...) -> None: ...

class InferenceRequest(_message.Message):
    __slots__ = ("prompt",)
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    prompt: str
    def __init__(self, prompt: _Optional[str] = ...) -> None: ...

class InferenceReply(_message.Message):
    __slots__ = ("response",)
    RESPONSE_FIELD_NUMBER: _ClassVar[int]
    response: str
    def __init__(self, response: _Optional[str] = ...) -> None: ...
