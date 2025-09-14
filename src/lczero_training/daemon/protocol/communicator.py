# ABOUTME: Core Communicator class for JSONL IPC between processes.
# ABOUTME: Handles serialization/deserialization and message dispatch via stdin/stdout.

import json
import types
from dataclasses import is_dataclass
from enum import Enum
from typing import Any, TextIO, Union, get_args, get_origin

import anyio
from anyio.streams.text import TextReceiveStream, TextSendStream
from google.protobuf.json_format import MessageToDict, ParseDict
from google.protobuf.message import Message

from .registry import CLASS_TO_TYPE_MAP, TYPE_TO_CLASS_MAP


def _to_serializable(obj: Any) -> Any:
    """Convert dataclass/protobuf objects to JSON-serializable dicts."""
    if isinstance(obj, Message):
        return MessageToDict(
            obj, preserving_proto_field_name=True, use_integers_for_enums=True
        )
    elif isinstance(obj, Enum):
        return obj.value
    elif is_dataclass(obj):
        return {
            f.name: _to_serializable(getattr(obj, f.name))
            for f in obj.__dataclass_fields__.values()
            if getattr(obj, f.name) is not None
        }
    elif isinstance(obj, (list, tuple)):
        return [_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    else:
        return obj


def _unwrap_optional(t: Any) -> Any:
    """Extract T from T | None or Union[T, None]."""
    if isinstance(t, types.UnionType) or get_origin(t) is Union:
        args = [a for a in get_args(t) if a is not type(None)]
        return args[0] if len(args) == 1 else t
    return t


def _is_protobuf(cls: type) -> bool:
    """Check if cls is a protobuf Message class."""
    try:
        return isinstance(cls, type) and issubclass(cls, Message)
    except TypeError:
        return False


def _from_serializable(cls: type, data: Any) -> Any:
    """Reconstruct dataclass/protobuf from dict."""
    if _is_protobuf(cls):
        instance = cls()
        ParseDict(data, instance)
        return instance

    if not is_dataclass(cls):
        return data

    args = {}
    for field in cls.__dataclass_fields__.values():
        if field.name not in data:
            continue

        value = data[field.name]
        field_type = _unwrap_optional(field.type)

        if get_origin(field_type) is list:
            item_type = get_args(field_type)[0]
            if is_dataclass(item_type) or _is_protobuf(item_type):
                value = [_from_serializable(item_type, item) for item in value]
        elif is_dataclass(field_type) or _is_protobuf(field_type):
            value = _from_serializable(field_type, value)
        elif isinstance(field_type, type) and issubclass(field_type, Enum):
            # Convert string value back to enum
            value = field_type(value)

        args[field.name] = value

    return cls(**args)


class Communicator:
    def __init__(
        self, handler: Any, input_stream: TextIO, output_stream: TextIO
    ) -> None:
        """
        Initializes the Communicator.

        Args:
            handler: An object with `on_<event_type>` methods.
            input_stream: A file-like object to read incoming messages from (e.g., sys.stdin).
            output_stream: A file-like object to write outgoing messages to (e.g., sys.stdout).
        """
        self.handler = handler
        self.input = input_stream
        self.output = output_stream

    def send(self, payload_instance: Any) -> None:
        """
        Serializes and sends a payload object as a notification.
        The event type is automatically looked up from the registry.
        """
        payload_cls = type(payload_instance)
        event_type = CLASS_TO_TYPE_MAP.get(payload_cls)

        if event_type is None:
            raise TypeError(
                f"Object of type {payload_cls.__name__} is not a registered payload."
            )

        payload_dict = _to_serializable(payload_instance)
        message = {"type": event_type, "payload": payload_dict}

        json.dump(message, self.output)
        self.output.write("\n")
        self.output.flush()

    def run(self) -> None:
        """
        Starts the blocking listener loop.
        Reads from the input stream line-by-line, deserializes notifications,
        and dispatches them to the appropriate handler method.

        This method blocks until the input stream is closed.
        """
        for line in self.input:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            event_type = data["type"]
            payload_dict = data["payload"]

            payload_cls = TYPE_TO_CLASS_MAP[event_type]
            payload_instance = _from_serializable(payload_cls, payload_dict)

            handler_method_name = f"on_{event_type}"
            handler_method = getattr(self.handler, handler_method_name)

            handler_method(payload_instance)


class AsyncCommunicator:
    def __init__(
        self,
        handler: Any,
        input_stream: TextReceiveStream,
        output_stream: TextSendStream,
    ) -> None:
        """
        Initializes the AsyncCommunicator.

        Args:
            handler: An object with async `on_<event_type>` methods.
            input_stream: A TextReceiveStream to read incoming messages from.
            output_stream: A TextSendStream to write outgoing messages to.
        """
        self.handler = handler
        self.input_stream = input_stream
        self.output_stream = output_stream

    async def send(self, payload_instance: Any) -> None:
        """
        Serializes and sends a payload object as a notification.
        The event type is automatically looked up from the registry.
        """
        payload_cls = type(payload_instance)
        event_type = CLASS_TO_TYPE_MAP.get(payload_cls)

        if event_type is None:
            raise TypeError(
                f"Object of type {payload_cls.__name__} is not a registered payload."
            )

        payload_dict = _to_serializable(payload_instance)
        message = {"type": event_type, "payload": payload_dict}

        message_line = json.dumps(message) + "\n"
        await self.output_stream.send(message_line)

    async def run(self) -> None:
        """
        Starts the async listener loop.
        Reads from the input stream line-by-line, deserializes notifications,
        and dispatches them to the appropriate async handler method as tasks.

        This method runs until the input stream is closed.
        """
        async with anyio.create_task_group() as task_group:
            async for line in self.input_stream:
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)
                event_type = data["type"]
                payload_dict = data["payload"]

                payload_cls = TYPE_TO_CLASS_MAP[event_type]
                payload_instance = _from_serializable(payload_cls, payload_dict)

                handler_method_name = f"on_{event_type}"
                handler_method = getattr(self.handler, handler_method_name)

                task_group.start_soon(handler_method, payload_instance)
