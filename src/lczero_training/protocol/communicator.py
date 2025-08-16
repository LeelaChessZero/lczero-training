# ABOUTME: Core Communicator class for JSONL IPC between processes.
# ABOUTME: Handles serialization/deserialization and message dispatch via stdin/stdout.

import json
from dataclasses import is_dataclass, asdict
from typing import get_origin, get_args

import anyio
from anyio.streams.text import TextReceiveStream, TextSendStream

from .registry import TYPE_TO_CLASS_MAP, CLASS_TO_TYPE_MAP


def _from_dict(cls, data):
    """
    Recursively constructs a dataclass instance from a dictionary.
    Handles nested dataclasses and lists of dataclasses.
    """
    if not is_dataclass(cls):
        return data

    constructor_args = {}
    for field in cls.__dataclass_fields__.values():
        field_value = data.get(field.name)
        if field_value is None:
            continue

        # Handle lists of dataclasses
        origin_type = get_origin(field.type)
        if origin_type is list or origin_type is list:
            list_item_type = get_args(field.type)[0]
            if is_dataclass(list_item_type):
                constructor_args[field.name] = [
                    _from_dict(list_item_type, item) for item in field_value
                ]
            else:
                constructor_args[field.name] = field_value
        # Handle nested dataclasses
        elif is_dataclass(field.type):
            constructor_args[field.name] = _from_dict(field.type, field_value)
        # Handle primitives, dicts, etc.
        else:
            constructor_args[field.name] = field_value

    return cls(**constructor_args)


class Communicator:
    def __init__(self, handler, input_stream, output_stream):
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

    def send(self, payload_instance):
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

        payload_dict = asdict(payload_instance)
        message = {"type": event_type, "payload": payload_dict}

        json.dump(message, self.output)
        self.output.write("\n")
        self.output.flush()

    def run(self):
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
            payload_instance = _from_dict(payload_cls, payload_dict)

            handler_method_name = f"on_{event_type}"
            handler_method = getattr(self.handler, handler_method_name)

            handler_method(payload_instance)


class AsyncCommunicator:
    def __init__(
        self,
        handler,
        input_stream: TextReceiveStream,
        output_stream: TextSendStream,
    ):
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

    async def send(self, payload_instance):
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

        payload_dict = asdict(payload_instance)
        message = {"type": event_type, "payload": payload_dict}

        message_line = json.dumps(message) + "\n"
        await self.output_stream.send(message_line)

    async def run(self):
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
                payload_instance = _from_dict(payload_cls, payload_dict)

                handler_method_name = f"on_{event_type}"
                handler_method = getattr(self.handler, handler_method_name)

                task_group.start_task(handler_method, payload_instance)
