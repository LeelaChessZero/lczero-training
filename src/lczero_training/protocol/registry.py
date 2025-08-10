# ABOUTME: Registry system for mapping event type strings to payload dataclasses.
# ABOUTME: Provides @register decorator and maintains bidirectional mapping dicts.

import inspect

# These maps will be populated by the @register decorator
TYPE_TO_CLASS_MAP = {}
CLASS_TO_TYPE_MAP = {}


def register(event_type: str):
    """A decorator to register a payload dataclass with its event type string."""

    def decorator(cls):
        if not inspect.isclass(cls):
            raise TypeError(
                "The @register decorator can only be used on classes."
            )

        if event_type in TYPE_TO_CLASS_MAP:
            raise ValueError(
                f"Event type '{event_type}' is already registered."
            )

        if cls in CLASS_TO_TYPE_MAP:
            raise ValueError(f"Class '{cls.__name__}' is already registered.")

        TYPE_TO_CLASS_MAP[event_type] = cls
        CLASS_TO_TYPE_MAP[cls] = event_type
        return cls

    return decorator
