"""Test script for the protocol registry system."""

import pytest
from dataclasses import dataclass

from lczero_training.protocol.registry import (
    register,
    TYPE_TO_CLASS_MAP,
    CLASS_TO_TYPE_MAP,
)


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear registry maps before each test."""
    TYPE_TO_CLASS_MAP.clear()
    CLASS_TO_TYPE_MAP.clear()
    yield
    TYPE_TO_CLASS_MAP.clear()
    CLASS_TO_TYPE_MAP.clear()


def test_basic_registration():
    """Test basic event type registration."""

    @register("test_event")
    @dataclass
    class BasicPayload:
        content: str

    # Check forward mapping
    assert TYPE_TO_CLASS_MAP["test_event"] == BasicPayload
    # Check reverse mapping
    assert CLASS_TO_TYPE_MAP[BasicPayload] == "test_event"


def test_duplicate_event_type():
    """Test that duplicate event types are rejected."""

    @register("duplicate_event")
    @dataclass
    class FirstPayload:
        data: str

    with pytest.raises(
        ValueError, match=r".*duplicate_event.*already registered.*"
    ):

        @register("duplicate_event")  # Should fail
        @dataclass
        class SecondPayload:
            other_data: int


def test_duplicate_class():
    """Test that duplicate classes are rejected."""

    @dataclass
    class PayloadClass:
        data: str

    # Register once
    register("first_event")(PayloadClass)

    # Try to register same class again - should fail
    with pytest.raises(
        ValueError, match=r".*PayloadClass.*already registered.*"
    ):
        register("second_event")(PayloadClass)


def test_non_class_registration():
    """Test that non-classes are rejected."""
    with pytest.raises(TypeError, match=r".*can only be used on classes.*"):
        # Try to register a string instead of a class
        @register("invalid_event")
        def not_a_class():
            pass


def test_multiple_registrations():
    """Test multiple valid registrations work correctly."""

    @register("event_one")
    @dataclass
    class PayloadOne:
        data: str

    @register("event_two")
    @dataclass
    class PayloadTwo:
        value: int

    @register("event_three")
    @dataclass
    class PayloadThree:
        items: list

    # Check all mappings exist
    assert TYPE_TO_CLASS_MAP["event_one"] == PayloadOne
    assert TYPE_TO_CLASS_MAP["event_two"] == PayloadTwo
    assert TYPE_TO_CLASS_MAP["event_three"] == PayloadThree

    assert CLASS_TO_TYPE_MAP[PayloadOne] == "event_one"
    assert CLASS_TO_TYPE_MAP[PayloadTwo] == "event_two"
    assert CLASS_TO_TYPE_MAP[PayloadThree] == "event_three"

    # Check we have exactly 3 entries in each map
    assert len(TYPE_TO_CLASS_MAP) == 3
    assert len(CLASS_TO_TYPE_MAP) == 3


def test_registry_persistence():
    """Test that registry persists across imports."""

    @register("persistent_event")
    @dataclass
    class PersistentPayload:
        data: str

    # Re-import the module
    from lczero_training.protocol.registry import (
        TYPE_TO_CLASS_MAP as imported_type_map,
    )
    from lczero_training.protocol.registry import (
        CLASS_TO_TYPE_MAP as imported_class_map,
    )

    # Check the registration persists
    assert imported_type_map["persistent_event"] == PersistentPayload
    assert imported_class_map[PersistentPayload] == "persistent_event"
