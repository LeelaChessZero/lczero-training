#!/usr/bin/env python3
"""Test script for the protocol registry system."""

import sys
from dataclasses import dataclass
from pathlib import Path

# Add the src directory to Python path
src_dir = Path(__file__).parent / "src"
if src_dir.exists():
    sys.path.insert(0, str(src_dir))

try:
    from lczero_training.protocol.registry import register, TYPE_TO_CLASS_MAP, CLASS_TO_TYPE_MAP
    print("✓ Successfully imported registry components")
except ImportError as e:
    print(f"✗ Failed to import registry: {e}")
    sys.exit(1)


@dataclass
class TestPayload:
    message: str
    value: int


@dataclass
class AnotherTestPayload:
    data: str


def test_basic_registration():
    """Test basic event type registration."""
    # Clear maps for clean test
    TYPE_TO_CLASS_MAP.clear()
    CLASS_TO_TYPE_MAP.clear()
    
    @register("test_event")
    @dataclass
    class BasicPayload:
        content: str
    
    try:
        # Check forward mapping
        assert TYPE_TO_CLASS_MAP["test_event"] == BasicPayload
        # Check reverse mapping
        assert CLASS_TO_TYPE_MAP[BasicPayload] == "test_event"
        print("✓ Basic registration works correctly")
        return True
    except Exception as e:
        print(f"✗ Basic registration failed: {e}")
        return False


def test_duplicate_event_type():
    """Test that duplicate event types are rejected."""
    TYPE_TO_CLASS_MAP.clear()
    CLASS_TO_TYPE_MAP.clear()
    
    @register("duplicate_event")
    @dataclass
    class FirstPayload:
        data: str
    
    try:
        @register("duplicate_event")  # Should fail
        @dataclass  
        class SecondPayload:
            other_data: int
        
        print("✗ Should have failed on duplicate event type")
        return False
    except ValueError as e:
        if "already registered" in str(e) and "duplicate_event" in str(e):
            print("✓ Correctly rejected duplicate event type")
            return True
        else:
            print(f"✗ Wrong error message for duplicate event type: {e}")
            return False
    except Exception as e:
        print(f"✗ Unexpected error type: {e}")
        return False


def test_duplicate_class():
    """Test that duplicate classes are rejected."""
    TYPE_TO_CLASS_MAP.clear()
    CLASS_TO_TYPE_MAP.clear()
    
    @dataclass
    class PayloadClass:
        data: str
    
    # Register once
    register("first_event")(PayloadClass)
    
    try:
        # Try to register same class again - should fail
        register("second_event")(PayloadClass)
        print("✗ Should have failed on duplicate class registration")
        return False
    except ValueError as e:
        if "already registered" in str(e) and "PayloadClass" in str(e):
            print("✓ Correctly rejected duplicate class")
            return True
        else:
            print(f"✗ Wrong error message for duplicate class: {e}")
            return False
    except Exception as e:
        print(f"✗ Unexpected error type: {e}")
        return False


def test_non_class_registration():
    """Test that non-classes are rejected."""
    TYPE_TO_CLASS_MAP.clear()
    CLASS_TO_TYPE_MAP.clear()
    
    try:
        # Try to register a string instead of a class
        @register("invalid_event")
        def not_a_class():
            pass
        
        print("✗ Should have failed on non-class registration")
        return False
    except TypeError as e:
        if "can only be used on classes" in str(e):
            print("✓ Correctly rejected non-class registration")
            return True
        else:
            print(f"✗ Wrong error message for non-class: {e}")
            return False
    except Exception as e:
        print(f"✗ Unexpected error type: {e}")
        return False


def test_multiple_registrations():
    """Test multiple valid registrations work correctly."""
    TYPE_TO_CLASS_MAP.clear()
    CLASS_TO_TYPE_MAP.clear()
    
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
    
    try:
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
        
        print("✓ Multiple registrations work correctly")
        return True
    except Exception as e:
        print(f"✗ Multiple registrations failed: {e}")
        return False


def test_registry_persistence():
    """Test that registry persists across imports."""
    TYPE_TO_CLASS_MAP.clear()
    CLASS_TO_TYPE_MAP.clear()
    
    @register("persistent_event")
    @dataclass
    class PersistentPayload:
        data: str
    
    try:
        # Re-import the module
        from lczero_training.protocol.registry import TYPE_TO_CLASS_MAP as imported_type_map
        from lczero_training.protocol.registry import CLASS_TO_TYPE_MAP as imported_class_map
        
        # Check the registration persists
        assert imported_type_map["persistent_event"] == PersistentPayload
        assert imported_class_map[PersistentPayload] == "persistent_event"
        
        print("✓ Registry persists across imports")
        return True
    except Exception as e:
        print(f"✗ Registry persistence failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing protocol registry system...")
    print("=" * 50)

    tests = [
        test_basic_registration,
        test_duplicate_event_type,
        test_duplicate_class,
        test_non_class_registration,
        test_multiple_registrations,
        test_registry_persistence,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        print(f"\nRunning {test.__name__}...")
        if test():
            passed += 1
        else:
            print(f"Test {test.__name__} failed!")

    print("\n" + "=" * 50)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All registry tests passed!")
        return 0
    else:
        print("❌ Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())