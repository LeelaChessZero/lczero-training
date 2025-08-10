#!/usr/bin/env python3
"""Test script for the YAML configuration parser."""

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List

# Add the src and build directories to Python path
src_dir = Path(__file__).parent / "src"
build_dir = Path(__file__).parent / "builddir"

if src_dir.exists():
    sys.path.insert(0, str(src_dir))
if build_dir.exists():
    sys.path.insert(0, str(build_dir))

try:
    from lczero_training.config.yaml_parser import from_yaml_file, ConfigParseError
    print("✓ Successfully imported YAML parser components")
except ImportError as e:
    print(f"✗ Failed to import YAML parser: {e}")
    sys.exit(1)


# Test dataclasses - independent of production config
@dataclass
class SimpleConfig:
    name: str
    value: int


@dataclass
class NestedConfig:
    inner: SimpleConfig
    count: int = 10


@dataclass
class ListConfig:
    items: List[SimpleConfig]
    enabled: bool = True


@dataclass
class ComplexConfig:
    nested: NestedConfig
    simple: SimpleConfig
    items: List[SimpleConfig]
    description: str = "default"


def create_test_yaml(content: str) -> str:
    """Create a temporary YAML file with given content."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(content)
        return f.name


def test_simple_dataclass():
    """Test parsing simple dataclass."""
    yaml_content = """
name: "test_name"
value: 42
"""

    yaml_file = None
    try:
        yaml_file = create_test_yaml(yaml_content)
        config = from_yaml_file(yaml_file, SimpleConfig)

        print("✓ Successfully parsed simple dataclass")
        print(f"  Name: {config.name}")
        print(f"  Value: {config.value}")

        assert config.name == "test_name"
        assert config.value == 42
        return True
    except Exception as e:
        print(f"✗ Failed to parse simple dataclass: {e}")
        return False
    finally:
        if yaml_file:
            os.unlink(yaml_file)


def test_nested_dataclass():
    """Test parsing nested dataclass."""
    yaml_content = """
inner:
  name: "nested_name"
  value: 100
count: 5
"""

    yaml_file = None
    try:
        yaml_file = create_test_yaml(yaml_content)
        config = from_yaml_file(yaml_file, NestedConfig)

        print("✓ Successfully parsed nested dataclass")
        print(f"  Inner name: {config.inner.name}")
        print(f"  Inner value: {config.inner.value}")
        print(f"  Count: {config.count}")

        assert config.inner.name == "nested_name"
        assert config.inner.value == 100
        assert config.count == 5
        return True
    except Exception as e:
        print(f"✗ Failed to parse nested dataclass: {e}")
        return False
    finally:
        if yaml_file:
            os.unlink(yaml_file)


def test_list_dataclass():
    """Test parsing dataclass with list field."""
    yaml_content = """
items:
  - name: "item1"
    value: 10
  - name: "item2"
    value: 20
enabled: false
"""

    yaml_file = None
    try:
        yaml_file = create_test_yaml(yaml_content)
        config = from_yaml_file(yaml_file, ListConfig)

        print("✓ Successfully parsed list dataclass")
        print(f"  Items count: {len(config.items)}")
        print(f"  First item: {config.items[0].name} = {config.items[0].value}")
        print(f"  Second item: {config.items[1].name} = {config.items[1].value}")
        print(f"  Enabled: {config.enabled}")

        assert len(config.items) == 2
        assert config.items[0].name == "item1"
        assert config.items[0].value == 10
        assert config.items[1].name == "item2"
        assert config.items[1].value == 20
        assert config.enabled == False
        return True
    except Exception as e:
        print(f"✗ Failed to parse list dataclass: {e}")
        return False
    finally:
        if yaml_file:
            os.unlink(yaml_file)


def test_complex_structure():
    """Test parsing complex nested structure."""
    yaml_content = """
nested:
  inner:
    name: "deep_nested"
    value: 999
  count: 3
simple:
  name: "simple_item"
  value: 123
items:
  - name: "list_item1"
    value: 50
  - name: "list_item2"
    value: 75
description: "complex test"
"""

    yaml_file = None
    try:
        yaml_file = create_test_yaml(yaml_content)
        config = from_yaml_file(yaml_file, ComplexConfig)

        print("✓ Successfully parsed complex structure")
        print(f"  Nested inner: {config.nested.inner.name}")
        print(f"  Simple: {config.simple.name}")
        print(f"  Items: {len(config.items)}")
        print(f"  Description: {config.description}")

        assert config.nested.inner.name == "deep_nested"
        assert config.simple.value == 123
        assert len(config.items) == 2
        assert config.description == "complex test"
        return True
    except Exception as e:
        print(f"✗ Failed to parse complex structure: {e}")
        return False
    finally:
        if yaml_file:
            os.unlink(yaml_file)


def test_missing_required_field():
    """Test error handling for missing required fields."""
    yaml_content = """
name: "test"
# Missing required 'value' field
"""

    yaml_file = None
    try:
        yaml_file = create_test_yaml(yaml_content)
        config = from_yaml_file(yaml_file, SimpleConfig)
        print("✗ Should have failed due to missing required field")
        return False
    except ConfigParseError as e:
        if "missing" in str(e).lower() and "value" in str(e):
            print(f"✓ Correctly caught missing field error: {e}")
            return True
        else:
            print(f"✗ Wrong error message for missing field: {e}")
            return False
    except Exception as e:
        print(f"✗ Unexpected error type: {e}")
        return False
    finally:
        if yaml_file:
            os.unlink(yaml_file)


def test_unknown_field():
    """Test error handling for unknown fields."""
    yaml_content = """
name: "test"
value: 42
unknown_field: "should fail"
"""

    yaml_file = None
    try:
        yaml_file = create_test_yaml(yaml_content)
        config = from_yaml_file(yaml_file, SimpleConfig)
        print("✗ Should have failed due to unknown field")
        return False
    except ConfigParseError as e:
        if ("unknown_field" in str(e) and 
            "root.unknown_field" in str(e)):
            print(f"✓ Correctly caught unknown field error: {e}")
            return True
        else:
            print(f"✗ Wrong error message format: {e}")
            return False
    except Exception as e:
        print(f"✗ Unexpected error type: {e}")
        return False
    finally:
        if yaml_file:
            os.unlink(yaml_file)


def test_type_mismatch():
    """Test error handling for type mismatches."""
    yaml_content = """
name: "test"
value: "should be int not string"
"""

    yaml_file = None
    try:
        yaml_file = create_test_yaml(yaml_content)
        config = from_yaml_file(yaml_file, SimpleConfig)
        print("✗ Should have failed due to type mismatch")
        return False
    except ConfigParseError as e:
        if "root.value" in str(e) and "int" in str(e):
            print(f"✓ Correctly caught type mismatch error: {e}")
            return True
        else:
            print(f"✗ Wrong error for type mismatch: {e}")
            return False
    except Exception as e:
        print(f"✗ Unexpected error type: {e}")
        return False
    finally:
        if yaml_file:
            os.unlink(yaml_file)


def test_nested_path_error():
    """Test error path reporting for nested structures."""
    yaml_content = """
inner:
  name: "test"
  value: "should be int"  # Type error in nested structure
count: 5
"""

    yaml_file = None
    try:
        yaml_file = create_test_yaml(yaml_content)
        config = from_yaml_file(yaml_file, NestedConfig)
        print("✗ Should have failed due to nested type mismatch")
        return False
    except ConfigParseError as e:
        if "root.inner.value" in str(e):
            print(f"✓ Correctly reported nested path error: {e}")
            return True
        else:
            print(f"✗ Wrong path in nested error: {e}")
            return False
    except Exception as e:
        print(f"✗ Unexpected error type: {e}")
        return False
    finally:
        if yaml_file:
            os.unlink(yaml_file)


def test_invalid_yaml_syntax():
    """Test error handling for invalid YAML syntax."""
    yaml_content = """
name: "test"
invalid: yaml: syntax: here
"""

    yaml_file = None
    try:
        yaml_file = create_test_yaml(yaml_content)
        config = from_yaml_file(yaml_file, SimpleConfig)
        print("✗ Should have failed due to invalid YAML syntax")
        return False
    except ConfigParseError as e:
        if "Invalid YAML syntax" in str(e):
            print(f"✓ Correctly caught YAML syntax error: {e}")
            return True
        else:
            print(f"✗ Wrong error message for YAML syntax: {e}")
            return False
    except Exception as e:
        print(f"✗ Unexpected error type: {e}")
        return False
    finally:
        if yaml_file:
            os.unlink(yaml_file)


def main():
    """Run all tests."""
    print("Testing YAML configuration parser...")
    print("=" * 50)

    tests = [
        test_simple_dataclass,
        test_nested_dataclass,
        test_list_dataclass,
        test_complex_structure,
        test_missing_required_field,
        test_unknown_field,
        test_type_mismatch,
        test_nested_path_error,
        test_invalid_yaml_syntax,
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
        print("🎉 All YAML parser tests passed!")
        return 0
    else:
        print("❌ Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())