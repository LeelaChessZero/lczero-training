# ABOUTME: Generic YAML to dataclass parser with validation and error reporting.
# ABOUTME: Provides type-safe YAML configuration loading with detailed error messages.

from dataclasses import fields, is_dataclass
from typing import Any, Type, TypeVar, Union
import yaml

T = TypeVar("T")


class ConfigParseError(Exception):
    """Exception raised when YAML configuration parsing fails."""

    def __init__(self, message: str, path: str = ""):
        if path:
            super().__init__(f"{message} at path '{path}'")
        else:
            super().__init__(message)
        self.path = path


def from_yaml_file(file_path: str, dataclass_type: Type[T]) -> T:
    """Parse YAML file into a dataclass instance.

    Args:
        file_path: Path to YAML configuration file
        dataclass_type: Target dataclass type to parse into

    Returns:
        Instance of dataclass_type populated from YAML

    Raises:
        ConfigParseError: If parsing fails due to validation errors
        FileNotFoundError: If YAML file doesn't exist
        yaml.YAMLError: If YAML syntax is invalid
    """
    try:
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        raise ConfigParseError(f"Configuration file not found: {file_path}")
    except yaml.YAMLError as e:
        raise ConfigParseError(f"Invalid YAML syntax: {e}")

    if not isinstance(data, dict):
        raise ConfigParseError("YAML root must be a dictionary")

    try:
        return _dict_to_dataclass(data, dataclass_type, "root")
    except ConfigParseError:
        raise
    except Exception as e:
        raise ConfigParseError(f"Unexpected parsing error: {e}")


def _dict_to_dataclass(data: dict, dataclass_type: Type, path: str) -> Any:
    """Recursively convert dictionary to dataclass instance.

    Args:
        data: Dictionary containing configuration data
        dataclass_type: Target dataclass type
        path: Current field path for error reporting

    Returns:
        Instance of dataclass_type

    Raises:
        ConfigParseError: If validation fails
    """
    if not is_dataclass(dataclass_type):
        return data

    # Get expected fields from dataclass
    expected_fields = {field.name: field for field in fields(dataclass_type)}

    # Check for unknown fields in YAML
    unknown_fields = set(data.keys()) - set(expected_fields.keys())
    if unknown_fields:
        unknown_field = next(iter(unknown_fields))
        raise ConfigParseError(
            f"Unknown field '{unknown_field}'", f"{path}.{unknown_field}"
        )

    # Convert each field
    converted_fields = {}
    for field_name, field_info in expected_fields.items():
        field_path = f"{path}.{field_name}"

        if field_name in data:
            field_value = data[field_name]
            converted_fields[field_name] = _convert_field_value(
                field_value, field_info.type, field_path
            )

    # Create dataclass instance (this will validate required fields automatically)
    try:
        return dataclass_type(**converted_fields)
    except TypeError as e:
        # Re-raise with path context for missing required fields
        raise ConfigParseError(str(e), path)


def _convert_field_value(
    value: Any, field_type: Union[Type, Any], path: str
) -> Any:
    """Convert a field value to the expected type.

    Args:
        value: Value from YAML
        field_type: Expected field type
        path: Current field path for error reporting

    Returns:
        Converted value

    Raises:
        ConfigParseError: If conversion fails
    """
    # Handle dataclass types
    if is_dataclass(field_type):
        if not isinstance(value, dict):
            raise ConfigParseError(
                f"Expected dictionary for dataclass field, got {type(value).__name__}",
                path,
            )
        return _dict_to_dataclass(value, field_type, path)

    # Handle list types
    if hasattr(field_type, "__origin__") and field_type.__origin__ is list:
        if not isinstance(value, list):
            raise ConfigParseError(
                f"Expected list, got {type(value).__name__}", path
            )

        list_item_type = field_type.__args__[0]
        converted_list = []
        for i, item in enumerate(value):
            item_path = f"{path}[{i}]"
            converted_item = _convert_field_value(
                item, list_item_type, item_path
            )
            converted_list.append(converted_item)
        return converted_list

    # Handle primitive types (int, str, bool, float)
    # Let Python handle type coercion naturally, but validate basic compatibility
    if field_type in (int, str, bool, float):
        if field_type is bool and not isinstance(value, bool):
            # YAML can parse "true"/"false" as bool, but protect against other types
            if value not in (True, False, "true", "false", "True", "False"):
                raise ConfigParseError(
                    f"Expected boolean, got {type(value).__name__}: {value}",
                    path,
                )
            return (
                bool(value)
                if isinstance(value, bool)
                else value.lower() == "true"
            )

        try:
            return field_type(value)
        except (ValueError, TypeError) as e:
            raise ConfigParseError(
                f"Cannot convert {type(value).__name__} to {field_type.__name__}: {e}",
                path,
            )

    # For other types, return as-is and let dataclass constructor validate
    return value
