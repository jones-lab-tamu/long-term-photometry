"""
Knob schema introspection for photometry_pipeline.config.Config.

Provides functions to auto-discover Config fields, types, and defaults
from the dataclass definition. No PySide6 dependency.
"""

import dataclasses
import typing
from typing import Any, Dict, get_type_hints


def get_config_field_specs() -> Dict[str, dict]:
    """Introspect photometry_pipeline.config.Config and return field specs.

    Returns a mapping of field_name -> {
        "name": str,
        "type": <type annotation>,
        "default": <default value or dataclasses.MISSING>,
        "has_default": bool,
        "optional": bool,  # True if Optional[T] or Union[..., None]
    }
    """
    from photometry_pipeline.config import Config

    hints = get_type_hints(Config)
    specs = {}

    for f in dataclasses.fields(Config):
        annotation = hints.get(f.name, f.type)

        has_default = (
            f.default is not dataclasses.MISSING
            or f.default_factory is not dataclasses.MISSING
        )
        default = (
            f.default if f.default is not dataclasses.MISSING
            else f.default_factory() if f.default_factory is not dataclasses.MISSING
            else dataclasses.MISSING
        )

        specs[f.name] = {
            "name": f.name,
            "type": annotation,
            "default": default,
            "has_default": has_default,
            "optional": _is_optional(annotation),
        }

    return specs


def is_config_key(name: str) -> bool:
    """Return True iff name is a field in Config."""
    from photometry_pipeline.config import Config
    return name in {f.name for f in dataclasses.fields(Config)}


def normalize_type(annotation) -> dict:
    """Return a normalized representation of a type annotation for UI building.

    Returns {
        "kind": "int"|"float"|"bool"|"str"|"enum"|"literal"|"optional"|"union"|"unknown",
        "choices": [...],     # for Literal, enum, union
        "inner": <normalized>,  # for Optional
    }
    """
    import enum

    # Handle None / NoneType
    if annotation is type(None):
        return {"kind": "unknown"}

    # Handle Enum classes
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        # Prefer serialized scalar values to match YAML expectations
        choices = []
        for member in annotation:
            val = member.value
            if isinstance(val, (str, int)):
                choices.append(val)
            else:
                choices.append(member.name)
        return {
            "kind": "enum",
            "choices": choices,
        }

    # Handle Optional[T] and Union[T1, T2]
    origin = typing.get_origin(annotation)
    if origin is typing.Union:
        if _is_optional(annotation):
            inner = _unwrap_optional(annotation)
            return {
                "kind": "optional",
                "inner": normalize_type(inner),
            }
        else:
            args = typing.get_args(annotation)
            return {
                "kind": "union",
                "choices": [normalize_type(a) for a in args],
            }

    # Handle Literal["a", "b"]
    args = typing.get_args(annotation)
    if origin is typing.Literal:
        return {
            "kind": "literal",
            "choices": list(args),
        }

    # Handle basic types
    if annotation is int:
        return {"kind": "int"}
    if annotation is float:
        return {"kind": "float"}
    if annotation is bool:
        return {"kind": "bool"}
    if annotation is str:
        return {"kind": "str"}

    return {"kind": "unknown"}


def _is_optional(annotation) -> bool:
    """Check if annotation is Optional[T] (i.e. Union[T, None])."""
    origin = typing.get_origin(annotation)
    if origin is typing.Union:
        args = typing.get_args(annotation)
        return type(None) in args
    return False


def _unwrap_optional(annotation):
    """Given Optional[T], return T."""
    args = typing.get_args(annotation)
    non_none = [a for a in args if a is not type(None)]
    if len(non_none) == 1:
        return non_none[0]
    # Multi-type union minus None — return the union itself
    return typing.Union[tuple(non_none)]
