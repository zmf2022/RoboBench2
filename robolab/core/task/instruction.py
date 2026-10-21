# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Instruction-field selection and policy-use tracking."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TypeAlias

InstructionInput: TypeAlias = str | Mapping[str, str]


def normalize_instruction_fields(instruction: InstructionInput) -> dict[str, str]:
    """Return a validated, insertion-ordered instruction-field dictionary."""
    if isinstance(instruction, str):
        if not instruction.strip():
            raise ValueError("instruction must be a nonempty string")
        return {"default": instruction}
    if not isinstance(instruction, Mapping) or not instruction:
        raise ValueError("instruction must be a nonempty string or mapping")
    fields = dict(instruction)
    for key, value in fields.items():
        if not isinstance(key, str) or not key:
            raise ValueError("instruction field names must be nonempty strings")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"instruction field '{key}' must be a nonempty string")
    return fields


def select_instruction_fields(
    instruction: InstructionInput,
    requested: str | Iterable[str] | None = None,
) -> dict[str, str]:
    """Return all fields, or only the explicitly requested fields.

    Plain-string tasks remain compatible with the historical behavior: any
    requested instruction type still resolves to their sole ``default`` field.
    """
    was_string = isinstance(instruction, str)
    fields = normalize_instruction_fields(instruction)
    if requested is None:
        return fields
    names = [requested] if isinstance(requested, str) else list(requested)
    if not names:
        raise ValueError("at least one instruction field must be requested")
    if len(set(names)) != len(names):
        raise ValueError(f"instruction fields must be unique: {names}")
    if was_string:
        return fields
    missing = [name for name in names if name not in fields]
    if missing:
        raise ValueError(
            f"Instruction field(s) not found: {missing}. Available: {list(fields)}"
        )
    return {name: fields[name] for name in names}


def primary_instruction(instruction: InstructionInput) -> tuple[str, str]:
    """Return ``(field_name, value)`` for legacy display and filenames."""
    fields = normalize_instruction_fields(instruction)
    name = "default" if "default" in fields else next(iter(fields))
    return name, fields[name]


class TrackedInstructionFields(dict[str, str]):
    """Read-only dict that records which values a policy client retrieves.

    Task authors only define ordinary dictionaries. The episode runner wraps a
    selected dictionary with this class immediately before policy inference.
    """

    def __init__(self, fields: InstructionInput):
        super().__init__(normalize_instruction_fields(fields))
        self._used: set[str] = set()

    def _mark(self, key: str) -> None:
        if dict.__contains__(self, key):
            self._used.add(key)

    def _mark_all(self) -> None:
        self._used.update(dict.keys(self))

    def __getitem__(self, key: str) -> str:
        value = dict.__getitem__(self, key)
        self._mark(key)
        return value

    def get(self, key: str, default=None):
        if not dict.__contains__(self, key):
            return default
        return self[key]

    def items(self):
        self._mark_all()
        return dict.items(self)

    def values(self):
        self._mark_all()
        return dict.values(self)

    def copy(self) -> dict[str, str]:
        self._mark_all()
        return dict(self)

    @property
    def instruction_fields(self) -> list[str]:
        return list(dict.keys(self))

    @property
    def instruction_fields_used(self) -> list[str]:
        return [key for key in dict.keys(self) if key in self._used]

    @staticmethod
    def _readonly(*_args, **_kwargs):
        raise TypeError("instruction fields are read-only during policy inference")

    __setitem__ = _readonly
    __delitem__ = _readonly
    clear = _readonly
    pop = _readonly
    popitem = _readonly
    setdefault = _readonly
    update = _readonly
    __ior__ = _readonly


def instruction_text(instruction: InstructionInput) -> str:
    """Return the primary string, tracking access when passed a tracked dict."""
    if isinstance(instruction, str):
        return instruction
    if not instruction:
        raise ValueError("instruction mapping must be nonempty")
    name = "default" if "default" in instruction else next(iter(instruction))
    value = instruction[name]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"instruction field '{name}' must be a nonempty string")
    return value
