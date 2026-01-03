"""
Lazy import helpers to keep phydcm import time minimal.

The public API remains stable, while heavy optional dependencies (e.g.,
TensorFlow, pydicom) are imported only when required.
"""
from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

def optional_import(module_name: str) -> ModuleType | None:
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None

def require_import(module_name: str, purpose: str) -> ModuleType:
    mod = optional_import(module_name)
    if mod is None:
        raise ImportError(
            f"Optional dependency '{module_name}' is required for {purpose}, "
            "but it is not available in the current Python environment."
        )
    return mod

def has_module(module_name: str) -> bool:
    return optional_import(module_name) is not None
