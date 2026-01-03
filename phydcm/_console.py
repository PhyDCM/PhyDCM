"""
Console utilities for consistent, academic-grade terminal messaging.

This module intentionally avoids heavy dependencies. ANSI colors are used when
supported; on Windows, optional Colorama initialization is attempted.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

@dataclass(frozen=True)
class _Ansi:
    reset: str = "\033[0m"
    red: str = "\033[31m"
    green: str = "\033[32m"
    yellow: str = "\033[33m"
    cyan: str = "\033[36m"
    gray: str = "\033[90m"

def _supports_ansi() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if sys.platform != "win32":
        return True
    # Windows: try to enable ANSI via Colorama if available.
    try:
        import colorama  # type: ignore
        colorama.just_fix_windows_console()
        return True
    except Exception:
        return False

_ANSI_ENABLED = _supports_ansi()
_A = _Ansi()

def style(text: str, color: str | None = None, bold: bool = False) -> str:
    if not _ANSI_ENABLED or not color:
        return text
    prefix = ""
    if bold:
        prefix += "\033[1m"
    prefix += getattr(_A, color, "")
    return f"{prefix}{text}{_A.reset}"

def info(msg: str) -> None:
    print(style(msg, "cyan"))

def success(msg: str) -> None:
    print(style(msg, "green"))

def warn(msg: str) -> None:
    print(style(msg, "yellow"))

def error(msg: str) -> None:
    print(style(msg, "red"))

def debug(msg: str) -> None:
    if os.environ.get("PHYDCM_DEBUG"):
        print(style(msg, "gray"))
