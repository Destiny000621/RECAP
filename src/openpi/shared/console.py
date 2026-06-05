"""Minimal console-formatting helpers used by pistar's train_value.py /
label_advantage_from_vlm.py.

This file ships missing on upstream pistar (`from openpi.shared import console`
fails). The helpers below produce ANSI-colored strings; consumers wrap log
messages with them.
"""

from __future__ import annotations

# ANSI SGR codes
_RESET = "\033[0m"
_BLUE = "\033[94m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_BOLD = "\033[1m"


def info(msg: object) -> str:
    return f"{_BLUE}ℹ {msg}{_RESET}"


def ok(msg: object) -> str:
    return f"{_GREEN}✓ {msg}{_RESET}"


def warn(msg: object) -> str:
    return f"{_YELLOW}⚠ {msg}{_RESET}"


def error(msg: object) -> str:
    return f"{_RED}✗ {msg}{_RESET}"


def bold(msg: object) -> str:
    return f"{_BOLD}{msg}{_RESET}"
