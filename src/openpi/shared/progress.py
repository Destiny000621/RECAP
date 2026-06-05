"""Minimal tqdm-progress helpers used by pistar's train_value.py /
label_advantage_from_vlm.py.

This file ships missing on upstream pistar (`from openpi.shared import progress`
fails). The only consumer is `sync_pbar_color(pbar)` — a no-op stub here.
"""

from __future__ import annotations

from typing import Any


def sync_pbar_color(pbar: Any) -> None:  # noqa: ARG001
    """No-op stub. Upstream pistar's intent was probably to recolor a tqdm
    progress bar based on training state, but the implementation isn't
    shipped on main. A no-op is safe — pbars continue to render normally."""
