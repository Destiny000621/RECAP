"""Add per-arm RECAP labels to a YAM bimanual LeRobot v2.1 dataset.

Motivation
----------
The single, frame-level ``adv_ind`` token is applied to the *joint* 14-dim
action (left = dims 0..6, right = dims 7..13). During a DAgger correction the
operator teleoperates only ONE arm; the other arm is *frozen* (held still).
The old pipeline force-labels every intervention frame ``positive``, so the
frozen arm's static action is regressed as a "positive" demonstration and the
policy, conditioned on ``positive`` at inference, collapses to pausing.

This script writes the per-arm signals that fix that, WITHOUT re-collecting:

1. Corrected-arm detection (no annotation needed): group contiguous
   ``intervention==1`` runs and compare per-arm motion energy. The frozen arm
   has ~0 energy; the teleoperated arm has high energy. Validated on the
   reference dataset: ~95% of runs are cleanly one-armed (energy ratio < 0.25).

2. ``loss_mask_left`` / ``loss_mask_right`` (float32 in {0,1}): 0 for the frozen
   arm on a one-arm correction run, 1 otherwise. The flow-matching loss is
   masked per-arm so the frozen action contributes ZERO gradient — a hard
   backstop that makes it impossible to learn the frozen behavior.

3. ``adv_ind_left`` / ``adv_ind_right`` (string): per-arm advantage condition,
   scaffolded the way limb supplies Stage-3 ``adv_ind``:
     - autonomous frame            -> ``none`` / ``none``   (Stage 5 fills these)
     - correction, right-corrected -> right ``positive``, left ``none``
     - correction, left-corrected  -> left  ``positive``, right ``none``
     - correction, both-corrected  -> both  ``positive``

4. ``value_label_left`` / ``value_label_right`` (float32 in [-1,0]), only when an
   ``--annotation_json`` of per-arm vial-insertion frames is supplied:
       value_label_arm,t = clip(completed_arm(t)/N_arm - 1, -1, 0)
   With the usual N_arm == 1 this is binary: -1 before the arm seats its vial,
   0 after. These are the Stage-4 per-arm value-model targets. The N-step
   reward term in Stage 5 reuses the shared, arm-agnostic ``reward_label``.

5. ``corrected_arm`` (string in {none,left,right,both}): the detector verdict
   per frame, for inspection.

The dataset is modified IN PLACE (parquet under ``data/`` + ``meta/info.json``).
Run it on a *copy* so the Stage-3 / pre-VLM variant stays intact.

Usage
-----
  # 1) inspect what the detector would do, write nothing
  python scripts/add_per_arm_labels.py --data_dir <ds> --dry_run

  # 2) emit a skeleton annotation to fill (one insert frame per arm)
  python scripts/add_per_arm_labels.py --data_dir <ds> --emit_skeleton anns.json

  # 3) write masks + corrected_arm + adv scaffold (no value labels yet)
  python scripts/add_per_arm_labels.py --data_dir <ds>

  # 4) full: also write per-arm value labels from the filled annotation
  python scripts/add_per_arm_labels.py --data_dir <ds> --annotation_json anns.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Left/right action layout for YAM (6 joints + gripper each). gripper indices
# are 6 (left) and 13 (right); "joints" excludes them to avoid gripper-toggle
# noise in the motion-energy detector.
LEFT_SLICE = slice(0, 7)
RIGHT_SLICE = slice(7, 14)
LEFT_JOINTS = slice(0, 6)
RIGHT_JOINTS = slice(7, 13)


# --------------------------------------------------------------------------- #
# scalar / column helpers (mirror scripts/label_advantage_from_vlm.py)
# --------------------------------------------------------------------------- #
def _scalar_int(value: Any) -> int:
    return int(np.asarray(value).reshape(-1)[0])


def _scalar_float(value: Any) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def _int_col(df: pd.DataFrame, name: str) -> np.ndarray:
    return np.asarray([_scalar_int(v) for v in df[name].tolist()], dtype=np.int64)


def _action_matrix(df: pd.DataFrame, action_col: str) -> np.ndarray:
    return np.stack([np.asarray(a, dtype=np.float32).reshape(-1) for a in df[action_col].tolist()])


def _episode_id(df: pd.DataFrame, fallback: int) -> int:
    if "episode_index" in df.columns and len(df) > 0:
        return _scalar_int(df["episode_index"].iloc[0])
    return fallback


# --------------------------------------------------------------------------- #
# corrected-arm detection
# --------------------------------------------------------------------------- #
def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive [start, end] index ranges where ``mask`` is True."""
    runs: list[tuple[int, int]] = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def _classify_run(
    action: np.ndarray,
    s: int,
    e: int,
    *,
    use_joints: bool,
    ratio_threshold: float,
    min_run: int,
) -> str:
    """Classify a correction run as 'left' / 'right' / 'both'.

    Energy is the summed absolute frame-to-frame action delta per arm over the
    run. The frozen arm has ~0 energy; we mask whichever arm is far quieter.
    Ambiguous (both moving) or too-short runs return 'both' -> no masking, the
    conservative choice that never wrongly masks a moving arm.
    """
    if e - s + 1 < max(min_run, 2):
        return "both"
    seg = action[s : e + 1]
    lsl, rsl = (LEFT_JOINTS, RIGHT_JOINTS) if use_joints else (LEFT_SLICE, RIGHT_SLICE)
    el = float(np.abs(np.diff(seg[:, lsl], axis=0)).sum())
    er = float(np.abs(np.diff(seg[:, rsl], axis=0)).sum())
    hi = max(el, er)
    if hi <= 0.0:
        return "both"
    if min(el, er) / hi < ratio_threshold:
        return "left" if el > er else "right"
    return "both"


def _release_candidates(gripper: np.ndarray, T: int, *, hold_frames: int = 30, startup_frac: float = 0.05) -> list[int]:
    """Frames where the gripper opens after a sustained closed hold (vial release).

    A cheap proprioceptive prior for insert timing; superseded by vision-based
    detection but kept as a fallback / fusion signal. Drops the startup window
    where the gripper is often already open.
    """
    g = np.asarray(gripper, dtype=np.float32)
    lo, hi = float(g.min()), float(g.max())
    if hi - lo < 1e-6:
        return []
    mid = (lo + hi) / 2.0
    closed = (g < mid).astype(int)
    ups = np.where(np.diff((g > mid).astype(int)) == 1)[0] + 1
    start_cut = int(T * startup_frac)
    out: list[int] = []
    for u in ups:
        if u < start_cut:
            continue
        s = max(0, u - hold_frames)
        if closed[s:u].mean() > 0.8:
            out.append(int(u))
    return out


def detect_corrected_arm(
    action: np.ndarray,
    intervention: np.ndarray,
    *,
    use_joints: bool,
    ratio_threshold: float,
    min_run: int,
) -> np.ndarray:
    """Per-frame corrected_arm in {none,left,right,both} for one episode."""
    out = np.array(["none"] * len(action), dtype=object)
    for s, e in _contiguous_runs(intervention == 1):
        label = _classify_run(
            action, s, e, use_joints=use_joints, ratio_threshold=ratio_threshold, min_run=min_run
        )
        out[s : e + 1] = label
    return out


# --------------------------------------------------------------------------- #
# per-arm value labels from annotation
# --------------------------------------------------------------------------- #
def _arm_spec(entry: dict, arm: str) -> tuple[int, list[int]]:
    """Resolve (n_assigned, insert_frames) for one arm from an annotation entry.

    Accepts either the explicit form ``{"left": {"n": 1, "insert_frames": [..]}}``
    or the shorthand ``{"left_insert": <frame or null>}`` (n defaults to 1).
    """
    if arm in entry and isinstance(entry[arm], dict):
        d = entry[arm]
        n = int(d.get("n", 1))
        frames = [int(f) for f in (d.get("insert_frames") or [])]
        return n, sorted(frames)
    short = entry.get(f"{arm}_insert", None)
    if short is None:
        return 1, []
    if isinstance(short, (list, tuple)):
        return len(short) or 1, sorted(int(f) for f in short)
    return 1, [int(short)]


def _per_arm_value(T: int, n_assigned: int, insert_frames: list[int]) -> np.ndarray:
    """value_label_arm,t = clip(completed(t)/N - 1, -1, 0)."""
    if n_assigned <= 0:
        return np.zeros((T,), dtype=np.float32)  # no task this episode -> "done"
    t = np.arange(T)
    completed = np.searchsorted(np.asarray(insert_frames, dtype=np.int64), t, side="right")
    val = completed.astype(np.float32) / float(n_assigned) - 1.0
    return np.clip(val, -1.0, 0.0).astype(np.float32)


def _load_annotation(path: Path) -> dict[int, dict]:
    with open(path) as f:
        raw = json.load(f)
    entries = raw["episodes"] if isinstance(raw, dict) and "episodes" in raw else raw
    return {int(e["episode_index"]): e for e in entries}


# --------------------------------------------------------------------------- #
# info.json
# --------------------------------------------------------------------------- #
def _update_info(info_path: Path, columns: dict[str, str]) -> None:
    if not info_path.exists():
        return
    info = json.loads(info_path.read_text())
    feats = info.setdefault("features", {})
    for name, dtype in columns.items():
        feats.setdefault(name, {"dtype": dtype, "shape": [1], "names": None})
    info_path.write_text(json.dumps(info, indent=4))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description="Add per-arm RECAP labels to a YAM LeRobot v2.1 dataset")
    p.add_argument("--data_dir", required=True, type=str)
    p.add_argument("--annotation_json", type=str, default=None, help="per-arm insertion frames -> value labels")
    p.add_argument("--emit_skeleton", type=str, default=None, help="write an annotation skeleton here and exit")
    p.add_argument("--action_col", type=str, default="action")
    p.add_argument("--human_col", type=str, default="intervention")
    p.add_argument("--min_run", type=int, default=3, help="ignore correction runs shorter than this (frames)")
    p.add_argument("--ratio_threshold", type=float, default=0.25, help="min/max energy below this => one-arm")
    p.add_argument("--energy", choices=["joints", "all"], default="joints", help="motion energy: joints-only or include gripper")
    p.add_argument("--dry_run", action="store_true", help="report only, write nothing")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    parquet_files = sorted(glob.glob(str(data_dir / "data" / "**" / "*.parquet"), recursive=True))
    if not parquet_files:
        raise SystemExit(f"no parquet files under {data_dir / 'data'}")
    use_joints = args.energy == "joints"

    # ---- skeleton emitter ------------------------------------------------- #
    if args.emit_skeleton:
        skel = []
        for fi, f in enumerate(parquet_files):
            df = pd.read_parquet(f)
            ep = _episode_id(df, fi)
            T = len(df)
            action = _action_matrix(df, args.action_col)
            # hint: the sparse-success frame (episode-level), NOT the per-arm insert
            succ = None
            if "reward" in df.columns:
                rw = np.asarray([_scalar_float(v) for v in df["reward"].tolist()])
                nz = np.where(rw > 0)[0]
                succ = int(nz[-1]) if len(nz) else None

            def _arm_block(gripper_idx: int) -> dict:
                cands = _release_candidates(action[:, gripper_idx], T)
                # best guess: the LAST sustained release (insert ends the cycle,
                # arm then retracts/idles). Review against _candidates.
                guess = [cands[-1]] if cands else []
                return {"n": 1, "insert_frames": guess, "_candidates": cands}

            skel.append(
                {
                    "episode_index": ep,
                    "_episode_len": int(T),
                    "_success_frame_hint": succ,
                    "left": _arm_block(6),
                    "right": _arm_block(13),
                }
            )
        Path(args.emit_skeleton).write_text(json.dumps({"episodes": skel}, indent=2))
        print(f"wrote skeleton for {len(skel)} episodes -> {args.emit_skeleton}")
        print("insert_frames is PRE-FILLED with a best guess (last sustained gripper release).")
        print("Review against _candidates / the head-camera video; set [] if that arm failed.")
        print("Loader ignores keys starting with '_', so leave them in as notes.")
        return

    annotation = _load_annotation(Path(args.annotation_json)) if args.annotation_json else None

    counts = {"left": 0, "right": 0, "both": 0, "none": 0}
    masked_left = masked_right = 0
    n_value_eps = 0
    total_frames = 0

    for fi, f in enumerate(parquet_files):
        df = pd.read_parquet(f)
        T = len(df)
        total_frames += T
        ep = _episode_id(df, fi)
        action = _action_matrix(df, args.action_col)
        intervention = _int_col(df, args.human_col)

        corrected = detect_corrected_arm(
            action, intervention,
            use_joints=use_joints, ratio_threshold=args.ratio_threshold, min_run=args.min_run,
        )
        for c in corrected:
            counts[str(c)] += 1

        loss_mask_left = np.ones((T,), dtype=np.float32)
        loss_mask_right = np.ones((T,), dtype=np.float32)
        loss_mask_left[corrected == "right"] = 0.0   # right corrected -> left frozen -> mask left
        loss_mask_right[corrected == "left"] = 0.0    # left corrected  -> right frozen -> mask right
        masked_left += int((loss_mask_left == 0).sum())
        masked_right += int((loss_mask_right == 0).sum())

        adv_left = np.array(["none"] * T, dtype=object)
        adv_right = np.array(["none"] * T, dtype=object)
        adv_left[(corrected == "left") | (corrected == "both")] = "positive"
        adv_right[(corrected == "right") | (corrected == "both")] = "positive"

        new_cols: dict[str, np.ndarray] = {
            "corrected_arm": corrected,
            "loss_mask_left": loss_mask_left,
            "loss_mask_right": loss_mask_right,
            "adv_ind_left": adv_left,
            "adv_ind_right": adv_right,
        }

        if annotation is not None:
            entry = annotation.get(ep)
            if entry is None:
                raise SystemExit(f"annotation missing episode_index={ep}")
            nl, fl = _arm_spec(entry, "left")
            nr, fr = _arm_spec(entry, "right")
            new_cols["value_label_left"] = _per_arm_value(T, nl, fl)
            new_cols["value_label_right"] = _per_arm_value(T, nr, fr)
            n_value_eps += 1

        if not args.dry_run:
            for name, arr in new_cols.items():
                df[name] = list(arr) if arr.dtype == object else arr.astype(np.float32)
            df.to_parquet(f, index=False)

    if not args.dry_run:
        feat_cols = {
            "corrected_arm": "string",
            "loss_mask_left": "float32",
            "loss_mask_right": "float32",
            "adv_ind_left": "string",
            "adv_ind_right": "string",
        }
        if annotation is not None:
            feat_cols["value_label_left"] = "float32"
            feat_cols["value_label_right"] = "float32"
        _update_info(data_dir / "meta" / "info.json", feat_cols)

    iv_total = counts["left"] + counts["right"] + counts["both"]
    print(f"episodes={len(parquet_files)}  frames={total_frames}")
    print(f"intervention frames: left={counts['left']} right={counts['right']} both={counts['both']} (total={iv_total})")
    print(f"masked frames: left-arm={masked_left} right-arm={masked_right}")
    if annotation is not None:
        print(f"wrote per-arm value labels for {n_value_eps} episodes")
    print("DRY RUN — nothing written." if args.dry_run else "written in place (parquet + info.json).")


if __name__ == "__main__":
    main()
