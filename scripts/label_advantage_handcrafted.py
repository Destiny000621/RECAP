"""Rule-based per-arm advantage labeling for the YAM vials dagger dataset.

Replaces the VLM value-model labeling (Stage 5) for the cases where we trust
hand rules more than a value model. The intervention itself is the supervision:
a human correction means the policy was about to fail, so the corrected arm's
behavior *just before* the correction is negative. The rules (per arm a):

  success autonomous rollout (no intervention)      -> a positive everywhere
  correction attributed to arm a (a is corrected)   -> a positive on the
      correction frames (human fix); a NEGATIVE on the window of `--neg_window`
      frames immediately BEFORE the correction starts (the bad behavior)
  correction attributed to the OTHER arm (a frozen) -> a left at its rollout
      default; its action is masked out of the loss (loss_mask from
      add_per_arm_labels), so the held pose is never regressed
  'both' correction                                 -> both arms negative in the
      pre-correction window, both positive on the correction frames
  failure autonomous rollout, arm a failed          -> a left as 'none' (handed
      to a VLM labeler later; we don't guess)

Per-arm success comes from lerobot_annotator SUMMARY mode (5-vote majority):
each episode's `vials: [{insert_arm, ...}]` tells which arm seated each vial.
Pass --summary_dir <annotations_summary/<repo>/<model>> or a precomputed
--success_json ({"episodes":[{"episode_index":i,"left":bool,"right":bool}]}).

Prereq: run scripts/add_per_arm_labels.py first (writes `corrected_arm`,
`loss_mask_*`, and the `adv_ind_*` scaffold this script overwrites).

Writes `adv_ind_left` / `adv_ind_right` in place. Run on a copy.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from add_per_arm_labels import _contiguous_runs, _episode_id, _int_col, _update_info  # noqa: E402


# --------------------------------------------------------------------------- #
# per-arm success from summary-mode annotations
# --------------------------------------------------------------------------- #
def _success_from_summary(summary_dir: Path) -> dict[int, dict[str, bool]]:
    out: dict[int, dict[str, bool]] = {}
    files = sorted(summary_dir.glob("episode_*.json"))
    if not files:
        raise SystemExit(f"no episode_*.json under {summary_dir}")
    for f in files:
        j = json.loads(f.read_text())
        ep = int(j["episode_index"])
        c: Counter = Counter()
        for v in j.get("vials") or []:
            arm = v.get("insert_arm") or v.get("grasp_arm")
            if arm in ("left", "right"):
                c[arm] += 1
            elif arm == "both":
                c["left"] += 1
                c["right"] += 1
        out[ep] = {
            "left": c["left"] > 0,
            "right": c["right"] > 0,
            "_counts": dict(c),
            "_length": int(j["length"]) if "length" in j else None,
        }
    return out


def _success_from_json(path: Path) -> dict[int, dict[str, bool]]:
    raw = json.loads(path.read_text())
    entries = raw["episodes"] if isinstance(raw, dict) and "episodes" in raw else raw
    return {int(e["episode_index"]): {"left": bool(e["left"]), "right": bool(e["right"])} for e in entries}


def _success_from_reward(parquet_files: list[str], reward_col: str = "reward") -> dict[int, dict[str, bool]]:
    """Per-episode success from the recorded sparse `reward` (reward>0 == full success).

    This is the RELIABLE success source. Summary mode's `total_vials`/`insert_arm`
    measures vials-present + which arm handled them (attribution), NOT seated
    success — it over-reports on failures (verified: 33/93 reward-failures still
    show total_vials>0). reward=1 means all vials seated => both arms succeeded;
    reward=0 => failure (both -> 'none' -> VLM residual, which sorts per-arm).
    """
    out: dict[int, dict[str, bool]] = {}
    for fi, f in enumerate(parquet_files):
        df = pd.read_parquet(f, columns=["episode_index", reward_col])
        ep = _episode_id(df, fi)
        rw = np.array([float(np.asarray(v).reshape(-1)[0]) for v in df[reward_col].tolist()], dtype=np.float32)
        succ = bool(rw.sum() > 0)
        out[ep] = {"left": succ, "right": succ}
    return out


# --------------------------------------------------------------------------- #
# labeling
# --------------------------------------------------------------------------- #
def _label_episode(
    corrected: np.ndarray,
    intervention: np.ndarray,
    success_left: bool,
    success_right: bool,
    neg_window: int,
) -> tuple[np.ndarray, np.ndarray]:
    T = len(corrected)
    # rollout default: positive if that arm succeeded, else 'none' (-> VLM)
    adv_l = np.array(["positive" if success_left else "none"] * T, dtype=object)
    adv_r = np.array(["positive" if success_right else "none"] * T, dtype=object)

    runs = _contiguous_runs(intervention == 1)
    for ri, (s, e) in enumerate(runs):
        arm = str(corrected[s])  # constant over the run (set per-run by detector)
        hit_left = arm in ("left", "both")
        hit_right = arm in ("right", "both")
        # correction frames: the corrected arm(s) demonstrate the human fix
        if hit_left:
            adv_l[s : e + 1] = "positive"
        if hit_right:
            adv_r[s : e + 1] = "positive"
        # frozen arm on a single-arm correction: its action is loss-masked, and we
        # explicitly DO NOT label its held pose 'positive' — set 'none' so the
        # frozen behavior never enters the positive conditioning bucket.
        if arm == "right":  # right corrected -> left frozen
            adv_l[s : e + 1] = "none"
        elif arm == "left":  # left corrected -> right frozen
            adv_r[s : e + 1] = "none"
        # pre-correction negative window for the corrected arm(s); clamp so it
        # never reaches back into the previous correction run.
        prev_end = runs[ri - 1][1] if ri > 0 else -1
        w0 = max(0, s - neg_window, prev_end + 1)
        if hit_left:
            adv_l[w0:s] = "negative"
        if hit_right:
            adv_r[w0:s] = "negative"
    return adv_l, adv_r


def main() -> None:
    p = argparse.ArgumentParser(description="Rule-based per-arm advantage labeling")
    p.add_argument("--data_dir", required=True, type=str)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--reward_success", action="store_true",
                   help="RECOMMENDED: per-episode success from the recorded `reward` column "
                        "(reward>0 => both arms succeeded). Reliable; summary mode over-reports.")
    g.add_argument("--summary_dir", type=str,
                   help="lerobot_annotator annotations_summary/<repo>/<model> dir. NOTE: its "
                        "total_vials/insert_arm is attribution/vials-present, NOT seated success "
                        "(over-reports on failures) — prefer --reward_success.")
    g.add_argument("--success_json", type=str, help="precomputed per-arm success json")
    p.add_argument("--reward_col", type=str, default="reward")
    p.add_argument("--neg_window", type=int, default=45, help="frames before a correction labeled negative (30fps: 45=1.5s)")
    p.add_argument("--human_col", type=str, default="intervention")
    p.add_argument("--strict_length", action="store_true",
                   help="error (not warn) if a summary episode's `length` != the v2.1 episode frame count "
                        "— catches v3.0<->v2.1 episode_index misalignment")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    parquet_files = sorted(glob.glob(str(data_dir / "data" / "**" / "*.parquet"), recursive=True))
    if not parquet_files:
        raise SystemExit(f"no parquet under {data_dir/'data'}")

    if args.reward_success:
        success = _success_from_reward(parquet_files, args.reward_col)
    elif args.summary_dir:
        success = _success_from_summary(Path(args.summary_dir))
    else:
        success = _success_from_json(Path(args.success_json))

    tally = {"left": Counter(), "right": Counter()}
    n_missing = 0
    length_mismatches: list[tuple[int, int, int]] = []  # (ep, summary_len, v21_len)
    for fi, f in enumerate(parquet_files):
        df = pd.read_parquet(f)
        if "corrected_arm" not in df.columns:
            raise SystemExit("missing 'corrected_arm' — run scripts/add_per_arm_labels.py first")
        ep = _episode_id(df, fi)
        T = len(df)
        corrected = np.array([str(v) for v in df["corrected_arm"].tolist()], dtype=object)
        intervention = _int_col(df, args.human_col)
        s = success.get(ep)
        if s is None:
            n_missing += 1
            s = {"left": False, "right": False}  # unknown -> treat as failure (-> VLM)
        elif s.get("_length") is not None and s["_length"] != T:
            # summary came from the v3.0 dataset; v2.1 is converted from it, so
            # frame counts must match. A mismatch means episode_index is misaligned
            # between the two versions -> wrong success attached to wrong episode.
            length_mismatches.append((ep, s["_length"], T))
        adv_l, adv_r = _label_episode(corrected, intervention, s["left"], s["right"], args.neg_window)
        tally["left"].update(adv_l.tolist())
        tally["right"].update(adv_r.tolist())
        if not args.dry_run:
            df["adv_ind_left"] = list(adv_l)
            df["adv_ind_right"] = list(adv_r)
            df.to_parquet(f, index=False)

    if not args.dry_run:
        _update_info(data_dir / "meta" / "info.json", {"adv_ind_left": "string", "adv_ind_right": "string"})

    if length_mismatches:
        msg = (
            f"{len(length_mismatches)} episode(s) have summary `length` != v2.1 frame count "
            f"(v3.0<->v2.1 episode_index misalignment). First few (ep, summary_len, v2.1_len): "
            f"{length_mismatches[:5]}"
        )
        if args.strict_length:
            raise SystemExit("ERROR: " + msg)
        print("WARNING: " + msg)

    print(f"episodes={len(parquet_files)}  success_map_missing={n_missing}  length_mismatches={len(length_mismatches)}")
    for arm in ("left", "right"):
        t = tally[arm]
        print(f"  {arm:5s}: positive={t['positive']} negative={t['negative']} none(->VLM)={t['none']}")
    print("DRY RUN — nothing written." if args.dry_run else "written in place (adv_ind_left/right + info.json).")


if __name__ == "__main__":
    main()
