"""Build a reliable per-arm success map: reward for successes + manual for failures.

Episode-level `reward` is reliable but binary; it can't express partial success
(left seats, right fails — common because the right arm is weak). Summary mode is
unreliable for success (it over-reports seated vials on failures). So:

  reward > 0  (full success)  -> {left: true, right: true}   (auto, both seated)
  reward == 0 (failure)       -> {left: false, right: false} + flagged _needs_review,
                                 and the final head-cam frame is extracted so you can
                                 glance and set which arm(s) actually seated a vial.

Unedited failures stay both-false => both arms -> 'none' -> VLM residual (safe: no
false positives). Edit the flagged entries to mark the arm(s) that succeeded; those
get rule-1 `positive` directly instead of going to the VLM.

Output is a --success_json for scripts/label_advantage_handcrafted.py:
  {"episodes": [{"episode_index": i, "left": bool, "right": bool}, ...]}

Usage:
  python scripts/prep_success_map.py --data_dir <v2.1_dataset> \
    --out success_map.json --frames_dir success_review
  # ... glance at success_review/ep00NN_final.png, edit success_map.json failures ...
  python scripts/label_advantage_handcrafted.py --data_dir <v2.1_dataset> \
    --success_json success_map.json
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def _episode_id(df: pd.DataFrame, fallback: int) -> int:
    if "episode_index" in df.columns and len(df) > 0:
        return int(np.asarray(df["episode_index"].iloc[0]).reshape(-1)[0])
    return fallback


def _extract_final_frame(video: Path, frame_idx: int, out: Path) -> bool:
    if not video.exists():
        return False
    try:
        subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(video),
             "-vf", f"select=eq(n\\,{frame_idx})", "-vframes", "1", "-y", str(out)],
            check=True, stdin=subprocess.DEVNULL,
        )
        return out.exists()
    except Exception:
        return False


def main() -> None:
    p = argparse.ArgumentParser(description="Per-arm success map: reward auto + manual failures")
    p.add_argument("--data_dir", required=True, type=str, help="v2.1 dataset (data/ + videos/)")
    p.add_argument("--out", required=True, type=str, help="output success_map.json")
    p.add_argument("--frames_dir", type=str, default=None, help="extract failure final frames here for review")
    p.add_argument("--reward_col", type=str, default="reward")
    p.add_argument("--head_cam", type=str, default="observation.images.head_camera")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    parquet_files = sorted(glob.glob(str(data_dir / "data" / "**" / "*.parquet"), recursive=True))
    if not parquet_files:
        raise SystemExit(f"no parquet under {data_dir/'data'}")
    frames_dir = Path(args.frames_dir) if args.frames_dir else None
    if frames_dir:
        frames_dir.mkdir(parents=True, exist_ok=True)

    episodes = []
    n_succ = n_fail = n_extracted = 0
    for fi, f in enumerate(parquet_files):
        df = pd.read_parquet(f, columns=["episode_index", args.reward_col])
        ep = _episode_id(df, fi)
        T = len(df)
        rw = np.array([float(np.asarray(v).reshape(-1)[0]) for v in df[args.reward_col].tolist()], dtype=np.float32)
        if rw.sum() > 0:
            episodes.append({"episode_index": ep, "left": True, "right": True})
            n_succ += 1
        else:
            entry = {"episode_index": ep, "left": False, "right": False, "_needs_review": True}
            n_fail += 1
            if frames_dir:
                video = data_dir / "videos" / Path(f).parent.name / args.head_cam / f"episode_{ep:06d}.mp4"
                out_png = frames_dir / f"ep{ep:04d}_final.png"
                if _extract_final_frame(video, T - 1, out_png):
                    entry["_final_frame"] = str(out_png)
                    n_extracted += 1
            episodes.append(entry)

    Path(args.out).write_text(json.dumps({"episodes": episodes}, indent=1))
    print(f"episodes={len(episodes)}  success(auto,both-true)={n_succ}  failure(review)={n_fail}  frames_extracted={n_extracted}")
    print(f"wrote {args.out}. Edit the {n_fail} _needs_review entries (set left/right seated by glancing at the frames),")
    print("then: label_advantage_handcrafted.py --success_json " + args.out)


if __name__ == "__main__":
    main()
