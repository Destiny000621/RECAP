"""Auto-detect per-arm vial-insertion frames with SigLIP zero-shot (3 cameras).

Per arm, the insert is a gripper *release at the rack*: the gripper opens after
holding a vial, and that arm's WRIST camera looks down at the vial seated in the
rack holes. We fuse:

  - proprioception (cheap timing prior): gripper close->open transitions give a
    short list of release candidates per arm;
  - vision (disambiguation): SigLIP scores each candidate's wrist frame for
    "a robot gripper inserting a test tube vial into a rack". The candidate whose
    wrist frame most matches wins; if none clears the margin, that arm failed.

SigLIP separates cleanly on this scene (~+13 logit between a true insert and an
empty-table/wall frame), so restricting it to release candidates makes the pick
a reliable few-way choice rather than an open-ended search.

Output is an annotation JSON consumable by ``scripts/add_per_arm_labels.py``:

  {"episodes": [{"episode_index": 0,
                 "left":  {"n": 1, "insert_frames": [557], "_score": 1.9, "_candidates": {...}},
                 "right": {"n": 1, "insert_frames": [1180], ...}}, ...]}

Review the ``_score`` / ``_candidates`` notes; ``add_per_arm_labels`` ignores
keys starting with '_'. Re-run a few episodes with a different --siglip_model
(e.g. google/siglip-so400m-patch14-224) for higher accuracy.

Usage:
  python scripts/detect_insert_frames.py --data_dir <ds> --out anns.json
  python scripts/detect_insert_frames.py --data_dir <ds> --out anns.json --episodes 0-4 --dump_dir /tmp/chk
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from add_per_arm_labels import _action_matrix, _episode_id, _release_candidates  # noqa: E402

GRIPPER_IDX = {"left": 6, "right": 13}
WRIST_DIR = {"left": "observation.images.left_wrist_camera", "right": "observation.images.right_wrist_camera"}

# prompt[0] is the target; the rest are foils. A frame "succeeds" when the target
# logit beats the best foil by --margin. This is robust to model/scale changes.
PROMPTS = [
    "a robot gripper inserting a test tube vial into a rack with holes",
    "a robot gripper over an empty wooden table",
    "a robot gripper pointing at a blank wall",
    "a robot gripper holding a vial in the air away from the rack",
]


def _parse_episodes(spec: str | None) -> set[int] | None:
    if not spec:
        return None
    out: set[int] = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def _decode_frames(video_path: Path, wanted: set[int]) -> dict[int, Image.Image]:
    """Sequentially decode and return PIL frames at the wanted indices."""
    frames: dict[int, Image.Image] = {}
    if not wanted:
        return frames
    cap = cv2.VideoCapture(str(video_path))
    last = max(wanted)
    i = 0
    while i <= last:
        ok, fr = cap.read()
        if not ok:
            break
        if i in wanted:
            frames[i] = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    return frames


class SiglipScorer:
    def __init__(self, model_name: str, device: str):
        from transformers import AutoModel, AutoProcessor

        self.device = device
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.proc = AutoProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def target_contrast(self, images: list[Image.Image]) -> np.ndarray:
        """For each image return (target_logit, target_logit - best_foil_logit)."""
        inp = self.proc(text=PROMPTS, images=images, padding="max_length", truncation=True, return_tensors="pt").to(
            self.device
        )
        logits = self.model(**inp).logits_per_image  # (n_images, n_prompts)
        tgt = logits[:, 0]
        foil = logits[:, 1:].max(dim=1).values
        return np.stack([tgt.cpu().numpy(), (tgt - foil).cpu().numpy()], axis=1)


def _candidates_for_arm(action: np.ndarray, T: int, arm: str, hold_frames: int, dense_step: int) -> list[int]:
    """Release candidates + a sparse dense grid fallback (clamped to [0,T-1])."""
    cand = set(_release_candidates(action[:, GRIPPER_IDX[arm]], T, hold_frames=hold_frames))
    if dense_step > 0:
        cand.update(range(int(T * 0.05), T, dense_step))
    return sorted(min(max(c, 0), T - 1) for c in cand)


def main() -> None:
    p = argparse.ArgumentParser(description="Auto-detect per-arm insert frames via SigLIP over wrist cameras")
    p.add_argument("--data_dir", required=True, type=str)
    p.add_argument("--out", required=True, type=str, help="output annotation JSON")
    p.add_argument("--siglip_model", type=str, default="google/siglip-base-patch16-224",
                   help="HF SigLIP id; siglip-so400m-patch14-224 for higher accuracy")
    p.add_argument("--episodes", type=str, default=None, help="subset, e.g. '0-4,9'")
    p.add_argument("--action_col", type=str, default="action")
    p.add_argument("--hold_frames", type=int, default=20, help="closed-hold window for release candidates")
    p.add_argument("--dense_step", type=int, default=20, help="also score every Nth frame as fallback (0=off)")
    p.add_argument("--offset", type=int, default=8, help="score the wrist frame this many frames after release")
    p.add_argument("--margin", type=float, default=0.5, help="target must beat best foil by this to count as insert")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dump_dir", type=str, default=None, help="save chosen wrist frames here for review")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    parquet_files = sorted(glob.glob(str(data_dir / "data" / "**" / "*.parquet"), recursive=True))
    if not parquet_files:
        raise SystemExit(f"no parquet under {data_dir/'data'}")
    keep = _parse_episodes(args.episodes)
    scorer = SiglipScorer(args.siglip_model, args.device)
    dump = Path(args.dump_dir) if args.dump_dir else None
    if dump:
        dump.mkdir(parents=True, exist_ok=True)

    episodes_out = []
    for fi, f in enumerate(parquet_files):
        df = pd.read_parquet(f, columns=["episode_index", args.action_col])
        ep = _episode_id(df, fi)
        if keep is not None and ep not in keep:
            continue
        action = _action_matrix(df, args.action_col)
        T = len(df)
        entry: dict = {"episode_index": ep}

        for arm in ("left", "right"):
            cands = _candidates_for_arm(action, T, arm, args.hold_frames, args.dense_step)
            # score the frame a few steps AFTER each candidate (seated vial is clearest then)
            score_at = {min(c + args.offset, T - 1) for c in cands}
            video = data_dir / "videos" / Path(f).parent.name / WRIST_DIR[arm] / f"episode_{ep:06d}.mp4"
            frames = _decode_frames(video, score_at)
            idxs = [i for i in sorted(score_at) if i in frames]
            if not idxs:
                entry[arm] = {"n": 1, "insert_frames": [], "_score": None, "_note": "no frames decoded"}
                continue
            scored = scorer.target_contrast([frames[i] for i in idxs])  # (k,2): tgt, contrast
            best = int(np.argmax(scored[:, 0]))
            best_frame = idxs[best]
            best_tgt, best_contrast = float(scored[best, 0]), float(scored[best, 1])
            success = best_contrast >= args.margin
            # snap to the originating release candidate (precise timing), else keep frame
            snap = min(cands, key=lambda c: abs((c + args.offset) - best_frame)) if cands else best_frame
            entry[arm] = {
                "n": 1,
                "insert_frames": [int(snap)] if success else [],
                "_score": round(best_tgt, 2),
                "_contrast": round(best_contrast, 2),
                "_best_frame": best_frame,
                "_candidates": {int(i): round(float(s), 2) for i, s in zip(idxs, scored[:, 0])},
            }
            if dump and success:
                frames[best_frame].save(dump / f"ep{ep:04d}_{arm}_{best_frame}_c{best_contrast:.1f}.png")

        episodes_out.append(entry)
        ls, rs = entry["left"], entry["right"]
        print(f"ep{ep:04d}  left={ls['insert_frames']}(c={ls.get('_contrast')})  "
              f"right={rs['insert_frames']}(c={rs.get('_contrast')})")

    Path(args.out).write_text(json.dumps({"episodes": episodes_out}, indent=2))
    n_l = sum(1 for e in episodes_out if e["left"]["insert_frames"])
    n_r = sum(1 for e in episodes_out if e["right"]["insert_frames"])
    print(f"\nwrote {args.out}: {len(episodes_out)} episodes  (left inserts={n_l}, right inserts={n_r})")
    print("review _score/_contrast/_candidates, then: add_per_arm_labels.py --annotation_json " + args.out)


if __name__ == "__main__":
    main()
