"""Open-loop action-error eval: rank checkpoints by how well they predict held-out action chunks.

For each sampled frame it runs the policy's `sample_actions` and compares the predicted action
chunk to the ground-truth chunk, in the model's (normalized, delta) action space. Lower = better.

It reuses the EXACT training data pipeline (`transform_dataset`), so model inputs are byte-identical
to training — no risk of image/format mismatch. `sample_actions` is jitted exactly like the serving
Policy does.

NOTE on "held-out": all vials_4 episodes are in training for both the base and aug configs, so there
is no held-out 4-vial data. Two complementary ways to use this:
  - Generalization  : run on 8ml episodes OUTSIDE the aug-46 set (truly unseen by both configs) ->
                      measures insertion-skill generalization (single-vial distribution).
  - In-distribution : run on vials_4 episodes -> measures action-fit on the actual 4-vial task
                      (train data for both configs, but the right distribution; good for precision).

Error is reported in normalized action space. Within a config (same norm-stats) it's exactly
comparable across checkpoints. Across configs (base vs aug) the norm-stats differ slightly (226 vs
180 eps over mostly-overlapping data), so treat cross-config diffs as approximate.

Usage:
    uv run python scripts/eval_open_loop.py \
        --config-name pi05_yam_vial_4_30fps \
        --repo-id local/8ml_vial_place_30fps_v21 --episodes 60-79 \
        --checkpoint /mnt/localssd/Sichang/openpi-checkpoints/pi05_yam_vial_4_30fps/v2/5000 \
        --checkpoint /mnt/localssd/Sichang/openpi-checkpoints/pi05_yam_vial_4_30fps/v2/10000 \
        --checkpoint /mnt/localssd/Sichang/openpi-checkpoints/pi05_yam_vial_4_30fps/v2/11999
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
from openpi.shared import nnx_utils
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

GRIPPER_DIMS = (6, 13)  # YAM 14-dim action: [L 6 joints + 1 grip, R 6 joints + 1 grip]
REAL_DIM = 14


def parse_episodes(spec: str, total: int) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))  # inclusive
        else:
            out.append(int(part))
    seen, res = set(), []
    for e in out:
        if e in seen:
            continue
        if not (0 <= e < total):
            raise ValueError(f"episode {e} out of range [0, {total})")
        seen.add(e)
        res.append(e)
    return res


def build_eval_frames(data_config, action_horizon, repo_id, episodes, stride, max_per_ep):
    """Build the transformed dataset and the list of frame indices to evaluate.

    Only frames with a FULL ground-truth action chunk inside their episode are kept (we drop the last
    `action_horizon` frames of each episode so the GT chunk isn't padded across the episode boundary).
    """
    meta = LeRobotDatasetMetadata(repo_id)
    delta = {k: [t / meta.fps for t in range(action_horizon)] for k in data_config.action_sequence_keys}
    # Load the FULL dataset (no episodes= filter): with a non-zero-based subset, lerobot 0.1.0's
    # episode_data_index is re-indexed to the subset while frames keep original episode_index, which
    # breaks __getitem__. Selecting frames by global index against the full index avoids that.
    ds = LeRobotDataset(repo_id, delta_timestamps=delta)
    dc = dataclasses.replace(data_config, repo_id=repo_id)
    ds_t = _data_loader.transform_dataset(ds, dc)

    edi = ds.episode_data_index  # indexed by original episode_index, 0..total-1
    frame_idxs: list[int] = []
    for ep in episodes:
        start, end = int(edi["from"][ep]), int(edi["to"][ep])
        last = end - action_horizon  # ensure a full, un-padded GT chunk
        if last <= start:
            continue
        frame_idxs.extend(list(range(start, last, stride))[:max_per_ep])
    return ds_t, frame_idxs


def collate(items: list[dict]) -> dict:
    return jax.tree.map(lambda *xs: np.stack(xs), *items)


def evaluate_checkpoint(config, ckpt: str, ds_t, frame_idxs, batch_size) -> np.ndarray:
    model = config.model.load(_model.restore_params(Path(ckpt) / "params", dtype=jnp.bfloat16))
    sample_actions = nnx_utils.module_jit(model.sample_actions)
    rng = jax.random.key(0)

    sq_sum = np.zeros(REAL_DIM, dtype=np.float64)
    n = 0
    for s in range(0, len(frame_idxs), batch_size):
        batch = collate([ds_t[i] for i in frame_idxs[s : s + batch_size]])
        actions_gt = np.asarray(batch.pop("actions"))[..., :REAL_DIM]  # (B, H, 14)
        obs = _model.Observation.from_dict(batch)
        rng, key = jax.random.split(rng)
        pred = np.asarray(sample_actions(key, obs))[..., :REAL_DIM]  # (B, H, 14)
        diff = pred - actions_gt
        sq_sum += (diff.astype(np.float64) ** 2).sum(axis=(0, 1))
        n += diff.shape[0] * diff.shape[1]
    return sq_sum / n  # per-dim MSE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", required=True)
    ap.add_argument("--repo-id", required=True, help="dataset to evaluate on")
    ap.add_argument("--episodes", required=True, help="'0-19' or '60-79,178-187' (inclusive ranges)")
    ap.add_argument("--checkpoint", action="append", required=True, help="checkpoint step dir (repeatable)")
    ap.add_argument("--stride", type=int, default=10, help="sample every Nth frame within an episode")
    ap.add_argument("--max-per-ep", type=int, default=8, help="cap frames per episode")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    config = _config.get_config(args.config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.norm_stats is None:
        raise SystemExit(f"No norm_stats for '{args.config_name}'. Run compute_norm_stats first.")
    horizon = config.model.action_horizon

    meta = LeRobotDatasetMetadata(args.repo_id)
    eps = parse_episodes(args.episodes, meta.total_episodes)
    ds_t, frame_idxs = build_eval_frames(data_config, horizon, args.repo_id, eps, args.stride, args.max_per_ep)
    logger.info(
        f"Config={args.config_name}  eval_repo={args.repo_id}  episodes={len(eps)}  "
        f"frames={len(frame_idxs)}  horizon={horizon}"
    )
    if not frame_idxs:
        raise SystemExit("No evaluable frames (episodes too short for the action horizon?).")

    rows = []
    for ckpt in args.checkpoint:
        mse = evaluate_checkpoint(config, ckpt, ds_t, frame_idxs, args.batch_size)
        overall = float(mse.mean())
        joints = float(np.delete(mse, list(GRIPPER_DIMS)).mean())
        gripper = float(mse[list(GRIPPER_DIMS)].mean())
        rows.append((Path(ckpt).parent.parent.name + "/" + Path(ckpt).name, overall, joints, gripper))
        logger.info(f"{ckpt}: overall={overall:.5f} joints={joints:.5f} gripper={gripper:.5f}")

    rows.sort(key=lambda r: r[1])
    print("\n=== ranking: normalized open-loop action MSE (lower = better) ===")
    print(f"  eval set: {args.repo_id}  episodes {args.episodes}  ({len(frame_idxs)} frames)")
    print(f"  {'checkpoint':28s} {'overall':>9s} {'joints':>9s} {'gripper':>9s}")
    for name, o, j, g in rows:
        print(f"  {name:28s} {o:9.5f} {j:9.5f} {g:9.5f}")
    print(f"\n  best: {rows[0][0]} (overall MSE {rows[0][1]:.5f})")


if __name__ == "__main__":
    main()
