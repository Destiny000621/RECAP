"""Precompute the deterministic pi0.5 base-action chunk per frame (residual RECAP).

Flavor-A residual RECAP trains the actor to predict `action - base_action`, where
`base_action = b(o)` is the FROZEN pi0.5 SFT policy run with a FIXED noise so it is
a deterministic function of the observation. This script runs that base over every
frame of the v2.1 dataset and writes a `base_action` column (the full
(action_horizon, 14) open-loop chunk per anchor frame), which:
  - the training transform `SubtractBaseAction` subtracts to form the residual
    target (config `pi06_yam_vial_30fps_from_sft_recap_residual`), and
  - inference reconstructs live via the SAME b(o) (same fixed noise) and adds the
    sampled residual back: a = b(o) + Δ(o).

Determinism is the whole point: pass `sample_kwargs={"noise": zeros}` here AND at
serving so the residual is always added to the exact base it was trained against.
`noise=zeros` integrates the flow ODE from the origin (a stable, reproducible base);
a fixed seed would also work as long as it is identical in both places.

GPU-gated; run once. Inputs the user must supply:
  --base_config   a registered pi0.5 SFT TrainConfig name (the base, pistar=False,
                  no advantage conditioning — its tokenizer must NOT add adv tokens)
  --checkpoint_dir the pi0.5 SFT checkpoint dir (contains params/ + assets/)
  --data_dir      the v2.1 dataset to annotate (writes base_action in place)

  python scripts/precompute_base_action.py \
    --base_config pi05_yam_vial_30fps --checkpoint_dir <pi05_ckpt> \
    --data_dir <v2.1_dataset>_perarm --num_steps 10

NOTE: open-loop chunk per anchor (what inference adds the residual to), NOT
re-chunked first-steps. Assumes adapt_to_pi=False (YAM) so the action space is
linear and raw subtraction == residual. Re-run compute_norm_stats on the residual
config afterwards. This is the one piece most likely to need a tweak on the first
real run (exact base config name / frame format); it is intentionally simple
(per-frame infer) over fast.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd


def _int_col(df: pd.DataFrame, c: str) -> np.ndarray:
    return np.array([int(np.asarray(v).reshape(-1)[0]) for v in df[c].tolist()], dtype=np.int64)


def main() -> None:
    import jax.numpy as jnp
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    p = argparse.ArgumentParser(description="Precompute deterministic pi0.5 base-action chunks")
    p.add_argument("--base_config", required=True, type=str, help="registered pi0.5 SFT TrainConfig name (the base)")
    p.add_argument("--checkpoint_dir", required=True, type=str, help="pi0.5 SFT checkpoint dir")
    p.add_argument("--data_dir", required=True, type=str, help="v2.1 dataset to annotate (writes base_action in place)")
    p.add_argument("--num_steps", type=int, default=10, help="flow ODE steps for the base")
    p.add_argument("--repo_id", type=str, default=None, help="override lerobot repo_id (default local/<basename>)")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    parquet_files = sorted(glob.glob(str(data_dir / "data" / "**" / "*.parquet"), recursive=True))
    if not parquet_files:
        raise SystemExit(f"no parquet under {data_dir/'data'}")

    base_config = _config.get_config(args.base_config)
    action_horizon = base_config.model.action_horizon
    action_dim = base_config.model.action_dim
    # Deterministic base: fixed (zero) noise reused verbatim at serving. Batch=1
    # because Policy.infer processes a single frame.
    fixed_noise = jnp.zeros((1, action_horizon, action_dim), dtype=jnp.float32)

    policy = _policy_config.create_trained_policy(
        base_config,
        args.checkpoint_dir,
        repack_transforms=base_config.data.repack_transforms,
        sample_kwargs={"num_steps": args.num_steps, "noise": fixed_noise},
    )

    # Raw frames (decoded images + state) from the lerobot dataset, fed to
    # policy.infer exactly as at serving.
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    repo_id = args.repo_id or f"local/{data_dir.name}"
    ds = LeRobotDataset(repo_id)
    if len(ds) == 0:
        raise SystemExit("empty dataset")

    # global index -> (action_horizon, 14) base chunk
    base_by_index: dict[int, np.ndarray] = {}
    for i in range(len(ds)):
        frame = ds[i]
        gidx = int(np.asarray(frame["index"]).reshape(-1)[0]) if "index" in frame else i
        out = policy.infer(frame)
        chunk = np.asarray(out["actions"], dtype=np.float32)[:, :14]  # (H, 14)
        base_by_index[gidx] = chunk
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(ds)} base chunks")

    if args.dry_run:
        print(f"DRY RUN — computed {len(base_by_index)} base chunks (shape {action_horizon}x14), nothing written.")
        return

    written = 0
    for f in parquet_files:
        df = pd.read_parquet(f)
        gidx = _int_col(df, "index")
        col = [base_by_index[int(g)] for g in gidx]  # one (H,14) array per row
        df["base_action"] = col
        df.to_parquet(f, index=False)
        written += len(df)

    # info.json feature entry
    import json
    info_path = data_dir / "meta" / "info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        info.setdefault("features", {}).setdefault(
            "base_action",
            {"dtype": "float32", "shape": [action_horizon, 14], "names": None},
        )
        info_path.write_text(json.dumps(info, indent=4))

    print(f"wrote base_action ({action_horizon}x14 per frame) to {written} rows. "
          f"Next: compute_norm_stats --config-name pi06_yam_vial_30fps_from_sft_recap_residual")


if __name__ == "__main__":
    main()
