"""Sanity-check a value-model checkpoint's RANKING (not just loss).

Runs the value model on a random sample of frames and compares predicted V to the
ground-truth `value_label` (success episodes ramp -1->0; failures stay -1). What
matters for residual labeling is whether V ranks success-frames above
failure-frames, so we report corr(V, value_label) and the success-vs-failure
mean-V separation. Used to pick all_backbones vs siglip_only before label_residual.

  python scripts/verify_value_ranking.py --checkpoint_dir checkpoints/value_model/<name> [--n 2000]
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    import jax.numpy as jnp
    import label_advantage_from_vlm as la
    from openpi.models.value_model_config import ValueModelConfig

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--checkpoint_name", default=None)
    p.add_argument("--data_dir", default="/mnt/localssd/Sichang/lerobot_home/local/vials_recap_v1_1_v21_perarm")
    p.add_argument("--tokenizer_path", default="/mnt/localssd/Sichang/recap/vlm_ckpt/tokenizer.model")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--use_ema", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--base_image_col", default="observation.images.head_camera")
    p.add_argument("--wrist_image_col", default="observation.images.left_wrist_camera")
    p.add_argument("--right_wrist_image_col", default="observation.images.right_wrist_camera")
    args = p.parse_args()

    DS = Path(args.data_dir)
    vlab: dict[int, float] = {}
    for f in sorted(glob.glob(str(DS / "data" / "**" / "*.parquet"), recursive=True)):
        d = pd.read_parquet(f, columns=["index", "value_label"])
        gi = [int(np.asarray(v).reshape(-1)[0]) for v in d["index"].tolist()]
        vl = [float(np.asarray(v).reshape(-1)[0]) for v in d["value_label"].tolist()]
        vlab.update(zip(gi, vl))
    rng = np.random.default_rng(0)
    selected = sorted(rng.choice(sorted(vlab), size=min(args.n, len(vlab)), replace=False).tolist())

    ckpt = la._resolve_checkpoint_path(Path(args.checkpoint_dir), args.checkpoint_name)
    cfg = ValueModelConfig()
    params = la._load_checkpoint_params(ckpt, use_ema=args.use_ema)
    model = cfg.load(params, remove_extra_params=True)
    supports = jnp.linspace(-1, 0, 201, dtype=jnp.float32)
    tok = la._resolve_local_gemma_tokenizer_path(args.tokenizer_path)
    la._validate_gemma_tokenizer(tok)
    tasks = la._load_tasks_map(DS) if (DS / "meta" / "tasks.jsonl").exists() else None

    ds = la._build_inference_dataset(
        data_dir=DS, model_config=cfg, tokenizer_path=tok, instruction_col=None,
        base_image_col=args.base_image_col, wrist_image_col=args.wrist_image_col,
        right_wrist_image_col=args.right_wrist_image_col, copy_wrist_to_right=False, tasks_map=tasks,
    )
    ds = la.IndexedDataset(ds, selected)
    cache = la._compute_values_with_dataloader(
        dataset=ds, model=model, supports=supports,
        batch_size=args.batch_size, num_workers=args.num_workers, seed=0,
    )
    V = np.array(cache.flat_values, dtype=np.float32)
    GT = np.array([vlab[g] for g in selected], dtype=np.float32)[: len(V)]
    succ = GT > -0.05
    print(f"\n=== RANKING CHECK: {Path(args.checkpoint_dir).name} (n={len(V)}) ===")
    print(f"corr(V, value_label) = {np.corrcoef(V, GT)[0, 1]:.3f}")
    print(f"V SUCCESS-like (GT~0): mean={V[succ].mean():.3f} std={V[succ].std():.3f} n={int(succ.sum())}")
    print(f"V FAILURE      (GT=-1): mean={V[~succ].mean():.3f} std={V[~succ].std():.3f} n={int((~succ).sum())}")
    print(f"separation (succ-fail) = {V[succ].mean() - V[~succ].mean():.3f}  (want clearly >0)")


if __name__ == "__main__":
    main()
