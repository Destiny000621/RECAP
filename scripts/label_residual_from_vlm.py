"""Fill rule-3 (`none`) advantage labels with the per-arm residual value model.

The rule-based labeler leaves failure-without-intervention frames `none`. After
training a per-arm value model on the rule-1/2 hand labels
(`train_value.py --arm {left,right}` on the `cls_value_*` targets from
`prep_residual_training.py`), this script runs it on exactly those residual
frames and writes `positive`/`negative` back into `adv_ind_{arm}`.

Residual frames for an arm = `adv_ind_{arm} == "none"` AND `loss_mask_{arm} == 1`
(the loss_mask==1 condition drops the frozen-arm `none` frames, which the policy
loss masks anyway). The value model predicts V in [-1,0]; V > --threshold ->
`positive`, else `negative`. Only those frames are touched; rule-1/2 hand labels
are left intact.

Run once per arm, reusing the value-inference machinery of
label_advantage_from_vlm.py:

  python scripts/label_residual_from_vlm.py --data_dir <v2.1> --arm left \
    --checkpoint_dir checkpoints/value_model/residual_left/step_00006000 \
    --tokenizer_path <gemma tokenizer.model> \
    --base_image_col observation.images.head_camera \
    --wrist_image_col observation.images.left_wrist_camera \
    --right_wrist_image_col observation.images.right_wrist_camera --use_ema
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import label_advantage_from_vlm as la  # heavy helpers: ckpt load, dataset build, value inference


def _str_col(df, c):
    return np.array([str(v) for v in df[c].tolist()], dtype=object)


def _f32_col(df, c):
    return np.array([float(np.asarray(v).reshape(-1)[0]) for v in df[c].tolist()], dtype=np.float32)


def _int_col(df, c):
    return np.array([int(np.asarray(v).reshape(-1)[0]) for v in df[c].tolist()], dtype=np.int64)


def _residual_global_indices(parquet_files: list[str], arm: str) -> set[int]:
    out: set[int] = set()
    for f in parquet_files:
        df = pd.read_parquet(f, columns=[f"adv_ind_{arm}", f"loss_mask_{arm}", "index"])
        adv = _str_col(df, f"adv_ind_{arm}")
        lm = _f32_col(df, f"loss_mask_{arm}")
        gidx = _int_col(df, "index")
        out.update(gidx[(adv == "none") & (lm == 1.0)].tolist())
    return out


def main() -> None:
    import jax.numpy as jnp
    from openpi.models.value_model_config import ValueModelConfig

    p = argparse.ArgumentParser(description="Fill rule-3 `none` labels with the per-arm residual value model")
    p.add_argument("--data_dir", required=True, type=str)
    p.add_argument("--arm", required=True, choices=["left", "right"])
    p.add_argument("--checkpoint_dir", required=True, type=str)
    p.add_argument("--checkpoint_name", type=str, default=None)
    p.add_argument("--index_file", type=str, default=None, help="residual_idx_{arm}.npy (else computed from parquet)")
    p.add_argument("--top_percent", type=float, default=30.0,
                   help="percentile cutoff: top X%% of rule-3 frames by V -> positive, rest -> negative "
                        "(robust to value-model bias; low value => more negatives). <=0 uses --threshold.")
    p.add_argument("--threshold", type=float, default=-0.5, help="fixed V cutoff, used only if --top_percent<=0")
    p.add_argument("--use_ema", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tokenizer_path", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--instruction_col", type=str, default=None)
    p.add_argument("--base_image_col", type=str, default="observation.images.head_camera")
    p.add_argument("--wrist_image_col", type=str, default="observation.images.left_wrist_camera")
    p.add_argument("--right_wrist_image_col", type=str, default="observation.images.right_wrist_camera")
    p.add_argument("--copy_wrist_to_right", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    parquet_files = sorted(glob.glob(str(data_dir / "data" / "**" / "*.parquet"), recursive=True))
    if not parquet_files:
        raise SystemExit(f"no parquet under {data_dir/'data'}")
    adv_col = f"adv_ind_{args.arm}"

    # residual frames (global `index`) to run inference on
    if args.index_file:
        residual = set(int(i) for i in np.load(args.index_file).tolist())
    else:
        residual = _residual_global_indices(parquet_files, args.arm)
    if not residual:
        print(f"no residual (none & loss_mask=1) frames for arm={args.arm}; nothing to do")
        return
    selected = sorted(residual)
    print(f"arm={args.arm}: {len(selected)} residual frames to label")

    # --- load value model + build inference dataset (reuse label_advantage helpers) ---
    checkpoint_path = la._resolve_checkpoint_path(Path(args.checkpoint_dir), args.checkpoint_name)
    config = ValueModelConfig()
    params = la._load_checkpoint_params(checkpoint_path, use_ema=args.use_ema)
    model = config.load(params, remove_extra_params=True)
    supports = jnp.linspace(-1.0, 0.0, 201, dtype=jnp.float32)

    resolved_tok = la._resolve_local_gemma_tokenizer_path(args.tokenizer_path)
    la._validate_gemma_tokenizer(resolved_tok)
    tasks_map = la._load_tasks_map(data_dir) if (data_dir / "meta" / "tasks.jsonl").exists() else None

    dataset = la._build_inference_dataset(
        data_dir=data_dir, model_config=config, tokenizer_path=resolved_tok,
        instruction_col=args.instruction_col, base_image_col=args.base_image_col,
        wrist_image_col=args.wrist_image_col, right_wrist_image_col=args.right_wrist_image_col,
        copy_wrist_to_right=args.copy_wrist_to_right, tasks_map=tasks_map,
    )
    dataset = la.IndexedDataset(dataset, selected)
    cache = la._compute_values_with_dataloader(
        dataset=dataset, model=model, supports=supports,
        batch_size=args.batch_size, num_workers=max(0, args.num_workers), seed=args.seed,
    )
    if not cache.values_by_episode_frame:
        raise SystemExit("value inference produced no per-(episode,frame) values; check image columns")

    # --- cutoff: percentile over ALL rule-3 V (robust to the value model's
    # absolute bias from the ~20-35:1 positive-imbalanced hand labels). Top
    # --top_percent% by V -> positive, the rest -> negative. Since rule-3 frames
    # are all from FAILED episodes, a low top_percent yields the many negatives we
    # want. Falls back to a fixed --threshold only if --top_percent <= 0. ---
    all_v = np.array([v for fm in cache.values_by_episode_frame.values() for v in fm.values()], dtype=np.float32)
    use_pct = args.top_percent is not None and args.top_percent > 0
    cutoff = float(np.percentile(all_v, 100.0 - args.top_percent)) if use_pct else args.threshold
    print(f"arm={args.arm}: {len(all_v)} residual frames, V∈[{all_v.min():.3f},{all_v.max():.3f}], "
          f"cutoff={cutoff:.4f} ({'top %.0f%% -> positive' % args.top_percent if use_pct else 'fixed threshold'})")

    # --- write back: only rule-3 frames (none & loss_mask==1), only adv_ind_{arm} ---
    n_pos = n_neg = 0
    for f in parquet_files:
        df = pd.read_parquet(f)
        ep = la._extract_episode_id(df, 0)
        frame_map = cache.values_by_episode_frame.get(ep)
        if not frame_map:
            continue
        adv = _str_col(df, adv_col)
        lm = _f32_col(df, f"loss_mask_{args.arm}")
        fidx = _int_col(df, "frame_index") if "frame_index" in df.columns else np.arange(len(df))
        changed = False
        for row in range(len(df)):
            if adv[row] == "none" and lm[row] == 1.0 and int(fidx[row]) in frame_map:
                v = frame_map[int(fidx[row])]
                label = "positive" if v >= cutoff else "negative"
                adv[row] = label
                n_pos += label == "positive"
                n_neg += label == "negative"
                changed = True
        if changed and not args.dry_run:
            df[adv_col] = list(adv)
            df.to_parquet(f, index=False)

    if not args.dry_run:
        la._update_info_json(data_dir / "meta" / "info.json", adv_col,
                             description="Advantage indicator (per-arm; rule-3 filled by residual value model).")
    print(f"arm={args.arm}: filled positive={n_pos} negative={n_neg} (cutoff V>={cutoff:.4f})  "
          f"{'[dry run]' if args.dry_run else ''}")


if __name__ == "__main__":
    main()
