"""Prepare per-arm residual-labeler training targets + inference index sets.

The rule-based labeler (label_advantage_handcrafted.py) labels rule-1/2 frames
`positive`/`negative` and leaves rule-3 (failure-without-intervention) frames
`none`. This script sets up the Stage-4 VLM residual labeler that fills those:

  TRAIN on rule-1/2 frames (per arm), supervised by adv_ind -> {positive, negative}
  APPLY to rule-3 frames = (adv_ind == "none") AND (loss_mask == 1)

For each arm it writes two columns into the v2.1 dataset (in place):
  cls_value_{arm}  : 0.0 if adv_ind_{arm}=="positive", -1.0 if "negative", else 0.0
                     (the value-model C51 support is [-1,0]; positive->top bin,
                     negative->bottom bin)
  cls_mask_{arm}   : 1.0 if adv_ind_{arm} in {positive,negative} else 0.0
                     (train_value.py zeroes the target distribution where mask=0,
                     so `none` frames contribute zero loss/gradient)

and emits, under --out_dir, the global-`index` arrays used downstream:
  residual_idx_{arm}.npy : frames to FILL at inference (none & loss_mask==1)
  train_count_{arm}.txt  : #positive / #negative (sanity)

`loss_mask==1` on a `none` frame excludes the frozen-arm `none` frames (those are
already masked from the policy loss); only genuine rule-3 autonomous failures
remain. Run AFTER add_per_arm_labels.py + label_advantage_handcrafted.py.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def _str_col(df: pd.DataFrame, c: str) -> np.ndarray:
    return np.array([str(v) for v in df[c].tolist()], dtype=object)


def _f32_col(df: pd.DataFrame, c: str) -> np.ndarray:
    return np.array([float(np.asarray(v).reshape(-1)[0]) for v in df[c].tolist()], dtype=np.float32)


def _int_col(df: pd.DataFrame, c: str) -> np.ndarray:
    return np.array([int(np.asarray(v).reshape(-1)[0]) for v in df[c].tolist()], dtype=np.int64)


def main() -> None:
    p = argparse.ArgumentParser(description="Prep per-arm residual-labeler targets + index sets")
    p.add_argument("--data_dir", required=True, type=str, help="labeled v2.1 dataset")
    p.add_argument("--out_dir", required=True, type=str, help="where to write residual_idx_{arm}.npy")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_files = sorted(glob.glob(str(data_dir / "data" / "**" / "*.parquet"), recursive=True))
    if not parquet_files:
        raise SystemExit(f"no parquet under {data_dir/'data'}")

    train = {"left": Counter(), "right": Counter()}
    residual_idx = {"left": [], "right": []}

    for f in parquet_files:
        df = pd.read_parquet(f)
        for col in ("adv_ind_left", "adv_ind_right", "loss_mask_left", "loss_mask_right", "index"):
            if col not in df.columns:
                raise SystemExit(f"missing column '{col}' — run add_per_arm_labels + label_advantage_handcrafted first")
        gidx = _int_col(df, "index")
        new_cols = {}
        for arm in ("left", "right"):
            adv = _str_col(df, f"adv_ind_{arm}")
            lm = _f32_col(df, f"loss_mask_{arm}")
            cls_value = np.where(adv == "negative", -1.0, 0.0).astype(np.float32)  # positive/none -> 0.0
            cls_mask = np.isin(adv, ("positive", "negative")).astype(np.float32)
            new_cols[f"cls_value_{arm}"] = cls_value
            new_cols[f"cls_mask_{arm}"] = cls_mask
            train[arm].update(adv[cls_mask > 0].tolist())
            residual_idx[arm].extend(gidx[(adv == "none") & (lm == 1.0)].tolist())
        if not args.dry_run:
            for name, arr in new_cols.items():
                df[name] = arr
            df.to_parquet(f, index=False)

    if not args.dry_run:
        info_path = data_dir / "meta" / "info.json"
        if info_path.exists():
            info = json.loads(info_path.read_text())
            feats = info.setdefault("features", {})
            for arm in ("left", "right"):
                for k in (f"cls_value_{arm}", f"cls_mask_{arm}"):
                    feats.setdefault(k, {"dtype": "float32", "shape": [1], "names": None})
            info_path.write_text(json.dumps(info, indent=4))
        for arm in ("left", "right"):
            idx = np.asarray(sorted(residual_idx[arm]), dtype=np.int64)
            np.save(out_dir / f"residual_idx_{arm}.npy", idx)
            (out_dir / f"train_count_{arm}.txt").write_text(
                f"positive={train[arm]['positive']} negative={train[arm]['negative']}\n"
            )

    for arm in ("left", "right"):
        t = train[arm]
        print(f"{arm:5s}: train positive={t['positive']} negative={t['negative']}  "
              f"residual(none&mask=1)={len(residual_idx[arm])}")
    print("DRY RUN — nothing written." if args.dry_run else f"written: cls_* columns + residual_idx_*.npy under {out_dir}")


if __name__ == "__main__":
    main()
