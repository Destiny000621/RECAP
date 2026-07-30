"""Visualize a RECAP lerobot dataset in rerun, with the VLM label columns
(adv_ind / value_label / reward / reward_label / intervention / phase) logged as
synced timelines alongside the camera video.

Headless usage (remote server): save an .rrd, scp it, open locally with `rerun`.

    HF_LEROBOT_HOME=/mnt/localssd/Sichang/lerobot_home \
      uv run python scripts/viz_recap_labels.py \
        --repo-id local/vials_recap_v1_v21 \
        --episodes 0 7 12 \
        --output-dir viz_rrd
"""

import argparse
import gc
from pathlib import Path

import rerun as rr
import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.visualize_dataset import EpisodeSampler, to_hwc_uint8_numpy

ADV_MAP = {"positive": 1.0, "negative": 0.0}


def _scalar(value: float):
    # rerun 0.23 renamed Scalar -> Scalars
    return rr.Scalars(value) if hasattr(rr, "Scalars") else rr.Scalar(value)


def viz_episode(dataset, repo_id, episode_index, output_dir, cameras, num_workers):
    sampler = EpisodeSampler(dataset, episode_index)
    loader = torch.utils.data.DataLoader(
        dataset, num_workers=num_workers, batch_size=32, sampler=sampler
    )
    rr.init(f"{repo_id}/episode_{episode_index}", spawn=False)
    gc.collect()

    for batch in loader:
        n = len(batch["index"])
        for i in range(n):
            rr.set_time("frame_index", sequence=batch["frame_index"][i].item())
            rr.set_time("timestamp", duration=batch["timestamp"][i].item())

            for key in cameras:
                rr.log(key, rr.Image(to_hwc_uint8_numpy(batch[key][i])).compress(jpeg_quality=75))

            # numeric label timelines
            for col in ("value_label", "reward", "reward_label"):
                if col in batch:
                    rr.log(f"labels/{col}", _scalar(batch[col][i].item()))
            if "intervention" in batch:
                rr.log("labels/intervention", _scalar(float(batch["intervention"][i].item())))

            # adv_ind: string -> 1/0 timeline + text
            if "adv_ind" in batch:
                adv = batch["adv_ind"][i]
                adv = adv if isinstance(adv, str) else str(adv)
                rr.log("labels/adv_positive", _scalar(ADV_MAP.get(adv, float("nan"))))
                rr.log("labels/adv_ind_text", rr.TextLog(adv))
            if "phase" in batch:
                phase = batch["phase"][i]
                rr.log("labels/phase", rr.TextLog(phase if isinstance(phase, str) else str(phase)))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rrd = output_dir / f"{repo_id.replace('/', '_')}_episode_{episode_index}.rrd"
    rr.save(rrd)
    return rrd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True)
    p.add_argument("--episodes", type=int, nargs="+", required=True)
    p.add_argument("--output-dir", default="viz_rrd")
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="0 (default) avoids a rerun+DataLoader multiprocessing deadlock; >0 is faster but can hang.",
    )
    p.add_argument(
        "--cameras",
        nargs="*",
        default=["observation.images.head_camera"],
        help="Camera keys to log. Pass 'all' for every camera (bigger .rrd).",
    )
    args = p.parse_args()

    dataset = LeRobotDataset(args.repo_id)
    cameras = dataset.meta.camera_keys if args.cameras == ["all"] else args.cameras

    for ep in args.episodes:
        rrd = viz_episode(dataset, args.repo_id, ep, args.output_dir, cameras, args.num_workers)
        size_mb = rrd.stat().st_size / 2**20
        print(f"saved {rrd}  ({size_mb:.1f} MiB)")


if __name__ == "__main__":
    main()
