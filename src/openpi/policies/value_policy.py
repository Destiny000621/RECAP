import dataclasses
import io

import numpy as np

from openpi import transforms


def _get_by_path(data: dict, path: str):
    # Direct key match first.
    if path in data:
        return data[path], True

    def _traverse(parts: list[str]):
        cur = data
        for part in parts:
            if not isinstance(cur, dict) or part not in cur:
                return None, False
            cur = cur[part]
        return cur, True

    if "/" in path:
        val, ok = _traverse(path.split("/"))
        if ok:
            return val, True
    if "." in path:
        val, ok = _traverse(path.split("."))
        if ok:
            return val, True
    return None, False


def _get_first(data: dict, candidates: list[str], *, required: bool = False, name: str = "value"):
    for key in candidates:
        val, ok = _get_by_path(data, key)
        if ok:
            return val
    if required:
        raise KeyError(f"Missing {name}. Tried keys: {candidates}. Available top-level keys: {list(data.keys())}")
    return None


def _parse_image(image) -> np.ndarray:
    if image is None:
        raise ValueError("Image is None")

    if isinstance(image, dict) and "bytes" in image:
        image = image["bytes"]

    if isinstance(image, dict) and "path" in image and image["path"] is not None:
        image = image["path"]

    if isinstance(image, (bytes, bytearray)):
        from PIL import Image

        image = Image.open(io.BytesIO(image)).convert("RGB")
        return np.asarray(image)

    if isinstance(image, str):
        from PIL import Image

        image = Image.open(image).convert("RGB")
        return np.asarray(image)

    try:
        import torch
    except Exception:
        torch = None

    if torch is not None and isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    return image


def _normalize_scalar_label(value, *, name: str) -> np.float32:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        return np.float32(array)
    if array.size == 1:
        return np.float32(array.reshape(()))
    raise ValueError(f"{name} must be scalar-like, got shape {array.shape}")


@dataclasses.dataclass(frozen=True)
class ValueInputs(transforms.DataTransformFn):
    """将 LeRobot 数据转换为 Value Model 期望的输入格式。"""

    def __call__(self, data: dict) -> dict:
        base_candidates = [
            "observation/image",
            "observation.image",
            "image",
            "observation/images/cam_high",
            "observation.images.cam_high",
            "images/cam_high",
            "images.cam_high",
            "observation/images/base_0_rgb",
            "observation.images.base_0_rgb",
            "images/base_0_rgb",
            "images.base_0_rgb",
            # YAM bimanual (limb convert-lerobot --pistar default names)
            "observation/images/head_camera",
            "observation.images.head_camera",
            "images/head_camera",
            "images.head_camera",
        ]
        wrist_candidates = [
            "observation/wrist_image",
            "observation.wrist_image",
            "wrist_image",
            "observation/images/cam_left_wrist",
            "observation.images.cam_left_wrist",
            "observation/images/cam_right_wrist",
            "observation.images.cam_right_wrist",
            "images/cam_left_wrist",
            "images.cam_left_wrist",
            "images/cam_right_wrist",
            "images.cam_right_wrist",
            "observation/wrist_image_left",
            "observation.wrist_image_left",
            "observation/wrist_image_right",
            "observation.wrist_image_right",
            # YAM bimanual (limb convert-lerobot --pistar default names)
            "observation/images/left_wrist_camera",
            "observation.images.left_wrist_camera",
            "observation/images/right_wrist_camera",
            "observation.images.right_wrist_camera",
            "images/left_wrist_camera",
            "images.left_wrist_camera",
            "images/right_wrist_camera",
            "images.right_wrist_camera",
        ]

        base_raw = _get_first(
            data,
            base_candidates,
            required=False,
            name="base_image",
        )
        wrist_raw = _get_first(
            data,
            wrist_candidates,
            required=False,
            name="wrist_image",
        )
        has_base = base_raw is not None
        if has_base and isinstance(base_raw, float) and np.isnan(base_raw):
            has_base = False
        has_wrist = wrist_raw is not None
        if has_wrist and isinstance(wrist_raw, float) and np.isnan(wrist_raw):
            has_wrist = False

        if not has_base and not has_wrist:
            raise KeyError(
                f"Missing base_image. Tried keys: {base_candidates}. "
                f"Fallback wrist_image keys: {wrist_candidates}. "
                f"Available top-level keys: {list(data.keys())}"
            )

        # 单相机数据集：没有 base_image 时，退化为把 wrist_image 当作唯一视觉输入。
        base_source = wrist_raw if not has_base else base_raw
        base_image = _parse_image(base_source)

        if has_wrist and has_base:
            wrist_image = _parse_image(wrist_raw)
            use_wrist = True
        else:
            wrist_image = np.zeros_like(base_image)
            use_wrist = False

        state = _get_first(
            data,
            [
                "observation/state",
                "observation.state",
                "state",
            ],
            required=False,
            name="state",
        )
        if state is None:
            state = np.zeros((1,), dtype=np.float32)

        prompt = _get_first(
            data,
            [
                "prompt",
                "task",
                "observation/task",
                "observation.task",
            ],
            required=False,
            name="prompt",
        )

        value = _get_first(
            data,
            [
                "value",
                "reward",
                "return",
            ],
            required=True,
            name="value",
        )

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "wrist_0_rgb": wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "wrist_0_rgb": np.True_ if use_wrist else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if prompt is not None:
            inputs["prompt"] = prompt
        if value is not None:
            inputs["value"] = _normalize_scalar_label(value, name="value")

        return inputs
