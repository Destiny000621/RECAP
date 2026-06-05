import dataclasses
import logging
import os
import pathlib
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import jax
import numpy as np
import orbax.checkpoint as ocp

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


@dataclasses.dataclass(frozen=True)
class ValueModelWeightLoader(WeightLoader):
    """Loads pretrained VLM weights for the pistar ValueModel.

    The bundle is the ybpy/vlm_ckpt distribution
    (https://huggingface.co/ybpy/vlm_ckpt). Expected on-disk layout::

        <vlm_ckpt_dir>/
        ├── gemma-3-270m/
        │   ├── step_00020000/         # orbax CheckpointManager save
        │   │   └── (params/ ema_params/ step subtrees at the top level)
        │   └── ... (CheckpointManager bookkeeping files)
        ├── siglip2-so400m-patch14-224-jax/  (consumed by SigLIP loader path if needed)
        └── tokenizer.model            (consumed by --tokenizer_path, not here)

    The orbax save contains a full ValueModel snapshot — SigLIP image tower,
    img_projection, Gemma3-270M LLM, cross-attention, value head — so we
    restore the ``params`` subtree and merge it into the freshly-initialized
    ValueModel param tree.

    Path resolution: ``vlm_ckpt_dir`` defaults to ``~/Downloads/vlm_ckpt``;
    override via the ``OPENPI_VLM_CKPT_DIR`` environment variable, or by
    instantiating ``ValueModelWeightLoader(vlm_ckpt_dir=...)``.

    NOTE: this loader is called from ``scripts/train_value.py`` as
    ``ValueModelWeightLoader()`` with no arguments. To override the path
    without editing that script, set ``OPENPI_VLM_CKPT_DIR``.
    """

    vlm_ckpt_dir: str = dataclasses.field(
        default_factory=lambda: os.environ.get(
            "OPENPI_VLM_CKPT_DIR", str(pathlib.Path("~/Downloads/vlm_ckpt").expanduser())
        )
    )
    # The bundle has both `params` (live) and `ema_params` (EMA-smoothed).
    # `params` is the typical choice for further fine-tuning; `ema_params` is
    # what you'd serve at inference. Override only if you know why.
    use_ema: bool = False

    def load(self, params: at.Params) -> at.Params:
        ckpt_path = pathlib.Path(self.vlm_ckpt_dir) / "gemma-3-270m" / "step_00020000"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"VLM checkpoint not found at {ckpt_path}. Download the bundle from "
                f"https://huggingface.co/ybpy/vlm_ckpt and place it at "
                f"{self.vlm_ckpt_dir!r}, or set the OPENPI_VLM_CKPT_DIR env var."
            )

        # Use a fully-replicated single-device sharding so the restore works on
        # any GPU count. We deserialize as np.ndarray and let the training code
        # shard / convert dtypes downstream.
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

        params_key = "ema_params" if self.use_ema else "params"

        with ocp.PyTreeCheckpointer() as ckptr:
            metadata = ckptr.metadata(ckpt_path)
            if params_key not in metadata:
                raise KeyError(
                    f"Bundle at {ckpt_path} doesn't have a {params_key!r} subtree; "
                    f"available top-level keys: {list(metadata.keys())}"
                )

            # orbax does a tree-structure match against the on-disk metadata,
            # so we must mirror every top-level key (params, ema_params, step).
            item = dict(metadata)
            restore_args = jax.tree.map(
                lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=np.ndarray),
                item,
            )

            restored = ckptr.restore(
                ckpt_path,
                ocp.args.PyTreeRestore(item=item, restore_args=restore_args),
            )

        loaded = restored[params_key]

        # If saved via nnx.State, every leaf key path ends with "value"; strip it
        # the same way openpi.models.model.restore_params does.
        flat = flax.traverse_util.flatten_dict(loaded)
        if flat and all(kp[-1] == "value" for kp in flat):
            flat = {kp[:-1]: v for kp, v in flat.items()}
            loaded = flax.traverse_util.unflatten_dict(flat)

        logger.info(
            "ValueModelWeightLoader: restored %d leaf arrays from %s (key=%s, step=%s)",
            sum(1 for _ in flax.traverse_util.flatten_dict(loaded)),
            ckpt_path,
            params_key,
            restored.get("step"),
        )

        # Merge: prefer loaded weights for any matching keys; keep the model's
        # fresh init for anything not in the bundle (e.g. value-head dropout
        # variants, future extensions). Same pattern as the other loaders.
        return _merge_params(loaded, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
