# pistar — RECAP / pi0.6 on YAM bimanual

JAX implementation of **RECAP** (*RL with Experience and Corrections via
Advantage-conditioned Policies*), the offline-RL algorithm behind **pi0.6**
([π★₀.₆: a VLA That Learns From Experience](https://arxiv.org/abs/2511.14759),
Physical Intelligence et al.). `pistar` is a fork of
[openpi](https://github.com/Physical-Intelligence/openpi); this repo is the
**training side** of an end-to-end RECAP pipeline on **YAM bimanual arms**.

This README documents the *full pipeline we actually run* on real hardware.
The other repos that collaborate in the stack:

- **Collection** — [`limb`](https://github.com/TToTMooN/limb): YAM control +
  DAgger sessions (AUTONOMOUS / PAUSED / CORRECTING phase machine) + serve client.
- **Conversion** — `limb convert-lerobot --pistar`: produces a LeRobot v3.0
  dataset with the five RECAP columns, then `openpi convert_v3_to_v21.py` → v2.1.
- **Initial SFT** — [`openpi`](https://github.com/Physical-Intelligence/openpi)
  (your YAM fork): the pi0.5 warm-start checkpoint that pistar fine-tunes from.
- **Training (this repo)** — Stages 3–6: pi0.6 fine-tune, VLM value model,
  VLM advantage labeling, full RECAP.
- **Evaluation** — openpi `serve_policy.py` + limb's `OpenPIClient`. pi0.6
  checkpoints serve through the standard openpi wire protocol with **no
  CFG-sampler shim**: `adv_ind` rides through the normal tokenizer.

> The mechanism in one sentence: train a VLM value model on the collected data,
> use it to classify each *autonomous* frame as high-advantage (`positive`) or
> low-advantage (`negative`), then continue fine-tuning the policy with the
> per-frame advantage class fed in as a tokenized conditioning signal
> (`adv_ind`). At inference, condition on `positive`.

---

## The six stages

| Stage | What it does | Tool | Repo |
| --- | --- | --- | --- |
| 0 | Collect DAgger rollouts (pedal + keyboard episode lifecycle) | `limb record …` | limb |
| 1 | Convert to LeRobot v3.0 + 5 RECAP columns, then v3→v2.1 | `limb convert-lerobot --pistar` + `openpi convert_v3_to_v21.py` | limb / openpi |
| 2 | Initial pi0.5 SFT on demos | `openpi/scripts/train.py` | openpi |
| **3** | **pi0.6 fine-tune from SFT, no VLM yet (limb-supplied `adv_ind`)** | `scripts/train.py` | **pistar** |
| **4** | **Train the VLM value model on `value_label`** | `scripts/train_value.py` | **pistar** |
| **5** | **Run the value model to relabel `adv_ind` on autonomous frames** | `scripts/label_advantage_from_vlm.py` | **pistar** |
| **6** | **Continue pi0.6 fine-tune on the relabeled dataset (full RECAP)** | `scripts/train.py` | **pistar** |

Stages 3–6 run in this repo and are documented below.

### The five RECAP columns

The LeRobot dataset that pistar consumes must carry these per-frame fields (in
addition to standard `observation.*` / `action` / indices). They are produced by
`limb convert-lerobot --pistar`:

| Field | Description |
| --- | --- |
| `intervention` | `1` = human/demo/correction frame, `0` = autonomous rollout frame. |
| `reward` | Sparse success reward; usually only the last frame of a successful episode is `1`. |
| `reward_label` | Dense reward used by the VLM when computing N-step advantage (`-1/T` non-terminal, `0` terminal). |
| `value_label` | Per-frame supervision target for the VLM value model, in `[-1, 0]`. |
| `adv_ind` | Advantage condition fed to the policy: `positive`, `negative`, or `none`. |

---

## Setup

### Repositories

The three repos live as siblings under one parent directory (this site assumes
`/home/ssc/Desktop/research/limb/`):

```text
limb/                 # YAM control + DAgger collection + serve client
├── openpi/           # JAX pi0.5 SFT (Stage 2)
├── pistar/           # this repo — JAX RECAP (Stages 3–6)
└── datasets/         # converted LeRobot v3.0 + v2.1 datasets
```

### pistar environment

Use a **dedicated** venv for pistar — do not share it with `openpi/` (they pin
different versions of openpi-internal modules).

```bash
git clone https://github.com/Destiny000621/RECAP.git pistar
cd pistar
git submodule update --init --recursive

uv venv ~/.venvs/pistar --python 3.11.9
source ~/.venvs/pistar/bin/activate

GIT_LFS_SKIP_SMUDGE=1 uv sync --active
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
uv pip install -r pistar_requirements.txt
```

### VLM checkpoint (for Stage 4)

The value model is initialized from a pretrained VLM bundle
(SigLIP-So400m + Gemma3-270M) distributed at
[`ybpy/vlm_ckpt`](https://huggingface.co/ybpy/vlm_ckpt) (Google Drive mirror also
available):

```bash
mkdir -p ~/Downloads/vlm_ckpt
huggingface-cli download ybpy/vlm_ckpt --local-dir ~/Downloads/vlm_ckpt
ls ~/Downloads/vlm_ckpt
# expect:
#   gemma-3-270m/                          (orbax checkpoint at step_00020000/)
#   siglip2-so400m-patch14-224-jax/
#   tokenizer.model
```

`ValueModelWeightLoader` reads `$OPENPI_VLM_CKPT_DIR` (default `~/Downloads/vlm_ckpt`)
and the orbax at `<dir>/gemma-3-270m/step_00020000/`.

### pi0.5 base weights

```bash
# Either cloud-pull on the first training step:
gcloud auth application-default login

# Or pre-download to a local mirror:
mkdir -p ~/pi05_base
gsutil -m rsync -r gs://openpi-assets/checkpoints/pi05_base ~/pi05_base
# then point CheckpointWeightLoader at "/home/<user>/pi05_base/params"
```

### Upstream patches (already applied in this repo)

Pistar `main` ships Stages 4 / 5 in an upstream-broken state. **15 targeted
patches** make them runnable; they are already applied here (local to `src/` and
`gemma/`, `openpi/` untouched). See [the patch reference](#upstream-patch-reference)
for the full list — useful if you re-base on upstream or hit one of the original
errors.

---

## Stage 3 — pi0.6 fine-tune from SFT (no VLM yet)

Take the SFT checkpoint from Stage 2 and continue training as **pi0.6** with
`pistar=True`, so the tokenizer learns to ingest `adv_ind`. At this stage we use
**limb's supplied `adv_ind`**: `positive` on intervention frames, `none` on
autonomous frames. This trains the conditioning channel end-to-end without the
VLM value model (Stages 4–5 fill those in later), and is the right first run on
small datasets where the value model would overfit.

```bash
cd /home/ssc/Desktop/research/limb/pistar
source ~/.venvs/pistar/bin/activate

# LoRA-from-SFT (single 24 GB GPU; the registered Stage 3 default)
XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  python scripts/train.py pi06_yam_vial_30fps_lora_from_sft \
    --exp-name=stage3_v0 --overwrite
```

Checkpoints land at `checkpoints/pi06_yam_vial_30fps_lora_from_sft/stage3_v0/<step>/`.

For a full fine-tune (8× H100) point a copy of `pi06_yam_vial_30fps` at your SFT
`params/` dir (see the [TrainConfig reference](#yam-trainconfig-reference); a
full `_from_sft` Stage 3 config is not registered — only `_lora_from_sft`). On
the reference 10-episode dataset Stage 3 is essentially the best you can do
without the VLM overfitting; going further requires more episodes.

To resume an existing experiment, replace `--overwrite` with `--resume`.

---

## Stage 4 — VLM value model training

Train the SigLIP-So400m + Gemma3-270M + 201-bin C51 critic head on per-frame
`value_label` supervision. Output: a value model that predicts `V(o_t)` from
`(image, wrist_image, state, prompt)`.

**Quick smoke test (5 steps, ~30 s)** — confirm the patched pipeline runs
end-to-end before committing to a long run:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
  python scripts/train_value.py \
    --data_dir /home/ssc/Desktop/research/limb/datasets/vial_rollout_v1_v21 \
    --checkpoint_dir checkpoints/value_model/yam_vial_v1 \
    --batch_size 4 --num_train_steps 5 \
    --save_interval 100 --val_interval 0 \
    --load_pretrained \
    --tokenizer_path ~/Downloads/vlm_ckpt/tokenizer.model \
    --wandb_mode disabled
```

**Real run** (reference dataset: 10 episodes, ~21k frames; ~5k steps ≈ 17 min on
a 24 GB GPU at ~0.2 s/step):

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
  python scripts/train_value.py \
    --data_dir /home/ssc/Desktop/research/limb/datasets/vial_rollout_v1_v21 \
    --checkpoint_dir checkpoints/value_model/yam_vial_v1 \
    --batch_size 4 --num_train_steps 5000 \
    --save_interval 1000 --val_interval 0 \
    --load_pretrained \
    --tokenizer_path ~/Downloads/vlm_ckpt/tokenizer.model \
    --wandb_mode disabled
```

**Paper-scale (8× H100, 30k steps, batch 64):**

```bash
accelerate launch --multi_gpu --num_processes=8 --mixed_precision=bf16 \
  $(which python) scripts/train_value.py \
    --data_dir <…> --checkpoint_dir <…> \
    --batch_size 64 --num_train_steps 30000 \
    --load_pretrained --tokenizer_path ~/Downloads/vlm_ckpt/tokenizer.model
```

Key flags:

| Flag | Default | Notes |
| --- | --- | --- |
| `--load_pretrained` | off | **Required** — invokes `ValueModelWeightLoader` against the VLM bundle. |
| `--tokenizer_path` | (auto) | Explicit path defeats pistar's hardcoded `/data/...` fallback search. |
| `--batch_size` | 32 | 4–8 on a single 24 GB GPU; 64+ on H100s. |
| `--num_train_steps` | 30000 | Bundle is already at step 20k; 5k more is plenty for small tasks. |
| `--peak_lr` | 2.5e-5 | Drop to 1e-5 if loss diverges. |
| `--freeze_mode` | `all_backbones` | Freezes SigLIP + LLM. `siglip_only` / `none` are slower, lower-bias. |
| `--use_ema` | — | Stage 5 uses `ema_params` by default. |

The training script reads `value_label` (and is back-compatible with the old
misspelled `value_lable`). A 5-step checkpoint is ~5.1 GB (SigLIP + Gemma3 +
heads + EMA + step); top-level keys are `{params, ema_params, step}`.

---

## Stage 5 — Advantage labeling (VLM relabel of `adv_ind`)

Use the Stage 4 value model to compute an N-step advantage per autonomous frame,
percentile-binarize, and write the result back into the dataset's `adv_ind`
column **in place**.

> ⚠️ This step **modifies the dataset on disk**. Always run it against a *copy*,
> not the Stage 1 original, so Stage 3 (pre-VLM) and Stage 6 (post-VLM) can both
> re-use their respective variants for comparison.

```bash
cd /home/ssc/Desktop/research/limb/datasets

# Materialize a standalone copy (cp -rL follows the v2.1 symlinks → real files)
cp -rL vial_rollout_v1_v21 vial_rollout_v1_v21_vlm_label

# Register the copy in pistar's lerobot cache so repo_id resolves
ln -sfn /home/ssc/Desktop/research/limb/datasets/vial_rollout_v1_v21_vlm_label \
        ~/.cache/huggingface/lerobot/local/vial_rollout_v1_v21_vlm_label
```

```bash
cd /home/ssc/Desktop/research/limb/pistar
source ~/.venvs/pistar/bin/activate

python scripts/label_advantage_from_vlm.py \
  --data_dir   /home/ssc/Desktop/research/limb/datasets/vial_rollout_v1_v21_vlm_label \
  --checkpoint_dir checkpoints/value_model/yam_vial_v1/step_00005000 \
  --tokenizer_path ~/Downloads/vlm_ckpt/tokenizer.model \
  --batch_size 8 \
  --lookahead 50 \
  --human_col intervention \
  --adv_col adv_ind \
  --base_image_col   observation.images.head_camera \
  --wrist_image_col  observation.images.left_wrist_camera \
  --right_wrist_image_col observation.images.right_wrist_camera \
  --use_ema
```

What it does (per the script docstring): skip all-`intervention` demo episodes;
run VLM value inference for rollout rows; compute N-step advantage
`A_t = Σ_{k=0}^{N-1} r_{t+k} + V_{t+N} − V_t`; threshold at the configured
percentile (`--positive_ratio 0.3` → top 30% become `positive`, the rest
`negative`); intervention frames stay `positive`. After a clean run, **every
autonomous frame is classified** — there should be **zero `none`** on a
rollout-only dataset (the relabel is idempotent; re-run if it crashed mid-way).

Runs on ~21k frames take ~10–12 min at batch 8 on a 24 GB GPU. Pass image
columns with dots (pistar uses dotted names verbatim, no `observation/` prefix
expansion).

---

## Stage 6 — Full RECAP fine-tune

Continue the pi0.6 fine-tune on the **VLM-labeled** dataset from Stage 5.
Autonomous frames now carry `adv_ind ∈ {positive, negative}` instead of `none`,
so the conditioning channel gets real value-graded supervision. This is the
closest match to the pi0.6 paper recipe.

```bash
cd /home/ssc/Desktop/research/limb/pistar
source ~/.venvs/pistar/bin/activate

# LoRA-from-SFT RECAP (single 24 GB GPU)
XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  python scripts/train.py pi06_yam_vial_30fps_lora_from_sft_recap \
    --exp-name=stage6_v1 --overwrite

# Full fine-tune RECAP (8× H100, paper-style, batch_size=56)
XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  python scripts/train.py pi06_yam_vial_30fps_from_sft_recap \
    --exp-name=stage6_v1 --overwrite
```

The `_recap` configs differ from their Stage 3 counterparts **only** by
`repo_id` (`local/vial_rollout_v1_v21_vlm_label`). Verify at runtime that the
log prints `repo_id='local/vial_rollout_v1_v21_vlm_label'` — if you see the
suffix-less `vial_rollout_v1_v21`, you launched the Stage 3 config by mistake.

To continue from a Stage 3 checkpoint instead of the SFT, point the
`weight_loader` at your Stage 3 `…/params` dir.

**Multi-iteration loop (paper-scale):** serve → collect new rollouts (Stage 0) →
convert + merge (Stage 1) → make a *fresh* copy (`..._vlm_label_v2`) → re-train
Stage 4 → relabel Stage 5 → add a `_recap_v2` config and re-run this stage. Each
round preserves prior datasets/checkpoints for comparison and rollback.

---

## Evaluation — serve + deploy

Because `adv_ind` rides through the standard openpi tokenizer, **no CFG-sampler
shim is required** — the same `serve_policy.py` that serves an SFT checkpoint
serves a Stage 6 RECAP checkpoint.

```bash
cd /home/ssc/Desktop/research/limb/pistar
source ~/.venvs/pistar/bin/activate

# Stage 6 full fine-tune
python scripts/serve_policy.py --port=8111 policy:checkpoint \
  --policy.config=pi06_yam_vial_30fps_from_sft_infer \
  --policy.dir=checkpoints/pi06_yam_vial_30fps_from_sft/stage6_v1/<step>

# Stage 3 LoRA-from-SFT smoke run
python scripts/serve_policy.py --port=8111 policy:checkpoint \
  --policy.config=pi06_yam_vial_30fps_lora_from_sft_infer \
  --policy.dir=checkpoints/pi06_yam_vial_30fps_lora_from_sft/stage3_v0/<step>
```

> ⚠️ **The `_infer` suffix matters.** Infer configs set `adv_ind_dropout=False`
> so the positive tag is always present at inference. Serving the non-infer
> variant randomly drops `adv_ind` ~90% of the time and silently loses the RECAP
> conditioning. LoRA checkpoints must serve through a `_lora_*_infer` config;
> full-fine-tune checkpoints through a `_from_sft_infer` config (the param trees
> differ).

On the **limb side**, `OpenPIObsTransform` must emit `adv_ind: "positive"` on
every wire observation for pistar/pi0.6 checkpoints — otherwise the server's
`TokenizePrompt` raises `ValueError: Adv_ind is required.` (the
`adv_ind_dropout=False` flag only controls the *server-side* tokenizer
randomization; the client still has to send the field). Then drive YAM with
`limb teleop` / `limb record` as usual. An evaluation run is **operator-passive**:
observe the policy and label success/failure; do not intervene.

---

## YAM TrainConfig reference

Ten pi0.6 configs are registered in `src/openpi/training/config.py` (paired
train/`_infer`). All share `Pi0Config(pi05=True, pistar=True)`, the 3-camera
Aloha-style repack (`cam_high` / `cam_left_wrist` / `cam_right_wrist`),
`adapt_to_pi=False` (YAM joint conventions, *not* Trossen Aloha), and the YAM
vial-handover `default_prompt`. Each train/`_infer` pair differs only in
`adv_ind_dropout` (`True` for training, `False` for serving).

| Config | Variant | Init weights | Dataset (`repo_id`) | Stage |
| --- | --- | --- | --- | --- |
| `pi06_yam_vial_30fps` | full | `pi05_base` | `local/vial_rollout_v1_v21` | 3 (full alt.) |
| `pi06_yam_vial_30fps_lora` | LoRA | `pi05_base` | `local/vial_rollout_v1_v21` | 3 (LoRA alt.) |
| `pi06_yam_vial_30fps_lora_from_sft` | LoRA | **SFT** | `local/vial_rollout_v1_v21` | **3 (default)** |
| `pi06_yam_vial_30fps_lora_from_sft_recap` | LoRA | **SFT** | `local/vial_rollout_v1_v21_vlm_label` | **6 (default)** |
| `pi06_yam_vial_30fps_from_sft_recap` | **full** | **SFT** | `local/vial_rollout_v1_v21_vlm_label` | **6 (8× H100)** |

Each has a matching `_infer` variant (`adv_ind_dropout=False`) for serving.

**Picking one:**

| Situation | Config |
| --- | --- |
| Single 24 GB GPU, reproduce Stage 3 | `pi06_yam_vial_30fps_lora_from_sft` |
| Single 24 GB GPU, reproduce Stage 6 (RECAP) | `pi06_yam_vial_30fps_lora_from_sft_recap` |
| 8× H100, paper-style RECAP | `pi06_yam_vial_30fps_from_sft_recap` |
| Pretraining from `pi05_base` (skip SFT) | `pi06_yam_vial_30fps` (full) or `_lora` |
| Serving any of the above | the matching `_infer` config |

> A note on scale (pi0.6 paper, App. A-F): the paper uses **287–450 correction
> episodes per iteration**. On ~10 episodes the VLM value model overfits and
> Stages 4–5 add little beyond Stage 3; at ~100 it starts to matter; at ~300+ it
> matches the paper's regime. Default to **full fine-tuning**; the LoRA variants
> are for single-GPU development and smoke tests.

---

## Data utilities

`scripts/merge_datasets.py` merges demo and rollout datasets that are already in
the pistar LeRobot schema. It only keeps the five RECAP columns plus
`timestamp`, `frame_index`, `episode_index`, `index`, `task_index`. It is a pure
merge — it does **not** fill missing fields, recompute labels, or convert image
layout. Re-convert a source dataset before merging if it is missing fields.

```bash
python scripts/merge_datasets.py \
  --sources \
    /path/to/datasets/libero_demo_pistar \
    /path/to/datasets/libero_rollout_round1 \
  --output /path/to/datasets/libero_mixed_round1 \
  --overwrite
```

`scripts/compute_norm_stats.py <config>` computes normalization statistics
before training a policy config.

---

## Upstream patch reference

Stages 4 / 5 are upstream-broken on pistar `main`. The **15 patches** below are
already applied in this repo (local to `src/openpi/` and `gemma/`; `openpi/` is
untouched). Patches 1–13 unblock Stage 4 (`train_value.py`); 14–15 apply the
same fixes to Stage 5 (`label_advantage_from_vlm.py`, which ships its own
duplicate copies of the data-config block and `GemmaValueTokenizer`).

| # | Symptom on `main` | File | Fix |
| --- | --- | --- | --- |
| 1 | `ImportError: cannot import name 'ValueModelWeightLoader'` | `src/openpi/training/weight_loaders.py` | add `ValueModelWeightLoader` class |
| 2 | `ModuleNotFoundError: No module named 'gemma.gm.data'` | `gemma/gemma/gm/data/` | copy missing dir from upstream gemma |
| 3 | `ModuleNotFoundError: No module named 'kauldron.ktyping'` | `gemma/gemma/gm/data/{_functional,_transforms}.py` | `kauldron.ktyping` → `kauldron.typing` |
| 4 | `ImportError: cannot import name 'ContextStack' from 'etils.edc'` | `gemma/gemma/gm/utils/_dtype_params.py` | remove broken top-level import |
| 5 | `AttributeError: 'etils.edc' has no attribute 'ContextStack'` | `gemma/gemma/gm/utils/_dtype_params.py` | local `_ContextStack(list)` fallback |
| 6 | `ImportError: cannot import name 'console' from 'openpi.shared'` | `src/openpi/shared/console.py` (new) | `info/ok/warn/error/bold` helpers |
| 7 | `ImportError: cannot import name 'progress' from 'openpi.shared'` | `src/openpi/shared/progress.py` (new) | `sync_pbar_color` no-op stub |
| 8 | `TypeError: DataConfig.__init__() unexpected kwarg 'local_data_dir'` | `scripts/train_value.py` | derive `repo_id` from path basename |
| 9 | `KeyError: 'actions'` (lerobot delta_timestamps on missing column) | `scripts/train_value.py` | pass `action_sequence_keys=()` |
| 10 | `AttributeError: data_loader has no 'create_value_data_loader'` | `src/openpi/training/data_loader.py` | add `create_value_data_loader` (action_horizon=1) |
| 11 | `DataLoaderImpl` missing `.dataset` / `__len__` | `src/openpi/training/data_loader.py` | store `_dataset`, add `dataset` property + `__len__` |
| 12 | `TypeError: Cannot interpret TrainState as an abstract array` | `scripts/train_value.py` | `TrainState` → `flax.struct.PyTreeNode` |
| 13 | `KeyError: 'actions'` in `__iter__`; tqdm timedelta; `tokenize()` extra kwarg | `src/openpi/training/data_loader.py` + `scripts/train_value.py` | `_ValueDataLoaderImpl` yields `(obs, value)`; `int(step)`; `**_ignored` on `tokenize` |
| 14 | `TypeError: DataConfig.__init__() unexpected kwarg 'local_data_dir'` (Stage 5) | `scripts/label_advantage_from_vlm.py` | same as 8/9 in `_build_inference_dataset` |
| 15 | `TypeError: GemmaValueTokenizer.tokenize() unexpected kwarg 'adv_ind_dropout'` (Stage 5) | `scripts/label_advantage_from_vlm.py` | `**_ignored` on the duplicate `GemmaValueTokenizer.tokenize` |

`ValueModelWeightLoader` resolves the VLM bundle via `$OPENPI_VLM_CKPT_DIR`
(default `~/Downloads/vlm_ckpt`), reads the orbax at
`<dir>/gemma-3-270m/step_00020000/`, and selects `ema_params` vs `params` via
`use_ema`.

---

## References

- pi0.6 / RECAP paper: [π★₀.₆: a VLA That Learns From Experience](https://arxiv.org/abs/2511.14759)
- Reference RECAP pipeline (sim-only, LIBERO): [RLinf RECAP page](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/recap.html)
- openpi (upstream): https://github.com/Physical-Intelligence/openpi
- VLM value-model checkpoint: [`ybpy/vlm_ckpt`](https://huggingface.co/ybpy/vlm_ckpt)
