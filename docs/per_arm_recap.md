# Per-arm RECAP for YAM bimanual (vials pick-and-insert)

This documents the modifications that make RECAP **per-arm** on the YAM dual-arm
vials task, why they were needed, and how to run the pipeline end to end.

---

## 1. The problem

Standard RECAP conditions the policy on a **single, frame-level `adv_ind`** token
that is applied to the **joint 14-dim action** (left = dims `0:7`, right = `7:14`;
6 joints + gripper each) and labels **every human-intervention frame `positive`**.

On YAM this breaks, because a DAgger correction usually fixes **one arm** while
the other is **frozen / held still** by the operator:

- The frozen arm's static action is regressed as a "positive" demonstration.
- Conditioned on `positive` at inference, the policy reproduces the most common
  pattern in the positive set — *an arm holding still* — and **collapses to
  pausing**. Post-RECAP policy was *worse* than the SFT baseline.

Measured on `vials_recap_v1_1_v21` (93 episodes, 105,592 frames): of **26,922**
intervention frames, **24,503 had one arm frozen** (17,010 right-corrected /
left-frozen, 7,493 left-corrected / right-frozen, 2,419 genuinely both). Those
24.5k frozen-arm frames were all wrongly `positive`.

A second, deeper issue: **advantage is per-arm**. In one frame the left arm can be
doing well while the right is failing (left SFT success >70%, right <40%). A single
scalar advantage cannot represent that.

## 2. The fix, in one line

Make the entire credit-assignment signal **per-arm**: two advantage tokens, a
per-arm action **loss mask**, and **rule-based** per-arm labels derived from the
intervention structure (the VLM value model is kept only for the residual case).

Two independent guarantees protect against the frozen-arm collapse:

1. **Loss mask** — the frozen arm's action dims are dropped from the flow loss
   (zero gradient), so its held pose is *never regressed*. Hard guarantee.
2. **Labeling** — the frozen arm is *not* labeled `positive` (set to `none`), so
   frozen poses never enter the positive conditioning bucket.

The "hold still while the other arm inserts" behavior (collision-avoidance
serialization) is still learned — from **successful autonomous rollouts**, where
the waiting arm's stillness is its own policy output rather than a teleop artifact.

## 3. New dataset columns

Written into the LeRobot v2.1 parquet (per frame) by the scripts in §5:

| Column | Type | Meaning |
| --- | --- | --- |
| `corrected_arm` | str | `none` / `left` / `right` / `both` — detector verdict per frame |
| `loss_mask_left`, `loss_mask_right` | f32 {0,1} | 0 = do not supervise this arm's action on this frame (frozen) |
| `adv_ind_left`, `adv_ind_right` | str | per-arm advantage condition: `positive` / `negative` / `none` |
| `value_label_left`, `value_label_right` | f32 [-1,0] | *optional* per-arm value target (only if a value model is used) |

`value_label_*` is binary because each arm is assigned one vial (`N=1`): `-1`
before that arm seats its vial, `0` after. It fits the C51 support `[-1, 0]`.

## 4. Labeling rules (rule-based, replaces VLM Stage 5 for most data)

The intervention itself is the supervision: a correction means the policy was
about to fail, so the corrected arm's behavior *just before* the correction is
negative. Per arm `a`, per rollout:

| Situation | Label for arm `a` |
| --- | --- |
| success autonomous rollout (no intervention) | `positive` everywhere |
| correction attributed to `a` (a corrected) | `positive` on correction frames; **`negative`** on the `neg_window` (~45 frames = 1.5 s) just **before** the correction starts |
| correction attributed to the **other** arm (a frozen) | `none` on those frames **and** `loss_mask_a = 0` (not learned) |
| `both` correction | both arms `negative` in the pre-window, `positive` on correction frames |
| failure autonomous rollout, `a` failed | `none` → handed to a VLM labeler |

Per-arm **success** (rule-1 vs rule-3) comes from `lerobot_annotator` **summary
mode** (`vials[].insert_arm`, 5-vote majority). Insert *frame* timing is **not**
needed — the rules key off intervention timing, not the seat frame.

## 5. Scripts (`scripts/`)

| Script | Purpose |
| --- | --- |
| `add_per_arm_labels.py` | Corrected-arm detector (contiguous `intervention` runs → per-arm motion energy; ratio < 0.25 ⇒ one-arm, ~95% clean) → writes `corrected_arm`, `loss_mask_{left,right}`, `adv_ind_{left,right}` scaffold, optional `value_label_{left,right}`. `--dry_run`, `--emit_skeleton`. |
| `label_advantage_handcrafted.py` | Applies the §4 rules → overwrites `adv_ind_{left,right}`. Per-arm success from `--success_json` (recommended; built by `prep_success_map.py` = reward + manual) or `--reward_success` (coarse) or `--summary_dir` (unreliable — summary over-reports success). `--neg_window 45`. |
| `detect_insert_frames.py` | SigLIP zero-shot wrist-cam insert-frame detector. **Shelved** — timing is no longer needed for labeling; kept for value-model / VLM-training use (has a threshold bug to fix if revived). |
| `prep_residual_training.py` | Derives per-arm classifier targets `cls_value_{arm}` (0.0/-1.0) + `cls_mask_{arm}` (1 on rule-1/2 frames) from `adv_ind_{arm}`, and emits `residual_idx_{arm}.npy` (the rule-3 frames = `none` & `loss_mask==1`). Pure pandas. |
| `train_value.py --arm {left,right}` | Trains the per-arm residual value model on the rule-1/2 hand labels (`cls_value_*`); `none` frames get a zero target distribution ⇒ zero loss/gradient (no loader change). Run once per arm. |
| `label_residual_from_vlm.py --arm {left,right}` | Applies the per-arm value model to the rule-3 frames; `V > --threshold` (default -0.5) → `positive` else `negative`; writes only `adv_ind_{arm}` on those frames. Reuses `label_advantage_from_vlm.py` inference helpers. |

The corrected-arm detector and masks need **no annotation**; only `value_label`
and rule-1/3 classification need the summary-mode success map.

## 6. Model wiring (`src/openpi/`)

All changes are **backward-compatible**: absent the new columns, training is
byte-identical to before.

| File | Change |
| --- | --- |
| `models/model.py` | `Observation.action_mask` field (`*b s`, optional) + `from_dict` pickup |
| `models/pi0.py` | `compute_loss` uses a **masked mean** over action dims when `action_mask` is set: `sum(sq·m)/clip(sum(m),1)`. All-ones ⇒ plain mean; gradient is **zero on masked dims** (verified). |
| `models/tokenizer.py` | `PaligemmaTokenizer.tokenize` emits `", Left advantage: {l}, Right advantage: {r}"` when `adv_ind_left/right` are given (per-arm takes precedence over single `adv_ind`); 30% dropout drops the whole clause; empty clause reproduces the no-adv prompt exactly |
| `transforms.py` | `TokenizePrompt` pops/forwards `adv_ind_left/right` (coerced to str, no strings leak to JAX); `PadStatesAndActions` pads `action_mask` to the model action dim with **1.0** (padded dims stay supervised) |
| `policies/aloha_policy.py` | `AlohaInputs` builds the 14-dim `action_mask` from `loss_mask_{left,right}` and forwards `adv_ind_{left,right}` |
| `training/config.py` | `pi06_yam_vial_30fps_from_sft_recap` repack maps the per-arm columns (`adv_ind_{left,right}`, `loss_mask_{left,right}`) in place of the single `adv_ind` |

Data flow: `RepackTransform` (column → canonical) → `AlohaInputs` (build
`action_mask`, forward `adv_ind_*`) → `ResizeImages` → `TokenizePrompt` (two adv
tokens) → `PadStatesAndActions` (pad mask) → `Normalize` (touches only
state/actions) → `Observation.from_dict` → `Pi0.compute_loss` (masked mean).

## 7. End-to-end pipeline

The recommended pipeline runs **entirely on the v2.1 dataset**
`Sichang0621/vials_recap_v1_1_v21` (openpi needs v2.1). Per-arm success comes from
the `reward` column + a quick manual pass on failures — **no annotator, no v3.0
dataset needed.**

> The v3.0 dataset `Sichang0621/vials_recap_v1.1` + `lerobot_annotator` are only
> for the *optional* full-automation success route (fixing summary mode's `seated`
> field — see `lerobot_annotator/docs/summary_mode_for_failure_episodes.md`). That
> route bridges v3.0↔v2.1 by `episode_index` (v2.1 is converted from v3.0);
> `label_advantage_handcrafted.py --summary_dir` cross-checks each episode's
> `length` to catch any misalignment. The reward+manual path below avoids all of
> this.

```bash
# 0) per-arm SUCCESS map. Summary mode is UNRELIABLE for success — it measures
#    vials-present/attribution, not seated success (33/93 reward-failures still
#    reported total_vials>0; e.g. ep2). So use reward + manual for the failures:
cd /mnt/localssd/Sichang/recap
.venv/bin/python scripts/prep_success_map.py --data_dir <v2.1_dataset> \
  --out success_map.json --frames_dir success_review
#   reward=1 (60 eps) -> both arms true (auto); reward=0 (33 eps) -> flagged, final
#   head-cam frame extracted to success_review/. Glance + edit left/right seated in
#   success_map.json (unedited failures stay both-false -> both -> VLM, which is safe).
#   (Full automation route: fix summary's `seated` field — see
#    lerobot_annotator/docs/summary_mode_for_failure_episodes.md.)

# 1) label a COPY of the v2.1 dataset (keep the pre-RECAP original intact).
cp -rL <v2.1_dataset> <v2.1_dataset>_perarm
.venv/bin/python scripts/add_per_arm_labels.py --data_dir <v2.1_dataset>_perarm
.venv/bin/python scripts/label_advantage_handcrafted.py --data_dir <v2.1_dataset>_perarm \
  --success_json success_map.json

# 2) (optional) rule-3 residual labeler: fill the `none` failure frames per arm
.venv/bin/python scripts/prep_residual_training.py --data_dir <v2.1_dataset>_perarm --out_dir residual_idx
for ARM in left right; do
  .venv/bin/python scripts/train_value.py --arm $ARM \
    --data_dir <v2.1_dataset>_perarm --checkpoint_dir checkpoints/value_model/residual_$ARM \
    --batch_size 64 --num_train_steps 6000 --load_pretrained --tokenizer_path <gemma tokenizer.model>
  .venv/bin/python scripts/label_residual_from_vlm.py --arm $ARM \
    --data_dir <v2.1_dataset>_perarm \
    --checkpoint_dir checkpoints/value_model/residual_$ARM/step_00006000 \
    --index_file residual_idx/residual_idx_$ARM.npy --tokenizer_path <gemma tokenizer.model> --use_ema
done

# 3) norm stats + RECAP fine-tune (v2.1)
.venv/bin/python scripts/compute_norm_stats.py --config-name pi06_yam_vial_30fps_from_sft_recap
.venv/bin/python scripts/train.py pi06_yam_vial_30fps_from_sft_recap --exp-name=perarm_v1 --overwrite
```

## 7b. Residual-action option (Flavor A — design, confirmed)

Optional `action_mode: residual` (default `absolute`): the pi0.6 actor learns a
**residual on the frozen pi0.5 base**, `a = b(o) + Δ(o)`. Rationale: the SFT base
already solves the **left** arm (>70%), so its residual stays ≈0 and the good
behavior is preserved by construction (complements the frozen-arm masking); the
weak **right** arm (<40%) only has to learn the *corrections* the interventions
demonstrate — a much smaller, sample-efficient target.

**Deterministic base (the correctness crux).** `b(o)` must be identical at
training and inference. `Pi0.sample_actions(rng, obs, *, num_steps, noise=…)`
takes an explicit `noise`; passing a **fixed** noise (zeros = ODE from the origin,
or a fixed seed) makes `b(o)` a deterministic function reused verbatim in both
places. No train/inference drift — this is what "mean/precomputed base" means here.

**Three components (all GPU-gated; build after the main pipeline validates):**
1. `precompute_base_action.py` — load pi0.5 SFT via `create_trained_policy`, run
   `sample_actions(o, noise=fixed)` over each anchor frame, store the **full
   (action_horizon, 14) open-loop chunk** as a `base_action` column. (Open-loop
   chunk, not re-chunked first-steps — matches what inference adds the residual to.)
2. Training — config `pi06_yam_vial_30fps_from_sft_recap_residual` sets
   `residual_base=True` (adds `SubtractBaseAction` before `AlohaInputs`, target =
   `action − base_action`) **and `use_delta_joint_actions=False`**: the residual is
   the correction to pi0.5's *raw* output, so the joints must NOT also be
   delta'd against the state (DeltaActions would subtract state from the residual).
   The residual is itself the small/relative quantity. **Re-run
   `compute_norm_stats`** on this config (residual statistics differ). Advantage
   tokens + per-arm loss mask carry over unchanged. The absolute baseline config
   is untouched (separate config; verified `inputs=[AlohaInputs, DeltaActions]` vs
   residual `[SubtractBaseAction, AlohaInputs]`).
3. Serving — a wrapper that runs pi0.5 with the **same fixed noise** for `b(o)` and
   the residual actor for `Δ(o)`, and returns `b(o) + Δ(o)` in raw action space.
   **Cost: 2× VLA forward passes per step** — the main latency caveat for YAM.

Pre-written (compile-verified, GPU-untested): `scripts/precompute_base_action.py`
(base via `create_trained_policy`, `sample_kwargs={noise: zeros}`, writes the raw
`base_action` chunk), `SubtractBaseAction` (transforms.py), `residual_base` flag +
the `_residual` config. Serving wrapper is the only remaining TODO (after the
absolute baseline validates).

## 8. Status / TODO

- **Done + verified:** rule-based labeling (both scripts); loss-mask wiring
  (4 files, math + zero-gradient verified); two-adv-token wiring (tokenizer +
  transforms, string output verified); training config repack.
- **Smoke test pending:** `compute_norm_stats` + a few train steps — needs the
  labeled dataset (run §7 steps 0–1 first).
- **Serving config:** `..._infer` still maps the single `adv_ind`. For two-token
  serving the client must send `adv_ind_left = adv_ind_right = "positive"` (or
  inject them); update its repack when wiring serving.
- **Stage 4 value model = the rule-3 residual labeler (confirmed approach).**
  Routing across the 93-ep dagger set (episode-level proxy; real split is
  per-arm via summary mode): rule 1 = 9 eps / 7.5k frames, rule 2 = 52 eps /
  71.4k frames, **rule 3 = 32 eps / 26.7k frames (~25%, e.g. episode_000002)**.
  Per-arm success shrinks rule 3 further (a succeeding arm in a "failure"
  episode becomes rule-1 positive). Recipe:
  1. Train a per-arm value/classifier (SigLIP+Gemma backbone, à la
     `train_value.py`) on the **rule-1/2** frames, supervised by
     `adv_ind_{left,right} ∈ {positive, negative}` — exclude `none` and
     `loss_mask=0` (frozen) frames.
  2. Run it on the **rule-3** frames — defined precisely as `adv_ind == none`
     **and** `loss_mask == 1` (this excludes the frozen-arm `none` frames, which
     the policy loss masks anyway) — and write back `positive`/`negative`.

  So `none`-with-`loss_mask=1` is the exact set the VLM fills; everything else is
  hand-labeled. **Pre-written** as `prep_residual_training.py` (targets+indices,
  pandas-tested) → `train_value.py --arm` (masked two-hot; `none` ⇒ zero
  loss/grad) → `label_residual_from_vlm.py --arm` (V>−0.5 ⇒ positive). Ready to
  run once the labels exist; GPU-untested.
