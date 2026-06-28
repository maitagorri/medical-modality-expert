# Engineering Decisions

This document records non-obvious parameter choices and the reasoning behind them, so the rationale doesn't live only in conversation history.

---

## Dependency stack

### torch 2.6.0+cpu + torchvision 0.21.0+cpu

**Why 2.6, not 2.12?**
The pytorch-cpu wheel index only provides `torchvision` wheels up to Python 3.11 for versions beyond 0.21.0. On Python 3.13, the only `torchvision+cpu` wheel with a `cp313` ABI tag is `0.21.0+cpu`, which corresponds to `torch 2.6.0`. Using PyPI's `torchvision 0.27.1` (built against torch 2.7) with pytorch-cpu's `torch 2.12.1+cpu` causes an ABI mismatch: `operator torchvision::nms does not exist` at import time. This crashes `transformers.image_utils` (which unconditionally imports torchvision in transformers 5.x), which cascades into `peft` failing to import `BloomPreTrainedModel`, which breaks ms-swift entirely.

**Constraint added to pyproject.toml:** `torch>=2.3,<2.7`

---

## Model loading

### Local HF snapshot path in training configs instead of model ID

**Why a hardcoded path?**
ms-swift 4.x defaults to downloading models from ModelScope (China CDN), even with `USE_MODELSCOPE_HUB=False` set in Python — the environment variable is read too late when ms-swift is launched as a subprocess. Passing the model's local HuggingFace cache path directly (`~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/<hash>`) forces all components (ms-swift, transformers, PEFT) to read from disk without any network call. This makes training launch in ~8s instead of waiting hours for a 3.96 GB download.

**Trade-off:** The path is machine-specific. Anyone cloning the repo on a different machine needs to run `uv run python scripts/verify_swift.py` first (which populates the local HF cache via `snapshot_download`), then update the `model:` field in the YAML configs to match their own snapshot hash.

---

## Image processing

### `max_pixels: 200704`

**Why this value?**
CXR images in CheXpert Plus are typically 2000–3000 px wide. Passed to Qwen3-VL without resizing, a 2828×2320 image produces ~6400 tokens just for the visual tokens — far beyond any feasible `max_length` for CPU training. `max_pixels = 200704` is `256 × 28 × 28`: it limits the image to the equivalent of 256 vision patches before downsampling, producing ~230–260 image tokens per example. This keeps the total sequence within 512 tokens (image + prompt + report text).

**How it works:** Qwen3-VL's processor dynamically resizes images to fit within the pixel budget while preserving aspect ratio. The resized image is then split into 28×28 patches and encoded by the visual encoder.

### `max_length: 512`

**Why not 256?**
The original plan specified `max_length=256` assuming only the text portion mattered. In a VLM, `max_length` applies to the *total* tokenized sequence: image tokens + system prompt + user prompt + assistant response. With `max_pixels=200704`, image tokens alone average ~250, and the radiology report text adds another 50–200 tokens. 62 of 74 training examples exceed 256 tokens. ms-swift raises a hard error (not silent truncation) when examples exceed `max_length` and it can't find a valid example after `n_try_fetch` retries. Setting `max_length=512` accommodates all 74 training examples (max observed: 460 tokens).

**RAM impact:** Attention is O(n²) in sequence length. Going from 256 to 512 doubles the effective sequence length but `gradient_checkpointing=True` means activations are recomputed on backward, keeping peak RAM roughly proportional to sequence length rather than quadratic.

---

## ms-swift configuration

### `gradient_checkpointing: true`

Required for CPU training. Without it, PyTorch stores all forward-pass activations in RAM for the backward pass. With a 2.1B parameter model in fp32 (~8.5 GB), there is insufficient RAM for both weights and activations. Gradient checkpointing recomputes activations during the backward pass at the cost of ~33% more compute, trading RAM for time.

### `dataloader_num_workers: 0`

Each worker subprocess spawned by the DataLoader loads a fresh copy of the model into memory to be safe across fork boundaries. On a 24 GB machine with 8.5 GB already in use by the model, even one additional worker would cause OOM. `num_workers=0` runs data loading in the main process.

### `tuner_type: lora` / `target_modules: all-linear` (not `train_type` / `lora_target_modules`)

ms-swift 4.x renamed these fields. The original plan used the 3.x names (`train_type`, `lora_target_modules`). The 4.x names are `tuner_type` and `target_modules`. Passing the old names results in `remaining_argv` errors at startup.

### Config loaded as positional argument, not `--config`

In ms-swift 4.x, `swift sft --config file.yaml` does **not** load the YAML — `--config` is an unrecognized flag that gets silently ignored, leaving all parameters at their defaults (including `model: None`, which immediately errors). The correct invocation is `swift sft file.yaml` (YAML path as the first positional argument). The YAML loading is handled in `swift/cli/main.py:parse_yaml_args()`, which checks `argv[0]` for a `.yaml`/`.yml` suffix.

### `report_to: none`

`wandb` is listed as a dependency but logging to Weights & Biases requires an API key and active account. Disabling avoids import errors and credential prompts during training. Re-enable by changing to `report_to: wandb` and running `wandb login` after setting `WANDB_API_KEY` in `.env`.

---

## CXR pretrain stopping criterion

Stopped at 4 epochs (76 steps, resumed from the 2-epoch checkpoint) with eval loss still decreasing:

| Checkpoint | Eval loss |
|---|---|
| Epoch 2 (v5 final) | 1.999 |
| Epoch ~3 (v8 step 50) | 1.940 |
| Epoch 4 (v8 final) | 1.898 |

**Normal practice** would be to keep training until eval loss turns upward (overfitting onset), then use the checkpoint just before the turn. No overfitting was observed here — loss was still improving at the point training was stopped.

**Why stopped early:** Time constraint. The model was still learning at epoch 4 on 75 training samples, so the final checkpoint slightly underperforms what a full run would achieve. For an assessment this is acceptable; in production you would run to convergence.

---

## SFT checkpointing: always match eval_steps to save_steps

During CXR SFT v0, `eval_steps: 25` and `save_steps: 10` caused the best checkpoint (step 25, eval_loss 0.070, token_acc 98.4%) to be deleted before the run ended — `save_total_limit: 3` pruned it while training continued past step 30. The checkpoint only existed between evaluations, and was deleted on the next save cycle.

**Fix applied to all subsequent SFT configs:**
- `eval_steps: 10` (matches `save_steps: 10`) — every saved checkpoint gets an eval score
- `save_total_limit: 5` — wider window so early-peak checkpoints survive
- `load_best_model_at_end: true` + `metric_for_best_model: eval_loss` — ms-swift copies the best checkpoint to `best_model_checkpoint` at the end of training

**CXR SFT peak behaviour:** best checkpoint appears very early (~step 20, epoch 0.5) then eval loss climbs before partially recovering. Both v0 and v1 runs showed this pattern. Likely cause: small validation set (42 examples) + fast convergence from the pretrained initialisation. The SFT task (binary yes/no) is much simpler than pretrain, so the model reaches near-optimal in under one epoch.

---

## Inference stack: transformers + PEFT directly (not ms-swift TransformersEngine)

ms-swift's `TransformersEngine` raises `ValueError: Mixed using with peft is not allowed now` when a second engine instance is created in the same process, even after the first is deleted and `gc.collect()` is called. Root cause: PEFT leaves global state in `peft.peft_model` that ms-swift's `SwiftModel.from_pretrained` checks.

**Fix:** All inference scripts (`evaluate.py`, `generate_examples.py`, `inference.py`) now use `transformers.Qwen3VLForConditionalGeneration.from_pretrained` + `peft.PeftModel.from_pretrained` directly. This avoids ms-swift's adapter loading path entirely.

**Adapter hot-swapping in inference.py:** PEFT's multi-adapter API (`model.load_adapter()` + `model.set_adapter()`) allows switching between the CXR and ECG adapters at zero cost after a single model load. Zero-shot routing uses `model.disable_adapter()` context. No second model load is ever needed.

---

## Evaluation sampling: stratified 10-per-label for CXR

CXR test set has 420 examples across 14 binary pathology labels (~30 per label). Running the full set × 3 passes (zero-shot, pretrained, SFT) would take ~9h. A random subsample of N examples risks under-representing rare labels.

**Decision:** Stratified sample of 10 examples per label (140 total, seed=42). Implemented via `--max-per-label 10` flag in `evaluate.py`. ECG test set (50 examples, 5 classes, 10 per class already) is used in full.

**Trade-off:** AUC estimates at 10 samples per class are noisy; accuracy and F1 trends are reliable enough to show the zero-shot → pretrained → SFT improvement and are defensible for an assessment submission.

---

## Training time estimates (CPU, Intel i7, 8 threads)

| Stage | Samples | Steps (÷4 grad_accum) | Time/step | Total |
|-------|---------|----------------------|-----------|-------|
| CXR pretrain | 74 × 2 epochs | 38 | ~280s | ~2.9 hrs |
| ECG pretrain | 75 × 2 epochs | 38 | ~240s | ~2.5 hrs |
| CXR SFT | 154 × 2 epochs | 78 | ~280s | ~6.1 hrs |
| ECG SFT | 75 × 2 epochs | 38 | ~240s | ~2.5 hrs |
| **Total** | | | | **~14 hrs** |

Times are estimates based on the observed 280s/step for the CXR pretrain run with `max_pixels=200704`, `max_length=512`, `gradient_checkpointing=True`.
