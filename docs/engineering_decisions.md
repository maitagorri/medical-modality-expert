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

## Training time estimates (CPU, Intel i7, 8 threads)

| Stage | Samples | Steps (÷4 grad_accum) | Time/step | Total |
|-------|---------|----------------------|-----------|-------|
| CXR pretrain | 74 × 2 epochs | 38 | ~280s | ~2.9 hrs |
| ECG pretrain | 75 × 2 epochs | 38 | ~240s | ~2.5 hrs |
| CXR SFT | 154 × 2 epochs | 78 | ~280s | ~6.1 hrs |
| ECG SFT | 75 × 2 epochs | 38 | ~240s | ~2.5 hrs |
| **Total** | | | | **~14 hrs** |

Times are estimates based on the observed 280s/step for the CXR pretrain run with `max_pixels=200704`, `max_length=512`, `gradient_checkpointing=True`.
