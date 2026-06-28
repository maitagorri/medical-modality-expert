# IKIM Assessment — Modality-Expert Medical AI System

## Project Overview

A multimodal medical AI system that fine-tunes [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) into per-modality experts for chest X-ray (CXR) and ECG analysis, using one shared base model with lightweight LoRA adapters. A zero-shot router (the base model itself) detects the input modality and activates the matching specialist adapter, so the whole system runs from a single model load on CPU.

## Architecture

```
Input image
     │
     ▼
Qwen3-VL-2B-Instruct (base, no adapter)
 Zero-shot modality routing prompt
     │
     ├── "xray"  ──► activate cxr adapter  ──► CXR specialist answers
     └── "ecg"   ──► activate ecg adapter  ──► ECG specialist answers
```

- **Base model:** Qwen3-VL-2B-Instruct — serves as both the zero-shot router and the LoRA backbone.
- **Router:** A zero-shot prompt on the base model (adapter disabled) classifies the modality. No router training required.
- **Specialist adapters:** One LoRA adapter per modality, each trained in two stages — **pretraining** (report / description generation) then **supervised fine-tuning** (label VQA).
- **Adapter switching:** PEFT multi-adapter — both adapters are registered on one model load and selected per input at zero cost, with no reloading between modalities.
- **Hardware:** CPU-only, tested on an Intel i7. No GPU required — a deliberate scoping decision for this assessment.

## Repository Structure

```
configs/                     ms-swift training configs (pretrain + SFT × CXR, ECG)
data/
  raw/                       source datasets (gitignored)
  processed/                 sampled splits + JSONL files (gitignored)
docs/
  data_catalog.md            dataset sources, access, and roles
  engineering_decisions.md   non-obvious parameter/dependency rationale
  literature.md              background reading
notebooks/figures/           EDA output plots (CXR + ECG examples)
outputs/                     checkpoints, training logs, results (gitignored)
scripts/
  data/
    download_redivis.py      download CheXpert Plus images + labels from Redivis
    join_chexpert_labels.py  merge raw CSV with binary-label JSON → df_enriched.csv
    eda_cxr.py               CheXpert Plus exploratory analysis
    eda_ecg.py               PTB-XL exploratory analysis + ECG-to-image rendering
    sample_datasets.py       stratified train/val/test splits per modality
    make_jsonl.py            convert sampled splits into ms-swift JSONL format
  train/
    verify_swift.py          smoke test: load model in fp32, run a forward pass
    train.py                 thin wrapper around `swift sft <config>`
    overnight.py             chain all 4 training stages with auto epoch extension
  eval/
    inference.py             end-to-end pipeline: route, then answer with adapter
    evaluate.py              zero-shot baseline + assembled metrics table
    generate_examples.py     qualitative examples over a mixed-modality stream
```

## Setup Instructions

```bash
bash setup.sh
```

This installs [uv](https://docs.astral.sh/uv/) (if absent), runs `uv sync` to install all dependencies, and runs a smoke test (`verify_swift.py`) that loads Qwen3-VL-2B-Instruct in fp32 and runs a forward pass.

**Requirements:** Python 3.10+, CPU-only PyTorch (pinned to `torch <2.7` — see [docs/engineering_decisions.md](docs/engineering_decisions.md) for why), and ~12 GB free RAM (the fp32 2B weights are ~8.5 GB; activations and image processing add overhead). **No GPU required** — all training and inference is designed for and tested on CPU (Intel i7).

All commands below assume `uv run` to use the synced environment.

### Model cache

Training configs point at a **local** HuggingFace snapshot path rather than a model ID, to force offline loading and avoid ms-swift's default ModelScope download (see [docs/engineering_decisions.md](docs/engineering_decisions.md)). On a fresh machine:

1. Run `uv run python scripts/train/verify_swift.py` once — this populates the local HF cache via `snapshot_download`.
2. Update the `model:` field in the four `configs/*.yaml` files to match your own snapshot hash.

## Data

See [docs/data_catalog.md](docs/data_catalog.md) for full details. Primary datasets:

| Dataset | Modality | Access | Role |
|---------|----------|--------|------|
| PTB-XL | ECG | **Open** (PhysioNet) | ECG training + eval |
| CheXpert Plus | CXR | **Credentialed** — registration required (Stanford AIMI) | CXR training + eval |

**CheXpert Plus requires credentialing** (Stanford AIMI registration) and is **not** included in this repo. Images are fetched via `scripts/download_redivis.py` (requires `REDIVIS_API_TOKEN` in `.env`). PTB-XL is openly available from PhysioNet.

### Preparing data

```bash
uv run python scripts/data/join_chexpert_labels.py   # merge labels → df_enriched.csv
uv run python scripts/data/sample_datasets.py        # stratified train/val/test splits
uv run python scripts/data/make_jsonl.py             # → ms-swift JSONL (pretrain + SFT)
```

This produces:

```
data/processed/chexpert_plus/{train,val,test}_pretrain.jsonl   report generation
data/processed/chexpert_plus/{train,val,test}_sft.jsonl        binary-label VQA
data/processed/ptbxl/{train,val,test}.jsonl                    ECG classification VQA
```

## Training

Training uses [ms-swift](https://github.com/modelscope/ms-swift) on CPU, in two stages per modality. The config path is passed as a **positional** argument — `swift sft <config>`, **not** `--config <config>` (which ms-swift 4.x silently ignores; see [docs/engineering_decisions.md](docs/engineering_decisions.md)).

**Stage 1 — Pretraining** (report / description generation):
```bash
swift sft configs/pretrain_cxr.yaml
swift sft configs/pretrain_ecg.yaml
```

**Stage 2 — Supervised fine-tuning** (label VQA; initialised from the Stage 1 adapter):
```bash
swift sft configs/sft_cxr.yaml     # CXR — 14-label yes/no
swift sft configs/sft_ecg.yaml     # ECG — 5 superclass labels
```

Key settings (full rationale in [docs/engineering_decisions.md](docs/engineering_decisions.md)): `lora_rank=8`, `batch_size=1`, `gradient_accumulation=4`, `max_length=512`, `max_pixels=200704`, `gradient_checkpointing=true`, `fp32`. Expect hours per epoch on CPU; configs checkpoint every 10 steps.

To chain all four stages unattended (with automatic epoch extension while val loss is still improving):
```bash
uv run python scripts/train/overnight.py
```

## Inference

Run the full routing pipeline on a single image — the base model detects the modality, then the matching specialist adapter answers:

```bash
uv run python scripts/eval/inference.py --image path/to/image.png
uv run python scripts/eval/inference.py --image path/to/image.png --query "Is there a pleural effusion?"
uv run python scripts/eval/inference.py --demo     # runs one CXR + one ECG example
```

Adapter checkpoint paths are registered in `ADAPTERS` at the top of [scripts/eval/inference.py](scripts/eval/inference.py).

## Evaluation

```bash
uv run python scripts/eval/evaluate.py            # zero-shot baseline + metrics table
uv run python scripts/eval/generate_examples.py   # qualitative mixed-modality examples
```

- **`evaluate.py`** runs the base model (no adapter) on the validation sets to produce a **zero-shot baseline**, then assembles a zero-shot → pretrain → SFT comparison by reading the best checkpoint metrics from each stage's training log. Writes `outputs/results/metrics.{json,md}`.
- **`generate_examples.py`** feeds a shuffled mix of CXR and ECG test examples through the routing pipeline and records, for each, the router's decision and the specialist's answer (with a ✓/✗ against ground truth). Writes `outputs/results/examples.md`.

Both load the model once and never hold two model copies in memory; run them sequentially.

**Interpreting the metrics:** the primary metric is **token accuracy** on held-out validation examples — the fraction of examples where the model's answer matches the ground-truth label. The zero-shot → pretrain → SFT progression shows how much each training stage contributes over the untuned base model.

## Results

Token accuracy on the held-out **validation** set (CXR: 42 examples from CheXpert Plus; ECG: 25 examples from PTB-XL). Pretrain and SFT figures are the best checkpoint from each stage's training log; the zero-shot row is produced by `evaluate.py`.

| Modality | Stage | Token Accuracy | Eval Loss |
|----------|-------|----------------|-----------|
| CXR | Zero-shot (base model) | 21.4% | — |
| CXR | Pretrain | 59.1% | 1.898 |
| CXR | **SFT** | **97.6%** | **0.061** |
| ECG | Zero-shot (base model) | 20.0% | — |
| ECG | Pretrain | 77.8% | 0.459 |
| ECG | **SFT** | **80.0%** | **0.462** |

Supervised fine-tuning lifts CXR from 59% → 98% token accuracy. For ECG, most of the signal is already captured in pretraining (78%), with SFT adding a smaller gain to 80%.

Qualitative end-to-end examples (router decision + specialist answer over a mixed CXR/ECG stream) are written to [outputs/results/examples.md](outputs/results/examples.md) by `generate_examples.py`.

> **Note:** the zero-shot baseline row and the qualitative examples file are regenerated by the two evaluation scripts; full numeric artifacts live in `outputs/results/`.

## Known Limitations

- **Dataset scale.** Splits are deliberately small (~75 train / 25 val per modality) so the full pipeline runs end-to-end on a local CPU. Results are indicative of the approach working, not statistically conclusive.
- **Two modalities.** Only CXR and ECG were implemented; the remaining modalities are documented as a roadmap, not built.
- **Validation-set metrics, not a full test sweep.** A larger held-out test-set evaluation (CheXpert `valid` split, PTB-XL test split) was scripted but not completed — repeated out-of-memory kills on the 24 GB CPU machine (the fp32 model alone is ~11 GB resident) made the multi-pass test run unreliable. Reported numbers come from the validation evaluations logged during training.
- **A best-performing CXR SFT checkpoint was pruned.** An early checkpoint scored higher on token accuracy but was removed by `save_total_limit` before it could be preserved; the reported SFT adapter is the best surviving checkpoint.
- **CPU / fp32 constraints.** No quantization (bitsandbytes has no CPU backend) and `max_length=512`, which bounds the report/context length the model sees during training.

## Roadmap

- **More modalities.** The router is zero-shot, so adding a modality needs **no router retraining**: add a JSONL conversion in `scripts/data/make_jsonl.py`, add a pretrain + SFT config pair in `configs/`, train the two stages, and register the new adapter path in `scripts/eval/inference.py`. The same recipe extends to the remaining target modalities.
- **Scale with GPU.** Move from fp32 LoRA to QLoRA (`quantization_bit: 4`) on a GPU, increase dataset size, and raise `max_length` — removing the CPU/RAM constraints that currently cap scale and context length.

## Further Documentation

- [docs/engineering_decisions.md](docs/engineering_decisions.md) — dependency pins, ms-swift 4.x gotchas, and parameter rationale.
- [docs/data_catalog.md](docs/data_catalog.md) — dataset sources, access requirements, and licensing.
- [docs/literature.md](docs/literature.md) — background reading.
