# IKIM Assessment — Modality-Expert Medical AI System

A multimodal medical AI system using [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) with per-modality LoRA adapters for chest X-ray (CXR) and ECG analysis.

A single base model handles routing; lightweight LoRA adapters specialise it per modality. This keeps the memory footprint to one model load while allowing each modality to be trained and swapped independently.

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
- **Adapters:** One LoRA adapter per modality, each trained in two stages (pretraining → SFT).
- **Routing:** Zero-shot prompt on the base model (adapter disabled). No router training required.
- **Adapter switching:** PEFT multi-adapter — both adapters are registered on one model load and selected per input at zero cost. No reloading between modalities.
- **Hardware:** CPU-only (Intel i7). No GPU required — a deliberate scoping decision for the assessment.

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
  Data preparation
    download_redivis.py      download CheXpert Plus images + labels from Redivis
    join_chexpert_labels.py  merge raw CSV with binary-label JSON → df_enriched.csv
    eda_cxr.py               CheXpert Plus exploratory analysis
    eda_ecg.py               PTB-XL exploratory analysis + ECG-to-image rendering
    sample_datasets.py       stratified train/val/test splits per modality
    make_jsonl.py            convert sampled splits into ms-swift JSONL format
  Training
    verify_swift.py          smoke test: load model in fp32, run a forward pass
    train.py                 thin wrapper around `swift sft <config>`
    overnight.py             chain all 4 training stages with auto epoch extension
  Inference & evaluation
    inference.py             end-to-end pipeline: route, then answer with adapter
    evaluate.py              zero-shot baseline + assembled metrics table
    generate_examples.py     qualitative examples over a mixed-modality stream
```

## Setup

```bash
bash setup.sh
```

This installs [uv](https://docs.astral.sh/uv/) (if absent), runs `uv sync` to install all dependencies, and runs a smoke test (`verify_swift.py`) that loads Qwen3-VL-2B-Instruct in fp32 and runs a forward pass.

**Requirements:** Python 3.10+, CPU-only PyTorch (pinned to `torch <2.7` — see [docs/engineering_decisions.md](docs/engineering_decisions.md) for why), and ~12 GB free RAM (the fp32 2B weights are ~8.5 GB; activations and processing add overhead). No GPU needed.

All commands below assume `uv run` to use the synced environment.

### Model cache

Training configs point at a **local** HuggingFace snapshot path rather than a model ID, to force offline loading and avoid ms-swift's default ModelScope download (see [docs/engineering_decisions.md](docs/engineering_decisions.md)). On a fresh machine:

1. Run `uv run python scripts/verify_swift.py` once — this populates the local HF cache via `snapshot_download`.
2. Update the `model:` field in the four `configs/*.yaml` files to match your own snapshot hash.

## Data

See [docs/data_catalog.md](docs/data_catalog.md) for full details. Primary datasets:

| Dataset | Modality | Access | Role |
|---------|----------|--------|------|
| PTB-XL | ECG | Open (PhysioNet) | ECG training + eval |
| CheXpert Plus | CXR | Registration (Stanford AIMI) | CXR training + eval |

Datasets with registration/credentialing requirements are **not** included in this repo. Download instructions are in the data catalog; CheXpert images are fetched via `scripts/download_redivis.py` (requires `REDIVIS_API_TOKEN` in `.env`).

### Preparing data

```bash
uv run python scripts/join_chexpert_labels.py   # merge labels → df_enriched.csv
uv run python scripts/sample_datasets.py        # stratified train/val/test splits
uv run python scripts/make_jsonl.py             # → ms-swift JSONL (pretrain + SFT)
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
uv run python scripts/overnight.py
```

## Inference

Run the full routing pipeline on a single image — the base model detects the modality, then the matching specialist adapter answers:

```bash
uv run python scripts/inference.py --image path/to/image.png
uv run python scripts/inference.py --image path/to/image.png --query "Is there a pleural effusion?"
uv run python scripts/inference.py --demo     # runs one CXR + one ECG example
```

Adapter checkpoint paths are registered in `ADAPTERS` at the top of [scripts/inference.py](scripts/inference.py).

## Evaluation & Examples

```bash
uv run python scripts/evaluate.py            # zero-shot baseline + metrics table
uv run python scripts/generate_examples.py   # qualitative mixed-modality examples
```

- `evaluate.py` runs the base model (no adapter) on the validation sets to produce a zero-shot baseline, then assembles a zero-shot → pretrain → SFT comparison from the training logs. Writes `outputs/results/metrics.{json,md}`.
- `generate_examples.py` feeds a shuffled mix of CXR and ECG test examples through the routing pipeline and records, for each, the router's decision and the specialist's answer. Writes `outputs/results/examples.md`.

Both load the model once and never hold two model copies in memory; run them sequentially.

## Extending to New Modalities

The router is zero-shot, so adding a modality requires **no router retraining**:

1. Add a JSONL conversion function in `scripts/make_jsonl.py` for the new dataset.
2. Add a pretrain + SFT config pair in `configs/`.
3. Train the two stages (`swift sft <config>`).
4. Register the new adapter path in `ADAPTERS` in `scripts/inference.py`.

## Further Documentation

- [docs/engineering_decisions.md](docs/engineering_decisions.md) — dependency pins, ms-swift 4.x gotchas, and parameter rationale.
- [docs/data_catalog.md](docs/data_catalog.md) — dataset sources, access requirements, and licensing.
- [docs/literature.md](docs/literature.md) — background reading.

Results, evaluation analysis, and limitations are documented separately (see `docs/`).
