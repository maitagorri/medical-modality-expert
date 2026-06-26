# IKIM Assessment — Modality-Expert Medical AI System

A multimodal medical AI system using [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) with per-modality LoRA adapters for chest X-ray (CXR) and ECG analysis.

## Architecture

```
Input image
     │
     ▼
Qwen3-VL-2B-Instruct (base, no adapter)
 Zero-shot modality routing prompt
     │
     ├── "xray"  ──► load cxr_sft adapter  ──► CXR specialist
     └── "ecg"   ──► load ecg_sft adapter  ──► ECG specialist
```

**Base model:** Qwen3-VL-2B-Instruct (router + adapter backbone)  
**Adapters:** One LoRA adapter per modality, trained in two stages (pretraining → SFT)  
**Hardware:** CPU-only (Intel i7). No GPU required — a deliberate scoping decision.

## Repository Structure

```
configs/          ms-swift training configs (pretrain + SFT for CXR and ECG)
data/
  raw/            source datasets (gitignored)
  processed/      sampled splits and JSONL files (gitignored)
docs/             data catalog, literature review
notebooks/figures EDA output plots
outputs/          checkpoints, logs, eval results (gitignored)
scripts/
  join_chexpert_labels.py  merge raw CSV with binary label JSON → df_enriched.csv
  eda_ecg.py               PTB-XL exploratory analysis + ECG-to-image validation
  eda_cxr.py               CheXpert Plus exploratory analysis
  sample_datasets.py       stratified train/val/test splits (75/25 per modality)
  make_jsonl.py            convert sampled splits to ms-swift JSONL format
  verify_swift.py          smoke test: load model, run forward pass
  train.py                 thin wrapper around swift sft
  evaluate.py              baseline vs. fine-tuned evaluation (accuracy, F1, AUC)
  inference.py             full end-to-end pipeline with router + adapter loading
```

## Setup

```bash
bash setup.sh
```

Installs all requirements and runs a smoke test (loads Qwen3-VL-2B-Instruct, prints parameter count).

**Requirements:** Python 3.10+, ~8 GB RAM (fp32 2B model), no GPU needed.

## Data

See [docs/data_catalog.md](docs/data_catalog.md) for full details. Primary datasets:

| Dataset | Modality | Access | Role |
|---------|----------|--------|------|
| PTB-XL | ECG | Open (PhysioNet) | ECG training |
| CheXpert Plus | CXR | Registration (Stanford AIMI) | CXR training |

Datasets with registration/credentialing requirements are not included in this repo. Download instructions are in the data catalog.

## Training

Training uses [ms-swift](https://github.com/modelscope/ms-swift) on CPU. Two stages per modality:

**Stage 1 — Pre-training (report/description generation):**
```bash
swift sft --config configs/pretrain_cxr.yaml   # CXR
swift sft --config configs/pretrain_ecg.yaml   # ECG
```

**Stage 2 — Supervised fine-tuning (VQA with labels):**
```bash
swift sft --config configs/sft_cxr.yaml        # CXR (14-label yes/no)
swift sft --config configs/sft_ecg.yaml        # ECG (5 superclass labels)
```

Settings: `lora_rank=8`, `batch_size=1`, `gradient_accumulation=4`, `max_length=256`, `fp32`.  
Expect ~hours per epoch on CPU. Configs use `save_steps=10` to checkpoint frequently.

## Evaluation

```bash
python scripts/evaluate.py \
  --adapter outputs/cxr_sft \
  --test-jsonl data/processed/chexpert_plus/test_sft.jsonl \
  --modality cxr

python scripts/evaluate.py \
  --adapter outputs/ecg_sft \
  --test-jsonl data/processed/ptbxl/test.jsonl \
  --modality ecg
```

Outputs accuracy, macro F1, and per-class AUC to `outputs/results/`.

## Results

_(populated after training — see `outputs/results/`)_

## Known Limitations

- Dataset scale scoped for local CPU training (~75 train examples per modality)
- Two modalities trained (CXR, ECG); remaining 7 documented as roadmap only
- Evaluation on small test sets (50–234 examples); results are indicative, not conclusive

## Roadmap

To extend to additional modalities: add data, write a new JSONL conversion function in `make_jsonl.py`, add a config pair in `configs/`, and register the adapter path in `inference.py`. The router is zero-shot and requires no retraining.

To scale: replace fp32 LoRA with QLoRA (`quantization_bit: 4`) on a GPU, increase dataset size, and raise `max_length` to 512+.

**Fallback model:** Qwen2.5-VL-2B-Instruct is a tested alternative if Qwen3-VL-2B-Instruct is unavailable.
