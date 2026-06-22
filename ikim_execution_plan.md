# IKIM Assessment — Execution Plan

## Committed Decisions

Before the plan begins, these choices are locked:

- **Model:** Qwen3-VL-2B-Instruct (LoRA / QLoRA). ms-swift support confirmed. Qwen2.5-VL-2B as fallback if anything breaks.
- **Primary modalities:** CXR (Röntgen), ECG, Text. Remaining 7 modalities documented as a roadmap, not trained.
- **ECG representation:** Waveform plotted as a PNG image, fed into the VL model like any other image.
- **Adapter architecture:** One LoRA adapter per modality (CXR adapter, ECG adapter), trained separately.
- **Router:** No third training run. The base Qwen2.5-VL-2B, prompted appropriately, identifies the modality and Python code loads the correct adapter.
- **Hardware strategy:** Training on Colab (free T4) or Kaggle (free P100). Local machine used for everything else — environment, data prep, evaluation, inference. This is stated transparently in the submission.
- **Dataset size:** ~300 training / 50 validation / 50 test examples per modality. Enough to show meaningful fine-tuning on CPU-adjacent hardware; honest about scale.

---

## How to Use Claude Throughout

**This project chat (Claude Opus 4.6):** Planning, methodology review, literature interpretation, writing the final Word summary. Use it when you're making a decision or need something written well in German.

**Claude Code (Claude Sonnet 4.6):** Everything that touches files or runs in a terminal — setup scripts, preprocessing pipelines, training configs, evaluation scripts, the README. Claude Code can run your scripts and fix errors iteratively, which is far more efficient than copy-pasting.

A practical pattern: when you hit a problem in Claude Code, paste the error and the surrounding context. Don't just paste the error alone — Claude Code needs to see the data shape, the config, and what you were trying to do.

---

## Week 1: Research & Data Acquisition

### Day 1 — Registrations and skeleton

**Registrations (do these in parallel, they take time):**

- PhysioNet credentialed access at physionet.org. Required for MIMIC-CXR and MIMIC-IV-ECG. Takes 2–5 days. The training module is straightforward but mandatory. Start this today.
- CheXpert at stanfordmlgroup.github.io/competitions/chexpert. Registration is fast, usually same-day.
- EchoNet-Dynamic at echonet.github.io/dynamic. Stanford, fast.

**Do this yourself — dataset verification:**

While registrations are processing, navigate to each repository manually: PhysioNet, HuggingFace (ReXGradient-160K), the Kaggle pages for NIH ChestX-ray14 and OpenI, and Grand Challenge. For each dataset you plan to use or document, record: the exact license, the data use agreement terms if any, the actual file sizes and formats, and any preprocessing notes in the README. This is what makes the data catalog credible — it reads like someone who actually opened the pages, not a summary of summaries. Budget 1–2 hours. The catalog template Claude Code generates is just a skeleton; the content has to come from you.

**Project skeleton — Claude Code prompt:**

> Create a Python project skeleton for a medical AI fine-tuning project. Directory structure: /data (with subdirs /raw and /processed), /scripts (preprocessing, training, eval, inference), /configs (ms-swift YAML files), /outputs (checkpoints, logs, results), /notebooks, /docs. Generate a requirements.txt including: torch, transformers>=4.57, peft, ms-swift, wandb, matplotlib, numpy, pandas, scikit-learn, wfdb, Pillow, qwen_vl_utils>=0.0.14, decord. Also create a .gitignore that excludes /data/raw, /outputs/checkpoints, and .env files. Write a setup.sh that installs requirements and runs a smoke test: loads Qwen3-VL-2B-Instruct in 4-bit quantization and prints the model parameter count.

Commit the skeleton to a new GitHub repo immediately. An early commit history looks better than a single dump at the end.

---

### Days 2–3 — Open datasets and EDA

PTB-XL is fully open and available immediately. Download it from PhysioNet without credentialing (it's a separate, open collection). It's 1.7 GB and contains 21,837 12-lead ECG recordings with diagnostic labels.

**Claude Code prompt for PTB-XL EDA:**

> Write a Python script scripts/eda_ecg.py that: loads PTB-XL using the wfdb library, plots 3 example ECGs as 12-panel PNG images (one per diagnostic superclass: NORM, MI, STTC), prints class distribution for the superclass labels, and saves the plots to /notebooks/figures/. The output images should be clean enough to include in documentation.

This script does two things: it validates your ECG-to-image conversion pipeline (which is the same pipeline you'll use for training), and it gives you figures for the data catalog.

Also begin literature review in parallel. This is graded work — the task explicitly asks for publications, authors, models, and benchmarks, and the documentation needs to read like someone who actually did the research.

**Do this yourself on Google Scholar and PubMed:**

Search terms to use: "vision-language model radiology", "multimodal medical foundation model", "ECG classification deep learning benchmark", "chest X-ray report generation". For each paper: read the abstract, skim the Methods and Data sections, note what dataset they used and what benchmark they evaluated on. Repeat until the same names start recurring — that's when you know you've covered the core of the field.

The papers to prioritise (minimum: abstract + methods section):

- **LLaVA-Med** (Li et al., 2023) — vision-language model for medicine, directly relevant to your architecture. Note their training data and evaluation benchmarks.
- **BioViL / BioViL-T** (Bannur et al., Microsoft Research) — vision-language pretraining on MIMIC-CXR, shows what's achievable with the same data you're using.
- **CheXagent** — CXR foundation model; find who built it and what benchmarks they report.
- **ECG-FM** — ECG foundation model; note their dataset and whether PTB-XL appears in their evaluation.
- **PTB-XL paper** (Wagner et al., 2020) — primary source for your ECG dataset; read enough to understand the label schema.
- **QLoRA paper** (Dettmers et al., 2023) — understand what you're doing technically; you'll likely be asked about it.

Aim for about an hour of real reading. You don't need to understand every paper deeply — you need to be able to discuss them.

**Then use this chat to:**
- Clarify a methodology section that's unclear
- Cross-reference newer work citing these papers ("are there ECG foundation models newer than ECG-FM?")
- Fill in the Qwen-based medical models section — this is narrow enough that a targeted search here is faster than browsing, and Claude will handle it
- After your reading pass, come back and ask Claude to check for gaps — benchmarks you might have missed, or authors whose other work is worth following

---

### Days 3–5 — CXR data and preprocessing specs

CheXpert should arrive within 1–2 days of registration. It's 439 GB in full — **do not download the full dataset**. CheXpert-Small (11 GB, 224×224px downsampled) is sufficient and what most papers use. The label file (CSV) is a separate small download and gives you 14 binary labels per image.

**For the self-supervised stage, free-text reports are needed — here are the options in order of preference:**

**ReXGradient-160K** is the best MIMIC-CXR substitute and requires no credentialing. It's a 2025 dataset from the Rajpurkar lab (Harvard) containing 160,000 chest X-ray studies with paired radiology reports across 79 medical sites, openly available on HuggingFace at `rajpurkarlab/ReXGradient-160K`. Download the reports first (small), then sample images as needed. Download this on Day 3 regardless of whether MIMIC access comes through.

**OpenI / IU-Xray** (Indiana University) is a smaller fully open alternative: 7,470 chest X-rays with 3,955 free-text radiology reports, available from the National Library of Medicine or as a clean Kaggle mirror at `raddar/chest-xrays-indiana-university`. At ~300MB it's fast to download and useful as a sanity-check corpus even if you use ReXGradient for training.

**MIMIC-CXR** remains the gold standard if credentialing comes through — prioritize the reports (text files, small) over the images. But with ReXGradient available, MIMIC is now a nice-to-have rather than a dependency.

**Claude Code prompt for data sampling:**

> Write a script scripts/sample_datasets.py that: for CheXpert-Small, samples 400 examples stratified across the 14 label columns (300 train, 50 val, 50 test), saves the splits as CSV files in /data/processed/chexpert/; for PTB-XL, samples 400 examples stratified across the 5 superclass labels (same split), saves to /data/processed/ptbxl/. Ensure no patient appears in more than one split (use the patient_id column where available). Print a summary of class balance in each split.

---

### Days 5–7 — Documentation, roadmap, and preprocessing pipeline

**Data catalog — Claude Code prompt:**

> Create docs/data_catalog.md with a structured table for each dataset: name, modality, source URL, access method (open / registration / credentialed), size, number of examples, label type (none / structured / free text), license, and notes on known preprocessing requirements. Fill in the following datasets: PTB-XL (open, ECG), CheXpert-Small (registration, CXR structured labels), ReXGradient-160K (open, CXR + free-text reports, HuggingFace rajpurkarlab/ReXGradient-160K), OpenI/IU-Xray (open, CXR + free-text reports, ~7k images), NIH ChestX-ray14 (open, CXR structured labels, 112k images, Kaggle), MIMIC-CXR (credentialed, CXR + free-text reports), EchoNet-Dynamic (registration, Echo), MIMIC-IV-ECG (credentialed, ECG). Add a role column indicating whether each dataset is used for: self-supervised pretraining, supervised fine-tuning, both, or documented only. Add a second section listing the remaining modalities (CT, MRI, Ultrasound, Coronary Angiography, General Time Series, Tables) with the best available public dataset for each and a brief note on why it was deferred.

For the "further acquisition strategies" section of your submission, the outline already covers the key approaches. Use this chat to draft a crisp German-language version — it should be 3–4 short paragraphs covering: weak supervision from paired reports, synthetic data for sparse modalities (Koronarangiographie is the example), federated learning as the path to institutional data under DSGVO, and existing pretrained models as proxies when data is inaccessible.

**JSONL conversion pipeline — Claude Code prompt:**

> Write a script scripts/make_jsonl.py that converts sampled datasets into ms-swift conversation format JSONL. For CXR: each example should be {"messages": [{"role": "user", "content": [{"type": "image", "image": "<path>"}, {"type": "text", "text": "Does this chest X-ray show <finding>?"}]}, {"role": "assistant", "content": "Yes" or "No"}]} for each of the 14 CheXpert labels. For ECG: first convert the waveform to a PNG image using matplotlib (all 12 leads stacked, clean axis labels, no title), then use the same conversation format with the PTB-XL superclass as the answer. Save outputs to /data/processed/chexpert/train.jsonl etc. and /data/processed/ptbxl/train.jsonl etc. Print 2 sample entries from each dataset to stdout for verification.

The ECG-to-image conversion function in this script is important — it becomes the shared utility used in both training and inference. Write it once, write it well, test it.

---

## Week 2: Training and Evaluation

### Day 8 — Environment verification and cloud setup

Run the smoke test from Day 1. If it passes locally, also set it up on Colab or Kaggle. The workflow is: all data prep runs locally, push processed JSONL files and the scripts to GitHub, pull them in the cloud notebook, run training there, pull checkpoints back locally for evaluation and inference.

**ms-swift verification — Claude Code prompt:**

> Write a script scripts/verify_swift.py that: loads Qwen3-VL-2B-Instruct in 4-bit quantization using ms-swift, runs a forward pass on a test image and text prompt, prints the output, and prints peak memory usage. If this fails, also try loading Qwen2.5-VL-2B as a fallback and report which one works.

This is your go/no-go check before investing any training time.

---

### Days 8–9 — Self-supervised pre-training stage

This is what the task calls "unsupervised fine-tuning." Concretely: train the model to predict radiology report text given a CXR image, and to predict ECG diagnostic descriptions given an ECG image. No structured labels required — the free text is the supervision signal.

For CXR: use ReXGradient-160K or OpenI reports as the primary text source — sample 300 image+report pairs and format them as report completion tasks ("Given this chest X-ray, write the radiology report findings:"). If MIMIC-CXR access comes through, substitute or supplement with MIMIC reports, which are more clinically diverse. Do not fall back to synthetic prompts if any of these three datasets are available.
**ms-swift config — Claude Code prompt:**

> Generate a ms-swift training YAML config file configs/pretrain_cxr.yaml for continual pre-training (next-token prediction on the text, given image) of Qwen3-VL-2B-Instruct with LoRA. Settings: lora_rank=16, lora_alpha=32, target modules = all linear layers in the language model, learning_rate=2e-4, batch_size=1, gradient_accumulation_steps=8, num_epochs=2, fp16=True, dataset path = data/processed/chexpert/train.jsonl, output_dir=outputs/cxr_pretrain. Also generate a corresponding configs/pretrain_ecg.yaml for the ECG dataset.

Run the CXR pre-training first. Watch the loss for the first 20 steps to confirm it's decreasing before leaving it to run.

---

### Days 10–11 — Supervised fine-tuning stage

Starting from the pre-trained LoRA checkpoint (not from scratch), continue training on the labeled data — CheXpert binary labels for CXR, PTB-XL superclass labels for ECG. The conversation format is VQA: question about a specific finding, yes/no or class-name answer.

**Claude Code prompt:**

> Generate configs/sft_cxr.yaml for supervised fine-tuning, starting from the checkpoint at outputs/cxr_pretrain/. Same LoRA settings, but lower learning rate (5e-5), 3 epochs, dataset = data/processed/chexpert/train.jsonl (the labeled split). Also generate configs/sft_ecg.yaml. Both configs should log to Weights & Biases with project name "ikim-assessment".

The output of this stage is your two adapter files: outputs/cxr_sft/adapter_model.bin and outputs/ecg_sft/adapter_model.bin.

---

### Day 12 — Evaluation

**Claude Code prompt:**

> Write a script scripts/evaluate.py that: takes a model adapter path and a test JSONL file as arguments, loads Qwen3-VL-2B-Instruct in 4-bit + the specified LoRA adapter, runs inference on each test example, parses yes/no answers for CXR (14 labels) and class names for ECG (5 superclasses), and computes: accuracy, macro F1, and per-class AUC where applicable. Run it twice per modality: once with adapter=None (zero-shot baseline), once with the fine-tuned adapter. Save results as JSON to outputs/results/. Also generate a results summary table in Markdown.

This gives you the central result of the project: a before/after table showing what fine-tuning gained. Even a 5–10% improvement on a small test set is a real finding.

---

### Day 13 — Router and end-to-end demo

**Claude Code prompt:**

> Write a script scripts/inference.py that implements the full modality-expert system. It should: accept an image path and an optional text query as arguments, use Qwen3-VL-2B-Instruct (no adapter) to classify the modality with a zero-shot prompt ("What medical imaging modality does this image show? Answer with one word: xray, ecg, echo, ct, mri, ultrasound, or other."), based on the answer, load the appropriate LoRA adapter (mapping defined in a config dict at the top of the file), run the specialist model on the image + query, and print the response. Include a --demo flag that runs 3 example inputs (one CXR, one ECG) and prints a formatted output showing: input modality detected, adapter loaded, and model response.

Test this end-to-end with a few examples from your held-out test set. The demo output is something you can show and describe in your submission.

---

### Day 14 — Deliverables

**Git repository checklist:**
- README.md explaining the project, architecture, and how to reproduce training
- /docs/data_catalog.md
- All scripts (eda, preprocessing, training configs, evaluation, inference)
- /outputs/results/ (JSON + Markdown table)
- Sample ECG-to-image conversion outputs in /notebooks/figures/
- Requirements and setup script
- No raw data, no model weights (too large — reference the HuggingFace model ID instead)

**README — Claude Code prompt:**
> Write a README.md for this project. Sections: Project Overview (2 sentences), Architecture (describe the modality-expert system with base router + specialist adapters), Repository Structure, Setup Instructions (setup.sh + Colab link), Data (reference data_catalog.md, note which datasets require credentialing), Training (how to run each config), Evaluation (how to run evaluate.py and interpret results), Results (embed the results table from outputs/results/), Known Limitations (dataset scale, CPU constraint, 3-modality scope), and Roadmap (how to extend to the remaining 7 modalities). Base model throughout is Qwen3-VL-2B-Instruct; note Qwen2.5-VL-2B as the tested fallback.

**Word summary — use this chat:** Paste your results table, the data catalog section headers, and the training approach, and ask Claude to draft a tight one-page German summary covering: Suchstrategie, verwendete Daten, Vorgehen beim Training, wichtigste Ergebnisse. The constraint is one page, so every sentence has to earn its place.

---

## Timeline at a Glance

| Day | Focus | Key Output |
|-----|-------|-----------|
| 1 | Registrations, project skeleton | GitHub repo, smoke test |
| 2–3 | PTB-XL download, ECG EDA, literature | EDA plots, ECG→image pipeline |
| 3–5 | CheXpert + ReXGradient-160K + OpenI download, data sampling | Stratified CSV splits, report corpus ready |
| 5–7 | Data catalog, JSONL pipeline | docs/data_catalog.md, train/val/test JSONL |
| 8 | Cloud setup, ms-swift verify | Confirmed training environment |
| 8–9 | Self-supervised pre-training (CXR + ECG) | Two pre-trained LoRA checkpoints |
| 10–11 | Supervised fine-tuning (CXR + ECG) | Two SFT LoRA checkpoints |
| 12 | Evaluation | Results table: baseline vs. fine-tuned |
| 13 | Router + end-to-end demo | inference.py running full pipeline |
| 14 | Git cleanup, Word summary | Final submission |

---

## Risk Register

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| PhysioNet approval delayed or denied | Medium | No longer a blocking risk. ReXGradient-160K (open, HuggingFace) covers the self-supervised CXR stage; CheXpert + PTB-XL cover supervised training. Document MIMIC in the catalog as the production-scale resource. |
| ms-swift incompatible with Qwen3-VL-2B-Instruct | Very low (confirmed) | Fall back to Qwen2.5-VL-2B; document the version issue |
| Colab session timeout during training | Medium | Use Kaggle (persistent sessions up to 12h); checkpoint every epoch |
| Loss doesn't decrease (training is broken) | Low | Verify on 10 examples first; paste error + config into Claude Code |
| Evaluation shows no improvement over baseline | Low-medium | Report honestly; a flat result on 50 test examples is still a valid finding and shows evaluation rigor |
