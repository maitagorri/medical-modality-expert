# Literature & Research Log

## 1. Search Strategy

a) identify openly available medical imaging datasets suitable for CPU-scale fine-tuning
b) understand how the field builds multimodal medical models, so the architecture would be grounded in established practice rather than invented from scratch.

**Approach:**

1. **Seed from the task.** Started from the assessment brief itself, using an LLM to surface the canonical entry points (LLaVA-Med for medical VLMs, QLoRA for parameter-efficient fine-tuning).
2. **Iterative keyword search** on Google Scholar / arXiv with terms: *"vision-language model radiology"*, *"multimodal medical foundation model"*, *"chest X-ray report generation"*, *"ECG classification deep learning benchmark"*, *"medical mixture of experts"*. For each hit: read the abstract, skim Methods + Data, record the dataset and benchmark used. Stopped when the same names began recurring (BioViL, CheXagent, MIMIC-CXR, CheXpert).
3. **Manual repository inspection.** Rather than trusting secondary summaries, navigated directly to each data source — PhysioNet, HuggingFace, Kaggle, Grand Challenge, and Stanford AIMI — to verify the actual license, data-use agreement, file formats, and sizes. This is what the [data catalog](data_catalog.md) is built from.
4. **Targeted follow-up** on the modality-expert / MoE angle once the architecture (one base model + per-modality adapters) was clear, which surfaced MedMoE and UniMed-CLIP.

**Search log:**

| Date | Source | Search terms / entry point | Led to |
|------|--------|----------------------------|--------|
| 20.08. | LLM-assisted | Assessment task text | LLaVA-Med, QLoRA |
| 20.08. | arXiv / Scholar | "multimodal medical foundation model", "CXR report generation" | BioViL-T, CheXagent |
| 20.08. | PhysioNet | ECG datasets, open access | PTB-XL, MIMIC-IV-ECG |
| 20.08. | Stanford AIMI | CXR datasets with reports | CheXpert Plus |
| 22.08. | arXiv | "medical mixture of experts", "unified medical image-text" | MedMoE, UniMed-CLIP |
| 22.08. | HuggingFace / Kaggle | Open CXR fallbacks | ReXGradient-160K, OpenI/IU-Xray |

---

## 2. Dataset Papers (data actually used)

### CheXpert Plus (Chambon, Delbrouck et al. 2024) — **primary CXR source**

**Citation:** Chambon P, Delbrouck J-B, Sounack T, Huang S-C, Chen Z, Varma M, Truong SQH, Chuong CT, Langlotz CP. *CheXpert Plus: Augmenting a Large Chest X-ray Dataset with Text Radiology Reports, Patient Demographics and Additional Image Formats.* arXiv:2405.19538, 2024. (Stanford AIMI)

**Dataset:** 223,228 de-identified chest X-ray studies (64,725 patients) in DICOM + PNG, paired with 187,711 section-parsed free-text radiology reports — the largest publicly released radiology text corpus. Stanford Research Use Agreement (registration required).

**Relation to this project:** Primary CXR dataset for **both** training stages from a single source — the free-text reports drive the self-supervised pretraining stage, and the 14 structured labels drive the supervised fine-tuning stage. Note: the report CSV (`df_chexpert_plus_240401`) does *not* carry the 14 binary labels; those live in `findings_fixed.json` and are joined on `path_to_image` (see `scripts/data/join_chexpert_labels.py`).

### CheXpert (Irvin, Rajpurkar et al. 2019) — **label schema origin**

**Citation:** Irvin J, Rajpurkar P, Ko M, et al. *CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison.* AAAI 2019.

**Relation to this project:** Defines the 14-observation label schema (Cardiomegaly, Edema, Consolidation, Atelectasis, Pleural Effusion, …) reused as the yes/no VQA targets in the CXR SFT stage. CheXpert Plus inherits this schema, so the supervised task is directly comparable to the original CheXpert benchmark.

### PTB-XL (Wagner, Strodthoff et al. 2020) — **ECG source**

**Citation:** Wagner P, Strodthoff N, Bousseljot R-D, Kreiseler D, Lunze FI, Samek W, Schaeffter T. *PTB-XL, a large publicly available electrocardiography dataset.* Scientific Data 7, 154 (2020). https://doi.org/10.1038/s41597-020-0495-6 — PhysioNet, **fully open** (no credentialing).

**Dataset:** 21,837 clinical 12-lead ECGs (10 s each) from 18,885 patients, annotated by up to two cardiologists with 71 SCP-ECG statements, which roll up to 5 diagnostic superclasses (NORM, MI, STTC, CD, HYP).

**Relation to this project:** ECG modality for both stages. We use the 5 superclasses as the classification target and render each waveform to a 12-lead PNG so the same image-in/text-out VLM pipeline handles ECG without an ECG-specific encoder. Single-superclass records only, patient-disjoint splits (see `scripts/data/sample_datasets.py`).

---

## 3. Model & Method Papers (field context)

### LLaVA-Med (Li et al. 2023)

**Citation:** Li C, Wong C, Zhang S, Usuyama N, Liu H, Yang J, Naumann T, Poon H, Gao J. *LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day.* NeurIPS 2023 (Datasets & Benchmarks, Spotlight). (Microsoft Research)

**Relevance:** The reference point for adapting a general-domain VLM to medicine cheaply. Initialises from general LLaVA, then curriculum-trains (concept alignment → instruction tuning) on PubMed figure-caption data + GPT-4-generated instructions. Evaluated on VQA-RAD, PathVQA, SLAKE.

**Relation to this project:** Validates the core bet — start from a general instruction-tuned VLM (Qwen3-VL) and adapt with lightweight training rather than pretraining from scratch. Our two-stage (pretrain → SFT) recipe mirrors its curriculum idea at much smaller scale.

### BioViL-T (Bannur et al. 2023)

**Citation:** Bannur S, Hyland S, Liu Q, et al. *Learning to Exploit Temporal Structure for Biomedical Vision-Language Processing.* CVPR 2023. (Microsoft Health Futures)

**Relevance:** CNN-Transformer hybrid jointly trained with a text model on MIMIC-CXR; SOTA on progression classification, phrase grounding, and report generation. Releases the CXR-T temporal benchmark.

**Relation to this project:** Demonstrates what paired image+report pretraining achieves on the same kind of CXR data we use, and motivates the report-generation pretraining stage. Out of scope to reproduce (multi-image/temporal), but anchors the upper bound.

### CheXagent (Chen, Varma et al. 2024)

**Citation:** Chen Z, Varma M, Delbrouck J-B, et al. *CheXagent: Towards a Foundation Model for Chest X-Ray Interpretation.* arXiv:2401.12208, 2024. (Stanford AIMI + Stability AI)

**Relevance:** 8B instruction-tuned CXR foundation model. Introduces **CheXinstruct** (8.5M samples / 35 tasks from 28 datasets) and **CheXbench** (8 clinically relevant CXR tasks) for systematic evaluation.

**Relation to this project:** The full-scale, GPU-trained version of what this PoC sketches. CheXbench is the benchmark a production version of the CXR expert should ultimately report against; our small held-out token-accuracy eval is the CPU-scale stand-in.

### ECG-FM (McKeen et al. 2024)

**Citation:** McKeen K, et al. *ECG-FM: An Open Electrocardiogram Foundation Model.* arXiv:2408.05178, 2024 (JAMIA Open 2025).

**Relevance:** Open ECG foundation model — CNN + transformer encoder, self-supervised on ~1.5M ECGs via contrastive learning + masked reconstruction; benchmarked on MIMIC-IV-ECG including LVEF prediction. Strong in low-label regimes.

**Relation to this project:** The modality-native alternative to our approach — it consumes raw ECG signal, whereas we render ECGs to images and reuse the VLM. ECG-FM marks what a dedicated ECG encoder buys; the waveform-to-image route is the trade-off we accept to keep one unified architecture.

### MedMoE (Chopra et al. 2025) — *closest to our architecture*

**Citation:** Chopra S, Sanchez-Rodriguez G, Mao L, Feola AJ, Li J, Kira Z. *MedMoE: Modality-Specialized Mixture of Experts for Medical Vision-Language Understanding.* arXiv:2506.08356, 2025. (Georgia Tech)

**Relevance:** A Mixture-of-Experts framework that routes multi-scale image features (Swin pyramid) through modality-specialized expert branches, conditioned on diagnostic/report context — improving alignment and retrieval across modalities.

**Datasets:** Pretrained on **UniMed** (5.3M image-text pairs, six modalities). Benchmarked on nine radiology tasks: CheXpert, RSNA, Thyroid, Breast, ACL, Meniscus, and MediMeTA CT (axial / coronal / sagittal). 

**Relation to this project:** The published validation of this project's central design idea: **modality-specialized experts behind a shared backbone, selected by context.** Our system is a lightweight realization — LoRA adapters as the "experts" and a zero-shot prompt as the router — runnable on CPU. MedMoE is the direction a scaled-up version would take.

### UniMed-CLIP (Khattak et al. 2024)

**Citation:** Khattak MU, Kunhimon S, Naseer M, Khan S, Khan FS. *UniMed-CLIP: Towards a Unified Image-Text Pretraining Paradigm for Diverse Medical Imaging Modalities.* arXiv:2412.10372, 2024. (MBZUAI)

**Relevance:** Introduces **UniMed**, an open 5.3M image-text dataset spanning six modalities (X-ray, CT, MRI, Ultrasound, Pathology, Fundus), and trains a unified CLIP across all of them.

**Relation to this project:** One of the main datasets used in MedMOE. The multi-modality scaling path. UniMed is the concrete dataset for extending beyond CXR/ECG to the remaining modalities in the roadmap, and shows that one unified model across six modalities is feasible given the data.

### QLoRA (Dettmers et al. 2023)

**Citation:** Dettmers T, Pagnoni A, Holtzman A, Zettlemoyer L. *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS 2023. arXiv:2305.14314.

**Relevance:** 4-bit quantized base weights + LoRA adapters, making large-model fine-tuning feasible on a single GPU.

**Relation to this project:** The intended GPU upgrade path. We use fp32 LoRA on CPU here because `bitsandbytes` has no CPU backend (so no 4-bit quantization); QLoRA is the first change when moving to a GPU, freeing memory to raise `max_length` and dataset size.

---

## 4. Base Model

**Qwen3-VL-2B-Instruct** — instruction-tuned vision-language model used as both the zero-shot router and the LoRA backbone. Chosen for its small size (feasible in fp32 on CPU), native image+text chat format, and strong general visual grounding. Model card: https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct.

---

## 5. Notes on Coverage

- **Datasets documented but not used** (capacity/credentialing): MIMIC-CXR & MIMIC-IV-ECG (PhysioNet credentialed — production-scale upgrade), ReXGradient-160K (open HF fallback for CXR), OpenI/IU-Xray (small, pipeline validation), NIH ChestX-ray14. See [data_catalog.md](data_catalog.md).
- **Benchmarks identified for a scaled version:** CheXbench (CXR), MIMIC-IV-ECG eval (ECG). The current PoC reports held-out token accuracy on small splits as the CPU-feasible proxy.
