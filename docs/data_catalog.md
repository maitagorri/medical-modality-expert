# Data Catalog

Datasets surveyed during Week 1, with the two used for training marked. Access method, size, and license were verified by visiting each source directly (see [literature.md](literature.md) §1 for the search strategy). Paper citations for the used datasets are in [literature.md](literature.md) §2.

**Access values:** `open` (download directly) · `registration` (account / use-agreement, usually same-day) · `credentialed` (identity + training, days).
**Role values:** `pretrain` (self-supervised report/description) · `sft` (supervised labels) · `both` · `documented only`.

---

## 1. Datasets Used (this project)

| Name | Modality | Source | Access | Size | Examples | Label type | License | Role |
|------|----------|--------|--------|------|----------|------------|---------|------|
| **CheXpert Plus** | CXR | [Stanford AIMI](https://stanfordaimi.azurewebsites.net/datasets) · arXiv:2405.19538 | registration | ~1 TB full (we pulled PNG_valid + reports only) | 223,228 studies / 64,725 patients / 187,711 reports | 14 structured labels **+** free-text reports | Stanford Research Use Agreement | both |
| **PTB-XL** | ECG | [PhysioNet](https://doi.org/10.13026/kfzx-aw45) · Sci Data 2020 | open | 1.7 GB | 21,837 ECGs / 18,885 patients | 5 diagnostic superclasses (from 71 SCP statements) | CC BY 4.0 | both |

**Label-schema source (not downloaded separately):** the 14 CXR observations originate from the original **CheXpert** (Irvin et al., AAAI 2019); CheXpert Plus inherits them. In CheXpert Plus the labels live in `findings_fixed.json` (CheXpert Labels folder) and are joined onto the report CSV on `path_to_image` — see `scripts/data/join_chexpert_labels.py`.

**What we actually sampled** (CPU-scoped, see `scripts/data/sample_datasets.py`):
- CXR: 75 train / 25 val from the `train` split; all 234 `valid`-split images held out as test.
- ECG: single-superclass records only, patient-disjoint 75 / 25 / 50 (train / val / test), 30 per superclass balanced.

---

## 2. Datasets Evaluated or Documented (CXR / ECG alternatives)

| Name | Modality | Source | Access | Size | Examples | Label type | License | Role |
|------|----------|--------|--------|------|----------|------------|---------|------|
| MIMIC-CXR | CXR | [PhysioNet](https://physionet.org/content/mimic-cxr/) | credentialed | ~4.7 TB | 377,110 images / 227,835 reports | free-text reports | PhysioNet Credentialed Health Data License 1.5.0 | documented only (production upgrade) |
| ReXGradient-160K | CXR | [HF `rajpurkarlab/ReXGradient-160K`](https://huggingface.co/datasets/rajpurkarlab/ReXGradient-160K) | open | large | 160,000 studies + reports | free-text reports | research-use (per dataset card) | documented only (open fallback) |
| OpenI / IU-Xray | CXR | [Kaggle `raddar/chest-xrays-indiana-university`](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university) | open | ~300 MB | 7,470 images / 3,955 reports | free-text reports | open (educational/research) | documented only (pipeline validation) |
| NIH ChestX-ray14 | CXR | [NIH / Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data) | open | ~45 GB | 112,120 images / 30,805 patients | 14 structured labels | CC0 / public domain | documented only |
| MIMIC-IV-ECG | ECG | [PhysioNet](https://physionet.org/content/mimic-iv-ecg/) | credentialed | ~90 GB | ~800,000 ECGs | diagnostic (linked to MIMIC-IV) | PhysioNet Credentialed Health Data License 1.5.0 | documented only (production upgrade) |
| EchoNet-Dynamic | Echo (video) | [Stanford AIMI](https://echonet.github.io/dynamic/) | registration | ~7 GB | 10,030 echo videos | ejection fraction / tracings | Stanford Research Use Agreement | documented only |
| UniMed | X-ray, CT, MRI, US, Pathology, Fundus | [GitHub `mbzuai-oryx/UniMed-CLIP`](https://github.com/mbzuai-oryx/UniMed-CLIP) · arXiv:2412.10372 | open | large | 5.3M image-text pairs | image-text | Attribution-NonCommercial 4.0 International | documented only (multi-modality scaling) |

---

## 3. Deferred Modalities — best available public starting point

The assessment targets a broader set of modalities than two. For each not yet implemented, the most practical open dataset to start from is recorded below, so extension is a data-acquisition step, not a research-from-scratch step. (All `documented only`.)

| Modality | Candidate dataset | Source | Access | Notes |
|----------|-------------------|--------|--------|-------|
| CT | LIDC-IDRI (lung nodules) / DeepLesion | [TCIA](https://www.cancerimagingarchive.net/) / NIH | open / registration | TCIA is the central hub for open CT/MRI oncology data |
| MRI | fastMRI (knee/brain) / BraTS (brain tumor) | [fastMRI](https://fastmri.med.nyu.edu/) / [Synapse BraTS](https://www.synapse.org/) | registration | BraTS has well-defined segmentation labels |
| Ultrasound | BUSI (breast ultrasound) | [public](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset) | open | small, clean, benign/malignant/normal labels |
| Coronary Angiography | ARCADE / CADICA | [Grand Challenge](https://grand-challenge.org/) | registration | stenosis / vessel segmentation challenges |
| General time series | MIMIC-IV waveform / vital signs | [PhysioNet](https://physionet.org/) | credentialed | bedside monitor waveforms beyond ECG |
| Tables (structured EHR) | MIMIC-IV / eICU-CRD | [PhysioNet](https://physionet.org/content/mimiciv/) | credentialed | structured labs/vitals/diagnoses; pairs with the imaging modalities |

*Fields marked with "~" or "per dataset card" should be re-confirmed against the live source before any production use; licenses on credentialed sets in particular are version-specific.*

---

## 4. Additional Data-Acquisition Strategies (sketch)

Beyond downloading existing open/credentialed datasets, paths to scale data for the remaining modalities:

- **Credentialed access at scale.** Complete PhysioNet credentialing to unlock MIMIC-CXR, MIMIC-IV-ECG, and MIMIC-IV tables — one credential covers a large, multimodal, patient-linked corpus, enabling genuine cross-modality work.
- **Aggregated open corpora.** Use pre-assembled multi-modality collections (UniMed, 5.3M pairs across six modalities) instead of stitching sources individually — fastest route to breadth.
- **Challenge / consortium data.** Grand Challenge and TCIA host curated, labeled sets per modality (segmentation, detection) with clear evaluation protocols — good for benchmarking specific experts.
- **Weak / distant supervision.** Mine labels from free-text reports (as CheXpert's labeler does) to convert large unlabeled report corpora into structured training signal without manual annotation.
- **Synthetic & augmentation.** Waveform-to-image rendering (as done here for ECG) and report-conditioned image generation can expand coverage for scarce modalities — with the caveat that synthetic data supplements, never replaces, real clinical data.
- **Institutional partnership.** For modalities thin in public data (e.g. angiography), a data-use agreement with a clinical partner under appropriate ethics/IRB approval is the realistic source — the production analogue of the public-dataset approach used here.
