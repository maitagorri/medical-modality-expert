# Pipeline Examples — Mixed Modality Routing

Model: Qwen3-VL-2B-Instruct. Router = base model (no adapter). Specialist answers = SFT LoRA adapter per modality.
Test set: CheXpert Plus valid split (CXR) · PTB-XL held-out test (ECG).
Routing: 10/10 correct. Answers: 3/10 correct.

| # | Modality | Router | Question | Ground Truth | Prediction | Route ✓ | Answer ✓ |
|---|----------|--------|----------|-------------|------------|---------|---------|
| 1 | ecg | ecg | What is the primary cardiac diagnosis for this 12-lead ECG? … | STTC | STTC | ✓ | ✓ |
| 2 | ecg | ecg | What is the primary cardiac diagnosis for this 12-lead ECG? … | CD | STTC | ✓ | ✗ |
| 3 | xray | xray | Does this chest X-ray show Cardiomegaly? | No | Yes | ✓ | ✗ |
| 4 | ecg | ecg | What is the primary cardiac diagnosis for this 12-lead ECG? … | NORM | CD | ✓ | ✗ |
| 5 | xray | xray | Does this chest X-ray show Edema? | No | Yes | ✓ | ✗ |
| 6 | xray | xray | Does this chest X-ray show Enlarged Cardiomediastinum? | Yes | No | ✓ | ✗ |
| 7 | xray | xray | Does this chest X-ray show Consolidation? | Yes | Yes | ✓ | ✓ |
| 8 | xray | xray | Does this chest X-ray show Atelectasis? | Yes | Yes | ✓ | ✓ |
| 9 | ecg | ecg | What is the primary cardiac diagnosis for this 12-lead ECG? … | HYP | STTC | ✓ | ✗ |
| 10 | ecg | ecg | What is the primary cardiac diagnosis for this 12-lead ECG? … | MI | CD | ✓ | ✗ |
