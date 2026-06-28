# Evaluation Metrics

Val set: CheXpert Plus (CXR, 42 examples) · PTB-XL (ECG, 25 examples).  
**Free-gen Acc** = answer generated autoregressively, scored vs. ground truth 
(zero-shot base model, no adapter; and the SFT adapters). **Token Acc** = 
teacher-forced metric from the training logs (best checkpoint, same val set).  
The two agree for CXR (single-token Yes/No) but diverge for ECG, where 
teacher-forcing inflates multi-token class names.

## CXR — Binary Label Classification

| Stage | Token Acc (teacher-forced) | Free-gen Acc | Eval Loss |
|-------|----------------------------|--------------|-----------|
| Zeroshot   | — | 21.4% | — |
| Pretrain   | 59.1% | — | 1.8984 |
| SFT (best) | 97.6% | 92.9% | 0.0612 |

## ECG — Cardiac Diagnosis Classification

| Stage | Token Acc (teacher-forced) | Free-gen Acc | Eval Loss |
|-------|----------------------------|--------------|-----------|
| Zeroshot   | — | 20.0% | — |
| Pretrain   | 77.8% | — | 0.4594 |
| SFT (best) | 80.0% | 28.0% | 0.4615 |

