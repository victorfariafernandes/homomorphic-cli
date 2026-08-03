# FHE LS-SVM run report


## Run 2026-08-01T11:18:23Z — dataset=breast_cancer, k=40

### Results

**Class 0 (malignant_vs_benign vs rest)**

| Approach | Accuracy | Precision | F1 |
|---|---|---|---|
| Baseline (plaintext solve + encrypted agg) | 80.70% | 67.86% | 77.55% |
| Federated plaintext reference | 80.70% | 67.86% | 77.55% |
| Full-data plaintext reference | 95.61% | 97.44% | 93.83% |

FHE vs plaintext federated weights rel. error: 1.1298e-11


### Time spent

- **Total: n/a**
- Sum of FHE solve times: 0.0 min
