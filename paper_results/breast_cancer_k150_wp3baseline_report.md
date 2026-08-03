# FHE LS-SVM run report


## Run 2026-08-01T11:31:56Z — dataset=breast_cancer, k=150

### Results

**Class 0 (malignant_vs_benign vs rest)**

| Approach | Accuracy | Precision | F1 |
|---|---|---|---|
| Baseline (plaintext solve + encrypted agg) | 91.23% | 86.36% | 88.37% |
| Federated plaintext reference | 91.23% | 86.36% | 88.37% |
| Full-data plaintext reference | 95.61% | 97.44% | 93.83% |

FHE vs plaintext federated weights rel. error: 1.8764e-11


### Time spent

- **Total: n/a**
- Sum of FHE solve times: 0.0 min
