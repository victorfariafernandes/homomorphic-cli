# FHE LS-SVM run report


## Run 2026-08-01T11:18:23Z — dataset=iris, k=40

### Results

**Class 0 (setosa vs rest)**

| Approach | Accuracy | Precision | F1 |
|---|---|---|---|
| Baseline (plaintext solve + encrypted agg) | 100.00% | 100.00% | 100.00% |
| Federated plaintext reference | 100.00% | 100.00% | 100.00% |
| Full-data plaintext reference | 100.00% | 100.00% | 100.00% |

FHE vs plaintext federated weights rel. error: 2.8994e-12

**Class 1 (versicolor vs rest)**

| Approach | Accuracy | Precision | F1 |
|---|---|---|---|
| Baseline (plaintext solve + encrypted agg) | 86.67% | 71.43% | 83.33% |
| Federated plaintext reference | 86.67% | 71.43% | 83.33% |
| Full-data plaintext reference | 93.33% | 83.33% | 90.91% |

FHE vs plaintext federated weights rel. error: 2.2757e-12

**Class 2 (virginica vs rest)**

| Approach | Accuracy | Precision | F1 |
|---|---|---|---|
| Baseline (plaintext solve + encrypted agg) | 76.67% | 58.82% | 74.07% |
| Federated plaintext reference | 76.67% | 58.82% | 74.07% |
| Full-data plaintext reference | 96.67% | 90.91% | 95.24% |

FHE vs plaintext federated weights rel. error: 3.7672e-12

- Multiclass accuracy (Baseline, k=40): 86.67%

### Time spent

- **Total: n/a**
- Sum of FHE solve times: 0.0 min
