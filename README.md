# Optical digit recognition with RSSPCA

A Python implementation of Robust Sparse Smooth Principal Component Analysis (RSSPCA) using Eq. (32) and Algorithm 1 of [Wang et al., PLOS One, 2025](https://doi.org/10.1371/journal.pone.0323281). The experiment compares 16-coordinate projections with the same Euclidean one-nearest-neighbor classifier on the official UCI Optdigits held-out writers.

## Reproduce

Use Python 3.12 and a fresh environment. The source downloads the two official UCI files if needed and checks their digests.

```bash
python -m pip install -r requirements.txt
python -B -m unittest discover -s tests -v
python -B src/experiment.py select
python -B src/experiment.py evaluate
python -B src/analyze.py
```

`select` reads only the official training partition. It saves the fit/validation indices, validation grid, selected axes and source hashes. `evaluate` verifies these artifacts before reading the official test partition. `analyze.py` recomputes every metric from stored predictions and draws the figures. Use a new `--run-dir` for a fresh run.

The [protocol](docs/PROTOCOL.md) describes the paired clean and training-contamination conditions. Three seeded splits use the same held-out writers. `results` contains the measured selection, per-run scores and summary. The commands above regenerate the figures and per-image predictions.

## Measured result

RSSPCA achieved mean held-out accuracies of 96.72% with clean training and 96.74% with contaminated training. Classical PCA at the same dimension achieved 97.03% and 96.96%. The sparsity and graph terms controlled the loading structure. Recognition scores reflect a fixed dimension, dataset, corruption protocol and classifier.

## Sources

- J. Wang, X. Xie, L. Zhang, J. Li, H. Cai and Y. Feng. Robust sparse smooth principal component analysis for face reconstruction and recognition. PLOS One 20(5), e0323281, 2025. [DOI](https://doi.org/10.1371/journal.pone.0323281).
- Authors' [MATLAB reference](https://github.com/yuzhounh/RSSPCA/tree/58ef723884b915061af2342584cbffb87f304278). The Python implementation uses the paper's relative L1 stopping statistic and a symmetric equivalent of its linear update. No MATLAB files are distributed here.
- E. Alpaydin and C. Kaynak. [Optical Recognition of Handwritten Digits](https://doi.org/10.24432/C50P49). UCI Machine Learning Repository, 1998.
