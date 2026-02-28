# CXR-Sentinel: Obesity ML Competition (Crunch 1)

**Multi-Strategy Ensemble for Single-Gene Perturbation Prediction**

> Submission to the [Broad Institute Obesity ML Competition](https://hub.crunchdao.com/competitions/broad-obesity-1) (Crunch 1), organized by the Eric and Wendy Schmidt Center at the Broad Institute of MIT and Harvard, in partnership with CrunchDAO.

**Author:** Francesco Orsi

---

## Overview

This repository contains a complete, competition-ready solution for predicting single-cell transcriptomic responses to 2,863 unseen CRISPR/Cas9 single-gene perturbations in human adipocyte differentiation. The model generates 286,300 predicted cell profiles (100 cells per perturbation) across 10,238 genes, plus program proportion estimates for four cell states.

The approach is a classical ML pipeline that prioritizes feature engineering, principled ensemble tuning, and biologically grounded cell distribution modeling over deep learning complexity, motivated by [Ahlmann-Eltze et al. (2025)](https://www.nature.com/articles/s41592-024-02566-4) showing that linear baselines match or outperform deep learning on perturbation prediction.

## Architecture

```
                    Gene Embeddings (98D)
                   /                     \
    Response PCA (50D)          Coexpression PCA (40D) + Stats (8D)
           |                              |
    +-----------+-----------+     +-------+-------+
    |           |           |     |               |
  Ridge-All  Ridge-Reg    KNN   Prog-Ridge    Prog-KNN
    |           |           |     |               |
    +-----+-----+-----+----+     +-------+-------+
          |                               |
   LOO-CV Ensemble (tuned weights)   LOO-CV Blend
          |                               |
     Shrinkage                    3-State Normalization
          |                               |
    Mean Prediction              Program Proportions
          |                               |
          +---------------+---------------+
                          |
              Per-Program Low-Rank Covariance
                          |
              Mixture-of-Programs Sampling
                          |
                   100 Cells / Perturbation
                          |
                   Softplus Activation
                          |
                    prediction.h5ad
```

## Key Design Decisions

**Dual Gene Embeddings:** Two complementary PCA spaces capture different biological signals. Response embeddings (gene-by-perturbation delta matrix) encode how each gene reacts to perturbations. Coexpression embeddings (cell-by-gene matrix) encode baseline regulatory relationships. Together they form robust 98D feature vectors for unseen gene targets.

**Three-Strategy Ensemble with LOO-CV Tuning:** Ridge (flexible), Ridge (heavily regularized), and distance-weighted KNN are blended with weights optimized via 5-fold cross-validation on HVG Pearson correlation, not in-sample fit. A post-ensemble shrinkage factor prevents overconfident extrapolation.

**Per-Program Low-Rank Covariance:** Rather than applying uniform Gaussian noise, cells are generated from a mixture model where each program state (pre_adipo, adipo, lipo, other) has its own low-rank covariance structure estimated from training cells. This preserves gene-gene correlation patterns critical for MMD accuracy.

**Perturbation-Adaptive Noise:** Noise scaling is derived from KNN neighbors' observed variance, ensuring high-variance perturbation targets get appropriately dispersed cell distributions.

**Softplus Non-Negativity:** A steep softplus function replaces hard zero-clamping to avoid point masses at zero that distort MMD kernel estimates.

**Memory-Efficient Output:** The full 286,300 x 10,238 output matrix (11.7 GB) is pre-allocated and filled in-place, avoiding the 2x memory peak of list-then-vstack approaches.

## Repository Structure

```
.
├── main.py                    # Complete submission code (single file, ~1100 lines)
├── requirements.txt           # Python dependencies
├── Method Description.md      # Competition-required method documentation
├── main.ipynb                 # Jupyter notebook version of the submission
├── LICENSE                    # MIT License
└── README.md                  # This file
```

## Requirements

- Python 3.10+
- numpy, pandas, scipy, scikit-learn, anndata
- CPU only (runs on CrunchDAO's 16 vCPU / 64 GB RAM cloud environment)

## Usage

### With CrunchDAO CLI

```bash
pip install crunch-cli --upgrade
crunch setup broad-obesity-1 <your-model-name> --token <YOUR_TOKEN>
cd broad-obesity-1-<your-model-name>

# Replace main.py and requirements.txt with the files from this repo
cp /path/to/this/repo/main.py ./main.py
cp /path/to/this/repo/requirements.txt ./requirements.txt

# Test locally
crunch test

# Submit
crunch push --message "submission" --no-pip-freeze
```

### Standalone

```python
import main

# Train: reads obesity_challenge_1.h5ad, saves checkpoints to model_dir
main.train(data_directory_path="data/", model_directory_path="model/")

# Infer: generates prediction.h5ad and program_proportion.csv
main.infer(
    data_directory_path="data/",
    prediction_directory_path="predictions/",
    prediction_h5ad_file_path="predictions/prediction.h5ad",
    program_proportion_csv_file_path="predictions/predict_program_proportion.csv",
    model_directory_path="model/",
    predict_perturbations=["GENE1", "GENE2", ...],
    genes_to_predict=["G1", "G2", ...],
)
```

## Training Pipeline (5 Batches)

| Batch | Name | Description | Checkpoint Size |
|-------|------|-------------|-----------------|
| 1 | Core Statistics | Per-perturbation means, variances, program proportions, signature genes | ~22 MB |
| 2 | Gene Embeddings | Response PCA + coexpression PCA + feature engineering | ~21 MB |
| 3 | Ridge Ensemble | Dual Ridge + KNN + LOO-CV weights + shrinkage + noise scale | ~66 MB |
| 4 | Covariance | Global + per-program low-rank covariance (rank 10-20) | ~6 MB |
| 5 | Program Model | Per-program Ridge + KNN blend for proportion prediction | ~0.06 MB |

Total checkpoint size: ~115 MB. Training completes in ~2 minutes on the competition environment.

## Output Format

| File | Shape | Description |
|------|-------|-------------|
| `prediction.h5ad` | 286,300 x 10,238 | Dense float32 gene expression predictions |
| `predict_program_proportion.csv` | 2,863 x 5 | Columns: gene, pre_adipo, adipo, lipo, other |

Validation constraints:
- `pre_adipo + adipo + other = 1.0` (tolerance 0.001)
- `lipo` is a separate proportion (not included in the sum)
- `lipo_adipo` is computed by the scoring system as `lipo / adipo`

## Resource Budget

| Resource | Used | Limit | Margin |
|----------|------|-------|--------|
| Disk | 21.1 GB | 40 GB | 47% free |
| RAM (peak) | ~12 GB | 64 GB | 81% free |
| Time | ~16 min | ~30 min | ~50% free |

## Competition Context

The Broad Institute Obesity ML Competition challenges participants to predict single-cell transcriptomic responses to CRISPR/Cas9 perturbations in human adipocyte differentiation. The goal is to understand which transcription factors enhance or prevent differentiation into adipocytes, and which genes could convert white adipocytes into brown/thermogenic cells.

Evaluation metrics:
- **Pearson Delta** -- correlation between predicted and observed perturbation effects relative to perturbed mean (computed on top 1000 HVGs)
- **Maximum Mean Discrepancy (MMD)** -- distributional distance between predicted and observed single-cell profiles
- **L1 Program Proportions** -- weighted L1 distance on cell state proportions (75% four-state + 25% lipo/adipo ratio)

## References

- Ahlmann-Eltze, C. et al. "Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines." *Nature Methods* (2025).
- Roohani, Y. et al. "GEARS: Predicting transcriptional outcomes of novel multi-gene perturbations." *Nature Biotechnology* (2024).
- Broad Institute Obesity ML Competition: [Full Specifications](https://docs.crunchdao.com/competitions/competitions/broad-obesity/full-specifications)

## License

MIT License. See [LICENSE](LICENSE) for details.
