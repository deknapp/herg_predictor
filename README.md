# hERG Inhibition Predictor

Machine learning models for predicting hERG (human ether-à-go-go-related gene) potassium channel inhibition from molecular structure. hERG inhibition is a critical safety endpoint in drug discovery—compounds that block this channel can cause fatal cardiac arrhythmias (QT prolongation).

## Project Goals

1. Curate and prepare a high-quality hERG inhibition dataset from public sources
2. Compare multiple molecular featurization approaches (fingerprints, descriptors, learned representations)
3. Train and evaluate multiple model architectures (baseline ML, neural networks, graph neural networks)
4. Provide calibrated predictions with applicability domain assessment

## Installation

```bash
# Create conda environment
conda create -n herg python=3.10
conda activate herg

# Install PyTorch (adjust for your CUDA version, or use CPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
herg-predictor/
├── configs/                 # Model and experiment configurations
│   └── default.yaml
├── data/                    # Raw and processed datasets (not tracked in git)
├── notebooks/               # Exploration and analysis notebooks
├── src/herg_predictor/
│   ├── data/                # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── download.py      # Fetch data from ChEMBL
│   │   ├── preprocess.py    # Cleaning and standardization
│   │   └── splits.py        # Train/val/test splitting strategies
│   ├── features/            # Molecular featurization
│   │   ├── __init__.py
│   │   ├── fingerprints.py  # Morgan, MACCS, etc.
│   │   ├── descriptors.py   # RDKit descriptors
│   │   └── graphs.py        # Graph representations for GNNs
│   ├── models/              # Model architectures
│   │   ├── __init__.py
│   │   ├── baseline.py      # Random forest, XGBoost
│   │   ├── feedforward.py   # Simple neural networks
│   │   └── gnn.py           # Graph neural networks
│   ├── evaluation/          # Metrics and analysis
│   │   ├── __init__.py
│   │   ├── metrics.py       # Classification metrics
│   │   └── analysis.py      # Error analysis, applicability domain
│   └── train.py             # Training loop
├── tests/                   # Unit tests
├── requirements.txt
└── README.md
```

## Usage

```bash
# Download and preprocess data
python -m herg_predictor.data.download
python -m herg_predictor.data.preprocess

# Train a model
python -m herg_predictor.train --config configs/default.yaml

# Evaluate
python -m herg_predictor.evaluation.analysis --model checkpoints/best_model.pt
```

## Results

Baselines on Morgan/ECFP4 fingerprints (radius 2, 2048 bits), trained by
`scripts/train_baselines.py`. Every number here was produced by that script on
11,059 compounds; `results/baseline_metrics.json` holds the full output.

**Dataset.** 11,059 unique compounds from ChEMBL target CHEMBL240 (hERG), IC50
and Ki, exact (`=`) measurements in nM only, aggregated by compound. Blocker is
IC50 <= 10 µM, which leaves 7,226 blockers and 3,833 non-blockers — 65.3%
positive. That imbalance is a property of what gets measured for hERG, not of
the sampling: compounds reach a hERG assay because someone already suspected
them.

**Headline: scaffold split** (9,956 train / 1,103 test, 5,757 unique
scaffolds). Chemical series are kept whole, so a test compound never has a
close analogue in training. This is the number that matters.

| Model | AUROC | AUPRC | Balanced acc. | Sensitivity | Specificity |
|---|---|---|---|---|---|
| Logistic regression | 0.749 | 0.855 | 0.666 | 0.704 | 0.628 |
| Random forest | 0.801 | 0.880 | 0.724 | 0.784 | 0.665 |
| Gradient boosting | 0.800 | 0.883 | 0.692 | 0.865 | 0.519 |

**Random split** (9,953 train / 1,106 test), reported only to show how much the
easier split flatters:

| Model | AUROC | AUPRC | Balanced acc. | Sensitivity | Specificity |
|---|---|---|---|---|---|
| Logistic regression | 0.832 | 0.897 | 0.761 | 0.791 | 0.731 |
| Random forest | 0.860 | 0.919 | 0.773 | 0.812 | 0.734 |
| Gradient boosting | 0.858 | 0.916 | 0.758 | 0.871 | 0.645 |

The gap is the point. Random forest reads 0.860 AUROC on a random split and
0.801 on a scaffold split — **0.059 of the apparent performance is analogues
leaking across the split**, not chemistry the model learned. A published hERG
model quoting a random-split number is quoting the larger of the two.

Gradient boosting is worth reading past its AUROC: it ties random forest at
0.800 but gets there with 0.865 sensitivity against 0.519 specificity — it
calls almost everything a blocker. On a 65% positive set that is close to free.
Balanced accuracy (0.692 vs random forest's 0.724) is the honest comparison.

### What this is not

A ChEMBL-derived structure-activity model, not a validated safety assay. The
labels are heterogeneous — different assays, cell lines, and protocols pooled
under one threshold — and a 10 µM cutoff on aggregated public data is a coarse
instrument. Treat it as a triage signal over the chemical space ChEMBL covers,
and not as evidence about any compound outside it.

## Data Sources

- **ChEMBL**: hERG bioactivity data (IC50, Ki) from CHEMBL240 target
- Threshold for classification: IC50 < 10 μM = inhibitor (positive class)

## Methods

### Featurization
- Morgan fingerprints (ECFP4, 2048 bits)
- MACCS keys (166 bits)  
- RDKit 2D descriptors (200 physicochemical properties)
- Graph representation (atoms as nodes, bonds as edges)

### Models
- **Baseline**: logistic regression, random forest, gradient boosting
  (sklearn's `HistGradientBoostingClassifier`). XGBoost is deliberately not in
  the baseline set: it needs a system library and adds nothing over
  `HistGradientBoostingClassifier` on fingerprints. `XGBoostModel` still works
  if libomp is installed.
- **Neural Network**: Feed-forward network on fingerprints
- **Graph Neural Network**: Message-passing neural network (MPNN)

### Evaluation
- Stratified scaffold split (ensures chemical series don't leak between splits)
- Metrics: AUROC, AUPRC, balanced accuracy, sensitivity, specificity
- Applicability domain analysis

## References

- [ChEMBL hERG data](https://www.ebi.ac.uk/chembl/target_report_card/CHEMBL240/)
- Czodrowski, P. (2013). hERG Me Out. J. Chem. Inf. Model.
