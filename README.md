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
- **Baseline**: Random Forest, XGBoost
- **Neural Network**: Feed-forward network on fingerprints
- **Graph Neural Network**: Message-passing neural network (MPNN)

### Evaluation
- Stratified scaffold split (ensures chemical series don't leak between splits)
- Metrics: AUROC, AUPRC, balanced accuracy, sensitivity, specificity
- Applicability domain analysis

## References

- [ChEMBL hERG data](https://www.ebi.ac.uk/chembl/target_report_card/CHEMBL240/)
- Czodrowski, P. (2013). hERG Me Out. J. Chem. Inf. Model.
