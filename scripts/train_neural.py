"""Put the two neural models on the same split as the fingerprint baselines.

Companion to ``train_baselines.py``, deliberately separate rather than folded
into it. The baselines run in seconds on scikit-learn alone; these need torch
and torch-geometric and take minutes on a CPU. Splitting them keeps the cheap
numbers cheap to reproduce, and keeps a torch install from being a
prerequisite for the table everyone actually reads.

What matters is that the two scripts compare like with like: same threshold,
same seed, same scaffold split, same metrics function. Anything else and the
rows in the README are not on the same scale.

One difference, and it is a real one rather than an oversight. The baselines
merge train and val, because they have nothing to early-stop against. These
models do, so they keep the validation set and never see the extra 1,106
compounds the baselines train on. That favours the baselines slightly, and it
is the honest arrangement: giving a neural network its validation set back as
training data means selecting the stopping epoch on the test set.

    python scripts/train_neural.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

from herg_predictor.data.preprocess import preprocess_herg_data  # noqa: E402
from herg_predictor.data.splits import scaffold_split  # noqa: E402
from herg_predictor.evaluation.metrics import compute_classification_metrics  # noqa: E402
from herg_predictor.features.fingerprints import featurize_fingerprints  # noqa: E402
from herg_predictor.features.graphs import (  # noqa: E402
    MoleculeDataset,
    get_atom_feature_dim,
    get_bond_feature_dim,
)
from herg_predictor.models.feedforward import FeedForwardClassifier  # noqa: E402
from herg_predictor.models.gnn import GNNClassifier  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("neural")

SEED = 42
THRESHOLD_NM = 10000.0  # matches train_baselines.py; the two must agree
EPOCHS = 100
PATIENCE = 15
BATCH_SIZE = 64


def train_feedforward(X, y, tr, va, te) -> dict:
    """A dense network on the same ECFP4 bits the baselines use.

    The point of this row is that it isolates the architecture: it sees
    exactly the features random forest sees, so any difference between them is
    the model and not the representation.
    """
    torch.manual_seed(SEED)
    clf = FeedForwardClassifier(input_dim=X.shape[1], hidden_dims=[512, 256, 128])
    clf.fit(
        X[tr], y[tr].astype("float32"),
        X_val=X[va], y_val=y[va].astype("float32"),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        early_stopping_patience=PATIENCE, verbose=False,
    )
    return compute_classification_metrics(y[te], clf.predict_proba(X[te]))


def train_gnn(smiles, y, tr, va, te) -> dict:
    """A message-passing network on the molecular graphs themselves.

    The only model here that is not given a fingerprint, so it is the only one
    that could in principle learn a feature ECFP4 cannot express.
    """
    torch.manual_seed(SEED)

    def loader(idx, shuffle):
        subset = [smiles[i] for i in idx]
        ds = MoleculeDataset(subset, y[idx].astype("float32"))
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle), ds

    train_loader, _ = loader(tr, True)
    val_loader, _ = loader(va, False)
    test_loader, test_ds = loader(te, False)

    # Same correction the baselines get from class_weight="balanced".
    n_pos = float(y[tr].sum())
    pos_weight = (len(tr) - n_pos) / n_pos

    clf = GNNClassifier(
        node_input_dim=get_atom_feature_dim(),
        edge_input_dim=get_bond_feature_dim(),
        hidden_dim=128, num_layers=3, dropout=0.2,
    )
    clf.fit(train_loader, val_loader, epochs=EPOCHS,
            early_stopping_patience=PATIENCE, pos_weight=pos_weight, verbose=False)

    # MoleculeDataset drops molecules RDKit cannot parse, so the labels to
    # score against are the dataset's, not y[te] -- which would silently
    # misalign predictions with truth if even one compound were dropped.
    return compute_classification_metrics(test_ds.labels, clf.predict_proba(test_loader))


def main() -> None:
    df = preprocess_herg_data(
        input_path=ROOT / "data/raw/herg_chembl.csv",
        output_path=ROOT / "data/processed/herg_clean.parquet",
        activity_threshold_nm=THRESHOLD_NM,
    )
    smiles = df["smiles"].tolist()
    X, valid = featurize_fingerprints(smiles, "morgan", radius=2, n_bits=2048,
                                      show_progress=False)
    df = df.iloc[valid].reset_index(drop=True)
    y = df["label"].to_numpy()
    smiles = df["smiles"].tolist()

    tr, va, te = scaffold_split(smiles, 0.8, 0.1, 0.1, seed=SEED)
    log.info("Scaffold split: %d train / %d val / %d test", len(tr), len(va), len(te))

    results = {
        "seed": SEED,
        "threshold_nm": THRESHOLD_NM,
        "split": {"n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te))},
        "note": "Train and val kept separate; the baselines merge them.",
        "models": {},
    }

    log.info("\nFeed-forward network on ECFP4 ...")
    results["models"]["Feed-forward (ECFP4)"] = train_feedforward(X, y, tr, va, te)
    log.info("    AUROC %.3f", results["models"]["Feed-forward (ECFP4)"]["auroc"])

    log.info("\nMessage-passing GNN on graphs ...")
    results["models"]["GNN (MPNN)"] = train_gnn(smiles, y, tr, va, te)
    log.info("    AUROC %.3f", results["models"]["GNN (MPNN)"]["auroc"])

    out = ROOT / "results" / "neural_metrics.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    log.info("\nWrote %s", out.relative_to(ROOT))

    log.info("\n| Model | AUROC | AUPRC | Balanced acc. | Sensitivity | Specificity |")
    log.info("|---|---|---|---|---|---|")
    for name, m in results["models"].items():
        log.info("| %s | %.3f | %.3f | %.3f | %.3f | %.3f |", name, m["auroc"],
                 m["auprc"], m["balanced_accuracy"], m["sensitivity"], m["specificity"])


if __name__ == "__main__":
    main()
