"""Train and evaluate the fingerprint baselines, and write the numbers down.

This is the script behind the Results table in the README. It reports both a
scaffold split and a random split on purpose: random splits leak chemical
series between train and test and flatter a model by several points of AUROC,
so the scaffold number is the honest one and the gap between them is worth
seeing.

    python scripts/train_baselines.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from herg_predictor.data.preprocess import preprocess_herg_data  # noqa: E402
from herg_predictor.data.splits import scaffold_split  # noqa: E402
from herg_predictor.evaluation.metrics import compute_classification_metrics  # noqa: E402
from herg_predictor.features.fingerprints import featurize_fingerprints  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("train")

SEED = 42
THRESHOLD_NM = 10000.0  # 10 uM -- the conventional hERG blocker cutoff


def models():
    """The three baselines, cheapest first. None need a system library."""
    return {
        "Logistic regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=SEED),
        "Random forest": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, class_weight="balanced",
            n_jobs=-1, random_state=SEED),
        "Gradient boosting": HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.08, random_state=SEED),
    }


def evaluate(X, y, train_idx, test_idx) -> dict[str, dict]:
    out = {}
    for name, model in models().items():
        model.fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[test_idx])[:, 1]
        out[name] = compute_classification_metrics(y[test_idx], proba)
        log.info("    %-20s AUROC %.3f", name, out[name]["auroc"])
    return out


def main() -> None:
    df = preprocess_herg_data(
        input_path=ROOT / "data/raw/herg_chembl.csv",
        output_path=ROOT / "data/processed/herg_clean.parquet",
        activity_threshold_nm=THRESHOLD_NM,
    )
    log.info("\nDataset: %d compounds", len(df))

    smiles = df["smiles"].tolist()
    X, valid = featurize_fingerprints(smiles, "morgan", radius=2, n_bits=2048,
                                      show_progress=False)
    df = df.iloc[valid].reset_index(drop=True)
    y = df["label"].to_numpy()
    smiles = df["smiles"].tolist()

    n_pos = int(y.sum())
    log.info("Featurized: %d compounds x %d bits", X.shape[0], X.shape[1])
    log.info("Class balance: %d blockers (%.1f%%) / %d non-blockers",
             n_pos, 100 * n_pos / len(y), len(y) - n_pos)

    results = {
        "dataset": {
            "n_compounds": int(len(y)),
            "n_blockers": n_pos,
            "n_non_blockers": int(len(y) - n_pos),
            "blocker_fraction": round(float(n_pos / len(y)), 4),
            "threshold_nm": THRESHOLD_NM,
            "target": "CHEMBL240 (hERG)",
            "features": "Morgan/ECFP4, radius 2, 2048 bits",
        },
        "splits": {},
    }

    # Scaffold split -- the headline. Train+val are merged; these baselines
    # have no early stopping to tune against.
    tr, va, te = scaffold_split(smiles, 0.8, 0.1, 0.1, seed=SEED)
    tr = np.concatenate([tr, va])
    log.info("\n  Scaffold split: %d train / %d test", len(tr), len(te))
    results["splits"]["scaffold"] = {
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "models": evaluate(X, y, tr, te),
    }

    # Random split -- shown only to quantify how much it flatters.
    idx = np.arange(len(y))
    tr_r, te_r = train_test_split(idx, test_size=0.1, random_state=SEED, stratify=y)
    log.info("\n  Random split: %d train / %d test", len(tr_r), len(te_r))
    results["splits"]["random"] = {
        "n_train": int(len(tr_r)), "n_test": int(len(te_r)),
        "models": evaluate(X, y, tr_r, te_r),
    }

    payload = json.dumps(results, indent=2)
    out = ROOT / "results" / "baseline_metrics.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(payload)
    log.info("\nWrote %s", out.relative_to(ROOT))

    # The published page reads its own copy, and until now nothing kept the
    # two in step: re-running this script updated the repo's numbers and left
    # the live page showing the old ones, with no error anywhere. Written from
    # the same string in the same run so they cannot disagree.
    page = ROOT / "site" / "data" / "metrics.json"
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(payload)
    log.info("Wrote %s", page.relative_to(ROOT))

    # The markdown table that goes in the README.
    log.info("\n| Model | AUROC | AUPRC | Balanced acc. | Sensitivity | Specificity |")
    log.info("|---|---|---|---|---|---|")
    for name, m in results["splits"]["scaffold"]["models"].items():
        log.info("| %s | %.3f | %.3f | %.3f | %.3f | %.3f |", name, m["auroc"],
                 m["auprc"], m["balanced_accuracy"], m["sensitivity"], m["specificity"])


if __name__ == "__main__":
    main()
