"""Tests for hERG predictor."""

import numpy as np
import pytest


class TestFingerprints:
    """Test molecular fingerprint computation."""

    def test_morgan_fingerprint_valid(self):
        """Test Morgan fingerprint on valid SMILES."""
        from herg_predictor.features.fingerprints import compute_morgan_fingerprint

        smiles = "CCO"  # Ethanol
        fp = compute_morgan_fingerprint(smiles, radius=2, n_bits=2048)

        assert fp is not None
        assert fp.shape == (2048,)
        assert fp.dtype == np.float32
        assert set(np.unique(fp)).issubset({0.0, 1.0})

    def test_morgan_fingerprint_invalid(self):
        """Test Morgan fingerprint on invalid SMILES."""
        from herg_predictor.features.fingerprints import compute_morgan_fingerprint

        fp = compute_morgan_fingerprint("not_a_smiles")
        assert fp is None

    def test_maccs_fingerprint_valid(self):
        """Test MACCS fingerprint on valid SMILES."""
        from herg_predictor.features.fingerprints import compute_maccs_fingerprint

        smiles = "c1ccccc1"  # Benzene
        fp = compute_maccs_fingerprint(smiles)

        assert fp is not None
        assert fp.shape == (167,)  # MACCS has 166 + 1 keys


class TestPreprocessing:
    """Test data preprocessing functions."""

    def test_standardize_smiles_valid(self):
        """Test SMILES standardization."""
        from herg_predictor.data.preprocess import standardize_smiles

        # Test salt removal
        smiles_with_salt = "CCO.[Na]"
        standardized = standardize_smiles(smiles_with_salt)
        assert standardized == "CCO"

    def test_standardize_smiles_invalid(self):
        """Test standardization of invalid SMILES."""
        from herg_predictor.data.preprocess import standardize_smiles

        result = standardize_smiles("invalid_smiles")
        assert result is None


def _series(n_scaffolds: int, per_scaffold: int) -> list[str]:
    """SMILES spanning `n_scaffolds` genuinely distinct Murcko scaffolds.

    Getting this wrong is easy and quiet. A Murcko scaffold is the ring
    systems plus the linkers between them, with side chains stripped -- so
    "c1ccc(cc1)C", "c1ccc(cc1)CC" and "c1ccc(cc1)CCC" are three different
    molecules with *one* scaffold, benzene, and a fixture built that way tests
    a single-scaffold dataset while appearing to test forty. Two rings joined
    by a linker of varying length does give distinct scaffolds, because the
    linker sits between rings and is kept.
    """
    return [
        "c1ccccc1" + "C" * (i + 1) + "c1ccccc1"
        for i in range(n_scaffolds)
        for _ in range(per_scaffold)
    ]


class TestSplits:
    """Test data splitting functions."""

    def test_random_split_ratios(self):
        """Test that random split produces correct ratios."""
        from herg_predictor.data.splits import random_split

        n_samples = 1000
        train_idx, val_idx, test_idx = random_split(
            n_samples, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
        )

        # Check no overlap
        assert len(set(train_idx) & set(val_idx)) == 0
        assert len(set(train_idx) & set(test_idx)) == 0
        assert len(set(val_idx) & set(test_idx)) == 0

        # Check coverage
        assert len(train_idx) + len(val_idx) + len(test_idx) == n_samples

        # Check approximate ratios (allow some tolerance)
        assert 0.75 < len(train_idx) / n_samples < 0.85

    def test_scaffold_split_keeps_series_together(self):
        """A scaffold must not appear on both sides of the split.

        This is the entire point of a scaffold split: if the same chemical
        series is in train and test, the test AUROC measures memorisation of
        that series, not generalisation to a new one.
        """
        from herg_predictor.data.splits import get_scaffold, scaffold_split

        smiles = _series(n_scaffolds=20, per_scaffold=5)
        train, val, test = scaffold_split(smiles)

        parts = [{get_scaffold(smiles[i]) for i in idx}
                 for idx in (train, val, test)]
        assert not parts[0] & parts[1]
        assert not parts[0] & parts[2]
        assert not parts[1] & parts[2]

    def test_scaffold_split_fills_every_part(self):
        """No split may come back empty.

        The greedy version of this filled train to its quota, then val, then
        gave the remainder to test -- checking the quota *before* adding the
        series, so a series larger than the space left overshot rather than
        overflowing. One series bigger than the val quota consumed val and
        left test with nothing, silently.
        """
        from herg_predictor.data.splits import scaffold_split

        smiles = _series(n_scaffolds=40, per_scaffold=5)
        train, val, test = scaffold_split(smiles)

        assert len(train) and len(val) and len(test)
        assert len(train) + len(val) + len(test) == len(smiles)
        assert 0.75 < len(train) / len(smiles) < 0.85
        # Empty numpy arrays default to float64, and a float index into a
        # DataFrame raises a long way from here.
        for idx in (train, val, test):
            assert idx.dtype.kind == "i"

    def test_scaffold_split_refuses_impossible_ratios(self):
        """An unachievable split raises rather than returning an empty one.

        85 of 100 compounds sharing a scaffold cannot be divided 80/10/10,
        because that one series cannot be split without defeating the purpose.
        The honest outcome is an error naming the reason, not a test set of
        zero compounds and an AUROC computed over it.
        """
        from herg_predictor.data.splits import scaffold_split

        smiles = ["c1ccccc1"] * 85 + ["c1ccncc1"] * 15
        with pytest.raises(ValueError, match="empty"):
            scaffold_split(smiles)


class TestMetrics:
    """Test evaluation metrics."""

    def test_classification_metrics(self):
        """Test classification metrics computation."""
        from herg_predictor.evaluation.metrics import compute_classification_metrics

        y_true = np.array([0, 0, 1, 1, 1])
        y_pred_proba = np.array([0.1, 0.2, 0.7, 0.8, 0.9])

        metrics = compute_classification_metrics(y_true, y_pred_proba)

        assert "auroc" in metrics
        assert "auprc" in metrics
        assert "sensitivity" in metrics
        assert "specificity" in metrics
        assert 0 <= metrics["auroc"] <= 1
        assert metrics["n_samples"] == 5

    def test_find_optimal_threshold(self):
        """Test optimal threshold finding."""
        from herg_predictor.evaluation.metrics import find_optimal_threshold

        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        threshold, value = find_optimal_threshold(y_true, y_pred_proba, metric="f1")

        assert 0 < threshold < 1
        assert 0 <= value <= 1


class TestModels:
    """Test model classes."""

    def test_feedforward_forward_pass(self):
        """Test feedforward network forward pass."""
        import torch

        from herg_predictor.models.feedforward import FeedForwardNet

        model = FeedForwardNet(input_dim=2048, hidden_dims=[256, 128])
        x = torch.randn(16, 2048)

        output = model(x)

        assert output.shape == (16, 1)

    def test_feedforward_trains_on_a_learnable_signal(self):
        """The feedforward model must train, not merely have a forward pass.

        Both neural models in this repo were unrunnable and nothing noticed,
        because the only test that touched either of them built a bare
        ``nn.Module`` and pushed a tensor through it. The trainer wrapping it
        referred to numpy without importing it, so the module raised
        NameError on import; the test never reached that code path. A test
        that trains a model on a signal it should be able to find is the
        cheapest thing that would have caught it.
        """
        import torch
        from sklearn.metrics import roc_auc_score

        from herg_predictor.models.feedforward import FeedForwardClassifier

        # Seeded, and torch as well as numpy. Written without the torch seed
        # this test scored 0.84 and then 0.67 on identical code, because the
        # weight initialisation was free to vary -- a test that fails at random
        # gets muted, and a muted test is worse than none.
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        X = rng.normal(size=(600, 32)).astype("float32")
        y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype("float32")

        clf = FeedForwardClassifier(input_dim=32, hidden_dims=[32, 16])
        clf.fit(X[:450], y[:450], X_val=X[450:], y_val=y[450:],
                epochs=40, verbose=False)

        # The bar is "learned something real", not a tuned score: this exists
        # to catch a model that cannot train at all, which is the state it was
        # actually in.
        auroc = roc_auc_score(y[450:], clf.predict_proba(X[450:]))
        assert auroc > 0.75, f"learned nothing from a linear signal: {auroc}"

    def test_gnn_trains_on_real_molecules(self):
        """The GNN must build, batch, and train on actual SMILES.

        Four separate faults sat between this model and a single training
        step, each invisible to an import check:

        * numpy was used but never imported, as in the feedforward model;
        * ``MPNNLayer`` assigned the node *feature width* to ``self.node_dim``,
          which MessagePassing already uses for the propagation *axis*;
        * ``message()`` was annotated ``Tensor | None``, which PyG's signature
          parser cannot read;
        * ``MoleculeDataset`` yielded dicts, which PyG's loader cannot batch
          into the ``batch.x`` / ``batch.edge_index`` the trainer reads.

        So this test goes all the way through: SMILES to graphs to a batch to
        a gradient step to a prediction.
        """
        import torch
        from torch_geometric.loader import DataLoader

        from herg_predictor.features.graphs import (
            MoleculeDataset,
            get_atom_feature_dim,
            get_bond_feature_dim,
        )
        from herg_predictor.models.gnn import GNNClassifier

        torch.manual_seed(0)
        # A bond-less ion is in here on purpose: it produces an empty
        # edge_attr, and a batch containing it will not collate unless that
        # array is present with the right width.
        smiles = ["CCO", "c1ccccc1", "[Na+]", "CC(=O)Oc1ccccc1C(=O)O",
                  "CCN(CC)CC", "c1ccncc1", "CC(C)Cc1ccccc1", "OCC(O)CO"] * 4
        labels = np.array([0.0, 1.0] * 16, dtype="float32")

        loader = DataLoader(MoleculeDataset(smiles, labels), batch_size=8)
        clf = GNNClassifier(
            node_input_dim=get_atom_feature_dim(),
            edge_input_dim=get_bond_feature_dim(),
            hidden_dim=32,
            num_layers=2,
        )
        clf.fit(loader, epochs=3, verbose=False)

        preds = clf.predict_proba(loader)
        assert preds.shape == (len(smiles),)
        assert np.all((preds >= 0) & (preds <= 1))
        assert clf.history["train_loss"], "no training step was taken"

    def test_random_forest_fit_predict(self):
        """Test Random Forest training and prediction."""
        from herg_predictor.models.baseline import RandomForestModel

        X = np.random.randn(100, 50)
        y = np.random.randint(0, 2, 100)

        model = RandomForestModel(n_estimators=10)
        model.fit(X, y)

        predictions = model.predict(X)
        probas = model.predict_proba(X)

        assert predictions.shape == (100,)
        assert probas.shape == (100,)
        assert all(0 <= p <= 1 for p in probas)
