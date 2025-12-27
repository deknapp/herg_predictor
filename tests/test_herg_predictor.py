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
