"""Main training script for hERG prediction models."""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch_geometric.data import Data

from herg_predictor.data import get_split
from herg_predictor.features import featurize_fingerprints, featurize_descriptors, MoleculeDataset
from herg_predictor.features.graphs import get_atom_feature_dim, get_bond_feature_dim, mol_to_graph
from herg_predictor.models import (
    RandomForestModel,
    XGBoostModel,
    FeedForwardClassifier,
    GNNClassifier,
)
from herg_predictor.evaluation import compute_classification_metrics, compute_metrics_with_ci

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def prepare_fingerprint_data(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    fp_type: str = "morgan",
    radius: int = 2,
    n_bits: int = 2048,
) -> tuple:
    """Prepare fingerprint features for training."""
    logger.info(f"Computing {fp_type} fingerprints...")
    
    X_train, train_valid = featurize_fingerprints(
        df_train["smiles"].tolist(), fingerprint_type=fp_type, radius=radius, n_bits=n_bits
    )
    X_val, val_valid = featurize_fingerprints(
        df_val["smiles"].tolist(), fingerprint_type=fp_type, radius=radius, n_bits=n_bits
    )
    X_test, test_valid = featurize_fingerprints(
        df_test["smiles"].tolist(), fingerprint_type=fp_type, radius=radius, n_bits=n_bits
    )
    
    y_train = df_train["label"].values[train_valid]
    y_val = df_val["label"].values[val_valid]
    y_test = df_test["label"].values[test_valid]
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def prepare_gnn_data(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    batch_size: int = 64,
) -> tuple:
    """Prepare graph data loaders for GNN training."""
    logger.info("Preparing graph data...")
    
    def create_data_list(df: pd.DataFrame) -> list[Data]:
        data_list = []
        for _, row in df.iterrows():
            graph = mol_to_graph(row["smiles"])
            if graph is not None:
                data = Data(
                    x=torch.from_numpy(graph["node_features"]),
                    edge_index=torch.from_numpy(graph["edge_index"]),
                    y=torch.tensor(row["label"], dtype=torch.float),
                )
                if graph["edge_features"] is not None:
                    data.edge_attr = torch.from_numpy(graph["edge_features"])
                data_list.append(data)
        return data_list
    
    train_data = create_data_list(df_train)
    val_data = create_data_list(df_val)
    test_data = create_data_list(df_test)
    
    train_loader = GeometricDataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = GeometricDataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = GeometricDataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    return train_loader, val_loader, test_loader


def train_baseline_model(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict,
) -> RandomForestModel | XGBoostModel:
    """Train a baseline model (Random Forest or XGBoost)."""
    logger.info(f"Training {model_type} model...")
    
    if model_type == "random_forest":
        model = RandomForestModel(**config["model"]["random_forest"])
        model.fit(X_train, y_train)
    elif model_type == "xgboost":
        model = XGBoostModel(**config["model"]["xgboost"])
        model.fit(X_train, y_train, X_val, y_val)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def train_feedforward_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict,
) -> FeedForwardClassifier:
    """Train a feed-forward neural network."""
    logger.info("Training feed-forward neural network...")
    
    model = FeedForwardClassifier(
        input_dim=X_train.shape[1],
        **config["model"]["feedforward"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        device=config["training"]["device"],
    )
    
    model.fit(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=config["training"]["epochs"],
        batch_size=config["training"]["batch_size"],
        early_stopping_patience=config["training"]["early_stopping_patience"],
    )
    
    return model


def train_gnn_model(
    train_loader,
    val_loader,
    config: dict,
) -> GNNClassifier:
    """Train a graph neural network."""
    logger.info("Training graph neural network...")
    
    # Get feature dimensions
    node_dim = get_atom_feature_dim()
    edge_dim = get_bond_feature_dim()
    
    model = GNNClassifier(
        node_input_dim=node_dim,
        edge_input_dim=edge_dim,
        **config["model"]["gnn"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        device=config["training"]["device"],
    )
    
    # Compute class weight
    all_labels = []
    for batch in train_loader:
        all_labels.extend(batch.y.numpy())
    all_labels = np.array(all_labels)
    pos_weight = (all_labels == 0).sum() / (all_labels == 1).sum()
    
    model.fit(
        train_loader,
        val_loader,
        epochs=config["training"]["epochs"],
        early_stopping_patience=config["training"]["early_stopping_patience"],
        pos_weight=pos_weight,
    )
    
    return model


def evaluate_model(
    model,
    X_test: np.ndarray | None,
    y_test: np.ndarray,
    test_loader=None,
    is_gnn: bool = False,
) -> dict:
    """Evaluate model and return metrics."""
    if is_gnn:
        y_pred_proba = model.predict_proba(test_loader)
    else:
        y_pred_proba = model.predict_proba(X_test)
    
    metrics = compute_classification_metrics(y_test, y_pred_proba)
    metrics_ci = compute_metrics_with_ci(y_test, y_pred_proba)
    
    logger.info("Test Set Results:")
    logger.info(f"  AUROC: {metrics['auroc']:.4f} ({metrics_ci['auroc']['lower']:.4f}-{metrics_ci['auroc']['upper']:.4f})")
    logger.info(f"  AUPRC: {metrics['auprc']:.4f} ({metrics_ci['auprc']['lower']:.4f}-{metrics_ci['auprc']['upper']:.4f})")
    logger.info(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    logger.info(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    logger.info(f"  Specificity: {metrics['specificity']:.4f}")
    logger.info(f"  F1 Score: {metrics['f1']:.4f}")
    
    return {"metrics": metrics, "metrics_ci": metrics_ci, "y_pred_proba": y_pred_proba}


def main(config_path: str):
    """Main training pipeline."""
    config = load_config(config_path)
    
    # Create output directories
    Path(config["output"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["output"]["results_dir"]).mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info(f"Loading data from {config['data']['processed_path']}")
    df = pd.read_parquet(config["data"]["processed_path"])
    logger.info(f"Loaded {len(df)} compounds")
    
    # Split data
    df_train, df_val, df_test = get_split(
        df,
        strategy=config["splitting"]["strategy"],
        train_ratio=config["splitting"]["train_ratio"],
        val_ratio=config["splitting"]["val_ratio"],
        test_ratio=config["splitting"]["test_ratio"],
        seed=config["splitting"]["seed"],
    )
    
    model_type = config["model"]["type"]
    
    if model_type in ["random_forest", "xgboost", "feedforward"]:
        # Prepare fingerprint features
        X_train, y_train, X_val, y_val, X_test, y_test = prepare_fingerprint_data(
            df_train, df_val, df_test,
            fp_type="morgan",
            radius=config["features"]["fingerprints"]["morgan"]["radius"],
            n_bits=config["features"]["fingerprints"]["morgan"]["n_bits"],
        )
        
        if model_type in ["random_forest", "xgboost"]:
            model = train_baseline_model(model_type, X_train, y_train, X_val, y_val, config)
        else:
            model = train_feedforward_model(X_train, y_train, X_val, y_val, config)
        
        results = evaluate_model(model, X_test, y_test)
        
    elif model_type == "gnn":
        train_loader, val_loader, test_loader = prepare_gnn_data(
            df_train, df_val, df_test,
            batch_size=config["training"]["batch_size"],
        )
        
        # Get test labels
        y_test = np.array([batch.y.numpy() for batch in test_loader]).flatten()
        
        model = train_gnn_model(train_loader, val_loader, config)
        results = evaluate_model(model, None, y_test, test_loader, is_gnn=True)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Save results
    results_path = Path(config["output"]["results_dir"]) / f"{model_type}_results.yaml"
    with open(results_path, "w") as f:
        # Convert numpy types for YAML serialization
        metrics_serializable = {k: float(v) if isinstance(v, (np.floating, float)) else v 
                               for k, v in results["metrics"].items()}
        yaml.dump({"metrics": metrics_serializable}, f)
    
    logger.info(f"Results saved to {results_path}")
    
    return model, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train hERG prediction model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    args = parser.parse_args()
    
    main(args.config)
