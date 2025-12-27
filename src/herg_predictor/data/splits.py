"""Train/validation/test splitting strategies."""

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_scaffold(smiles: str) -> str | None:
    """Extract Murcko scaffold from SMILES."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, canonical=True)
    except Exception:
        return None


def scaffold_split(
    smiles_list: Sequence[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data by Murcko scaffolds to prevent data leakage.
    
    Compounds sharing the same scaffold are kept in the same split,
    ensuring the model is evaluated on truly novel chemical series.
    
    Args:
        smiles_list: List of SMILES strings
        train_ratio: Fraction for training
        val_ratio: Fraction for validation  
        test_ratio: Fraction for test
        seed: Random seed
        
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    rng = np.random.default_rng(seed)
    
    # Get scaffolds for each molecule
    scaffolds = [get_scaffold(smi) for smi in smiles_list]
    
    # Group molecules by scaffold
    scaffold_to_indices: dict[str, list[int]] = {}
    for idx, scaffold in enumerate(scaffolds):
        if scaffold is None:
            scaffold = f"NONE_{idx}"  # Treat invalid scaffolds as unique
        if scaffold not in scaffold_to_indices:
            scaffold_to_indices[scaffold] = []
        scaffold_to_indices[scaffold].append(idx)
    
    # Sort scaffolds by size (largest first) for more balanced splits
    scaffold_sets = list(scaffold_to_indices.values())
    scaffold_sets.sort(key=len, reverse=True)
    
    # Shuffle scaffold order (after sorting by size)
    rng.shuffle(scaffold_sets)
    
    # Assign scaffolds to splits
    n_total = len(smiles_list)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    for scaffold_indices in scaffold_sets:
        if len(train_indices) < n_train:
            train_indices.extend(scaffold_indices)
        elif len(val_indices) < n_val:
            val_indices.extend(scaffold_indices)
        else:
            test_indices.extend(scaffold_indices)
    
    logger.info(f"Scaffold split: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
    logger.info(f"Number of unique scaffolds: {len(scaffold_to_indices)}")
    
    return (
        np.array(train_indices),
        np.array(val_indices),
        np.array(test_indices),
    )


def random_split(
    n_samples: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    stratify: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Random split with optional stratification.
    
    Args:
        n_samples: Number of samples
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for test
        seed: Random seed
        stratify: Labels for stratified splitting
        
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    indices = np.arange(n_samples)
    
    # First split: train vs (val + test)
    train_indices, temp_indices = train_test_split(
        indices,
        train_size=train_ratio,
        random_state=seed,
        stratify=stratify,
    )
    
    # Second split: val vs test
    relative_val_ratio = val_ratio / (val_ratio + test_ratio)
    temp_stratify = stratify[temp_indices] if stratify is not None else None
    
    val_indices, test_indices = train_test_split(
        temp_indices,
        train_size=relative_val_ratio,
        random_state=seed,
        stratify=temp_stratify,
    )
    
    logger.info(f"Random split: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
    
    return train_indices, val_indices, test_indices


def get_split(
    df: pd.DataFrame,
    strategy: str = "scaffold",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe using specified strategy.
    
    Args:
        df: DataFrame with 'smiles' column
        strategy: 'scaffold' or 'random'
        train_ratio, val_ratio, test_ratio: Split fractions
        seed: Random seed
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    if strategy == "scaffold":
        train_idx, val_idx, test_idx = scaffold_split(
            df["smiles"].tolist(),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
    elif strategy == "random":
        train_idx, val_idx, test_idx = random_split(
            len(df),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            stratify=df["label"].values if "label" in df.columns else None,
        )
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")
    
    return df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]
