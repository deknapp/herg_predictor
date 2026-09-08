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

    scaffolds = [get_scaffold(smi) for smi in smiles_list]

    scaffold_to_indices: dict[str, list[int]] = {}
    for idx, scaffold in enumerate(scaffolds):
        if scaffold is None:
            scaffold = f"NONE_{idx}"  # Treat invalid scaffolds as unique
        scaffold_to_indices.setdefault(scaffold, []).append(idx)

    scaffold_sets = list(scaffold_to_indices.values())

    # Shuffle first, then stable-sort by size, so the seed decides the order
    # of equally-sized series and nothing else does. Sorting after shuffling
    # -- which is what this did -- threw the shuffle away; shuffling after
    # sorting threw the sort away, which is the bug below.
    rng.shuffle(scaffold_sets)
    scaffold_sets.sort(key=len, reverse=True)

    # Assign each series to whichever split is furthest below its target,
    # largest series first.
    #
    # The previous version filled train until it reached its quota, then val,
    # then gave the remainder to test. That silently produces empty splits,
    # because the check happens *before* the series is added: a series larger
    # than the space left does not overflow into the next split, it lands
    # whole and overshoots. One series bigger than the val quota therefore
    # consumes val entirely and leaves test with nothing.
    #
    # Not hypothetical, and not subtle once measured. On 100 compounds where
    # 85 share a scaffold, it returned train=100, val=0, test=0 -- and
    # returned it without complaint, so every metric downstream would have
    # been computed on an empty test set and reported as a number. The hERG
    # data happens to have no series large enough to trigger it, which is
    # exactly what makes it worth fixing now rather than after a dataset
    # change makes it fire.
    n_total = len(smiles_list)
    targets = [n_total * train_ratio, n_total * val_ratio, n_total * test_ratio]
    splits: list[list[int]] = [[], [], []]

    for scaffold_indices in scaffold_sets:
        # Deficit rather than free space: a split at 60% of a large target is
        # hungrier than one at 60% of a small one, which is what keeps the
        # proportions right instead of merely keeping every split non-empty.
        deficits = [t - len(s) for t, s in zip(targets, splits)]
        splits[deficits.index(max(deficits))].extend(scaffold_indices)

    train_indices, val_indices, test_indices = splits

    logger.info(
        f"Scaffold split: train={len(train_indices)}, "
        f"val={len(val_indices)}, test={len(test_indices)}"
    )
    logger.info(f"Number of unique scaffolds: {len(scaffold_to_indices)}")

    # A split with nothing in it is not a split, and every metric computed
    # from it downstream is meaningless rather than merely wrong. Say so here,
    # where the cause is visible, rather than letting an AUROC over zero
    # compounds reach a README.
    empty = [name for name, part in
             zip(("train", "val", "test"), splits) if not part]
    if empty:
        raise ValueError(
            f"Scaffold split left {', '.join(empty)} empty: {n_total} compounds "
            f"in {len(scaffold_sets)} scaffold series, the largest holding "
            f"{len(scaffold_sets[0])}. A single series larger than a split's "
            "share cannot be divided, so the requested ratios are not "
            "achievable on this data. Use strategy='random', or widen the "
            "ratios."
        )

    # int, explicitly: an empty np.array([]) is float64, and indexing a
    # DataFrame with float positions raises somewhere far from here.
    return (
        np.array(train_indices, dtype=int),
        np.array(val_indices, dtype=int),
        np.array(test_indices, dtype=int),
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
