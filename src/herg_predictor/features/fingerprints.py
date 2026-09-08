"""Molecular fingerprint computation."""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from tqdm import tqdm


def compute_morgan_fingerprint(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray | None:
    """
    Compute Morgan (ECFP) fingerprint for a molecule.

    Args:
        smiles: SMILES string
        radius: Fingerprint radius (2 = ECFP4, 3 = ECFP6)
        n_bits: Number of bits in fingerprint

    Returns:
        Binary fingerprint as numpy array, or None if invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp, dtype=np.float32)
    except Exception:
        return None


def compute_maccs_fingerprint(smiles: str) -> np.ndarray | None:
    """
    Compute MACCS keys fingerprint for a molecule.

    Args:
        smiles: SMILES string

    Returns:
        MACCS fingerprint (166 bits) as numpy array, or None if invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = MACCSkeys.GenMACCSKeys(mol)
        return np.array(fp, dtype=np.float32)
    except Exception:
        return None


def featurize_fingerprints(
    smiles_list: list[str],
    fingerprint_type: str = "morgan",
    radius: int = 2,
    n_bits: int = 2048,
    show_progress: bool = True,
) -> tuple[np.ndarray, list[int]]:
    """
    Compute fingerprints for a list of SMILES.

    Args:
        smiles_list: List of SMILES strings
        fingerprint_type: 'morgan' or 'maccs'
        radius: Morgan fingerprint radius
        n_bits: Number of bits for Morgan fingerprint
        show_progress: Show progress bar

    Returns:
        Tuple of (fingerprint_matrix, valid_indices)
        - fingerprint_matrix: shape (n_valid, n_bits)
        - valid_indices: indices of successfully featurized molecules
    """
    fingerprints = []
    valid_indices = []

    iterator = tqdm(smiles_list, desc=f"Computing {fingerprint_type} fingerprints") if show_progress else smiles_list

    for idx, smiles in enumerate(iterator):
        if fingerprint_type == "morgan":
            fp = compute_morgan_fingerprint(smiles, radius=radius, n_bits=n_bits)
        elif fingerprint_type == "maccs":
            fp = compute_maccs_fingerprint(smiles)
        else:
            raise ValueError(f"Unknown fingerprint type: {fingerprint_type}")

        if fp is not None:
            fingerprints.append(fp)
            valid_indices.append(idx)

    return np.stack(fingerprints, axis=0), valid_indices
