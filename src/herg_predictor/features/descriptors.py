"""RDKit molecular descriptor computation."""

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from tqdm import tqdm


# Get all 2D RDKit descriptor names
DESCRIPTOR_NAMES = [name for name, _ in Descriptors.descList]


def compute_rdkit_descriptors(smiles: str) -> np.ndarray | None:
    """
    Compute all 2D RDKit descriptors for a molecule.
    
    Args:
        smiles: SMILES string
        
    Returns:
        Array of descriptor values (~200 descriptors), or None if invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        calculator = MoleculeDescriptors.MolecularDescriptorCalculator(DESCRIPTOR_NAMES)
        descriptors = calculator.CalcDescriptors(mol)
        
        # Convert to array and handle any infinities
        arr = np.array(descriptors, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        
        return arr
    except Exception:
        return None


def featurize_descriptors(
    smiles_list: list[str],
    normalize: bool = True,
    show_progress: bool = True,
) -> tuple[np.ndarray, list[int], dict]:
    """
    Compute RDKit descriptors for a list of SMILES.
    
    Args:
        smiles_list: List of SMILES strings
        normalize: If True, standardize descriptors to zero mean and unit variance
        show_progress: Show progress bar
        
    Returns:
        Tuple of (descriptor_matrix, valid_indices, normalization_params)
        - descriptor_matrix: shape (n_valid, n_descriptors)
        - valid_indices: indices of successfully featurized molecules
        - normalization_params: dict with 'mean' and 'std' arrays (empty if not normalized)
    """
    descriptors = []
    valid_indices = []
    
    iterator = tqdm(smiles_list, desc="Computing RDKit descriptors") if show_progress else smiles_list
    
    for idx, smiles in enumerate(iterator):
        desc = compute_rdkit_descriptors(smiles)
        if desc is not None:
            descriptors.append(desc)
            valid_indices.append(idx)
    
    X = np.stack(descriptors, axis=0)
    
    normalization_params = {}
    if normalize:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        # Avoid division by zero
        std = np.where(std == 0, 1.0, std)
        X = (X - mean) / std
        normalization_params = {"mean": mean, "std": std}
    
    return X, valid_indices, normalization_params


def get_descriptor_names() -> list[str]:
    """Return list of RDKit descriptor names."""
    return DESCRIPTOR_NAMES.copy()
