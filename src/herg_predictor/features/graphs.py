"""Graph representations for graph neural networks."""

import numpy as np
import torch
from torch.utils.data import Dataset
from rdkit import Chem


# Atom feature specifications
ATOM_FEATURES = {
    "atomic_num": list(range(1, 119)),  # H to Og
    "degree": [0, 1, 2, 3, 4, 5, 6],
    "formal_charge": [-2, -1, 0, 1, 2, 3],
    "hybridization": [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
    "aromatic": [False, True],
    "num_hs": [0, 1, 2, 3, 4],
}

# Bond feature specifications
BOND_FEATURES = {
    "bond_type": [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    "is_conjugated": [False, True],
    "is_in_ring": [False, True],
    "stereo": [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
    ],
}


def one_hot_encode(value, allowable_set: list) -> list[int]:
    """One-hot encode a value, with unknown values getting all zeros."""
    encoding = [0] * (len(allowable_set) + 1)  # +1 for unknown
    if value in allowable_set:
        encoding[allowable_set.index(value)] = 1
    else:
        encoding[-1] = 1  # Unknown category
    return encoding


def get_atom_features(atom: Chem.Atom) -> np.ndarray:
    """Compute feature vector for an atom."""
    features = []
    
    features.extend(one_hot_encode(atom.GetAtomicNum(), ATOM_FEATURES["atomic_num"]))
    features.extend(one_hot_encode(atom.GetDegree(), ATOM_FEATURES["degree"]))
    features.extend(one_hot_encode(atom.GetFormalCharge(), ATOM_FEATURES["formal_charge"]))
    features.extend(one_hot_encode(atom.GetHybridization(), ATOM_FEATURES["hybridization"]))
    features.extend(one_hot_encode(atom.GetIsAromatic(), ATOM_FEATURES["aromatic"]))
    features.extend(one_hot_encode(atom.GetTotalNumHs(), ATOM_FEATURES["num_hs"]))
    
    return np.array(features, dtype=np.float32)


def get_bond_features(bond: Chem.Bond) -> np.ndarray:
    """Compute feature vector for a bond."""
    features = []
    
    features.extend(one_hot_encode(bond.GetBondType(), BOND_FEATURES["bond_type"]))
    features.extend(one_hot_encode(bond.GetIsConjugated(), BOND_FEATURES["is_conjugated"]))
    features.extend(one_hot_encode(bond.IsInRing(), BOND_FEATURES["is_in_ring"]))
    features.extend(one_hot_encode(bond.GetStereo(), BOND_FEATURES["stereo"]))
    
    return np.array(features, dtype=np.float32)


def mol_to_graph(smiles: str) -> dict | None:
    """
    Convert SMILES to graph representation.
    
    Args:
        smiles: SMILES string
        
    Returns:
        Dictionary with:
        - node_features: (num_atoms, atom_feature_dim) tensor
        - edge_index: (2, num_edges) tensor of edge indices
        - edge_features: (num_edges, bond_feature_dim) tensor
        Returns None if molecule is invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Get atom features
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(get_atom_features(atom))
        node_features = np.stack(atom_features, axis=0)
        
        # Get bond features and edge indices
        # We add edges in both directions for undirected graph
        edge_indices = []
        edge_features = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_feat = get_bond_features(bond)
            
            # Add edge in both directions
            edge_indices.append([i, j])
            edge_indices.append([j, i])
            edge_features.append(bond_feat)
            edge_features.append(bond_feat)
        
        if len(edge_indices) == 0:
            # Single atom molecule
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_attr = np.zeros((0, len(get_bond_features(mol.GetBonds().__iter__().__next__() if mol.GetNumBonds() > 0 else None))), dtype=np.float32)
        else:
            edge_index = np.array(edge_indices, dtype=np.int64).T
            edge_attr = np.stack(edge_features, axis=0)
        
        return {
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_features": edge_attr if len(edge_attr) > 0 else None,
            "num_nodes": len(atom_features),
        }
    except Exception:
        return None


class MoleculeDataset(Dataset):
    """PyTorch Dataset for molecular graphs."""
    
    def __init__(
        self,
        smiles_list: list[str],
        labels: np.ndarray | None = None,
    ):
        """
        Args:
            smiles_list: List of SMILES strings
            labels: Optional array of labels
        """
        self.smiles_list = smiles_list
        self.labels = labels
        
        # Pre-compute graphs and filter invalid molecules
        self.graphs = []
        self.valid_indices = []
        
        for idx, smiles in enumerate(smiles_list):
            graph = mol_to_graph(smiles)
            if graph is not None:
                self.graphs.append(graph)
                self.valid_indices.append(idx)
        
        # Filter labels to match valid molecules
        if labels is not None:
            self.labels = labels[self.valid_indices]
    
    def __len__(self) -> int:
        return len(self.graphs)
    
    def __getitem__(self, idx: int) -> dict:
        graph = self.graphs[idx]
        
        item = {
            "node_features": torch.from_numpy(graph["node_features"]),
            "edge_index": torch.from_numpy(graph["edge_index"]),
            "num_nodes": graph["num_nodes"],
        }
        
        if graph["edge_features"] is not None:
            item["edge_features"] = torch.from_numpy(graph["edge_features"])
        
        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return item


def get_atom_feature_dim() -> int:
    """Return total dimension of atom features."""
    dim = 0
    for values in ATOM_FEATURES.values():
        dim += len(values) + 1  # +1 for unknown category
    return dim


def get_bond_feature_dim() -> int:
    """Return total dimension of bond features."""
    dim = 0
    for values in BOND_FEATURES.values():
        dim += len(values) + 1
    return dim
