"""Preprocess and clean hERG bioactivity data."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def standardize_smiles(smiles: str) -> str | None:
    """
    Standardize a SMILES string: remove salts, neutralize, canonicalize.
    
    Args:
        smiles: Input SMILES string
        
    Returns:
        Standardized canonical SMILES, or None if invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Remove salts (keep largest fragment)
        mol = rdMolStandardize.FragmentParent(mol)
        
        # Neutralize charges where possible
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)
        
        # Return canonical SMILES
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def preprocess_herg_data(
    input_path: str | Path = "data/raw/herg_chembl.csv",
    output_path: str | Path = "data/processed/herg_clean.parquet",
    activity_threshold_nm: float = 10000,  # 10 μM in nM
    exact_only: bool = True,
) -> pd.DataFrame:
    """
    Clean and preprocess hERG data for modeling.
    
    Args:
        input_path: Path to raw ChEMBL data
        output_path: Where to save processed data
        activity_threshold_nm: IC50 threshold for binary classification (in nM)
        exact_only: If True, only keep exact measurements (relation = '=')
        
    Returns:
        Cleaned DataFrame with binary labels
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading raw data from {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} records")
    
    # Step 1: Filter for exact measurements if requested
    if exact_only:
        df = df[df["activity_relation"] == "="]
        logger.info(f"After filtering for exact measurements: {len(df)} records")
    
    # Step 2: Convert units to nM
    # Most ChEMBL IC50 values are in nM, but verify
    df = df[df["activity_units"] == "nM"].copy()
    logger.info(f"After filtering for nM units: {len(df)} records")
    
    # Step 3: Standardize SMILES
    logger.info("Standardizing SMILES...")
    df["smiles_standard"] = df["smiles"].apply(standardize_smiles)
    n_failed = df["smiles_standard"].isna().sum()
    logger.info(f"Failed to standardize {n_failed} SMILES")
    df = df.dropna(subset=["smiles_standard"])
    
    # Step 4: Handle duplicates - take median activity for same compound
    logger.info("Aggregating duplicate compounds...")
    df_agg = (
        df.groupby("smiles_standard")
        .agg({
            "activity_value": "median",
            "molecule_chembl_id": "first",
            "activity_type": lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
        })
        .reset_index()
    )
    logger.info(f"After deduplication: {len(df_agg)} unique compounds")
    
    # Step 5: Create binary labels
    # IC50 < threshold means inhibitor (positive class = 1)
    df_agg["label"] = (df_agg["activity_value"] < activity_threshold_nm).astype(int)
    df_agg["pIC50"] = -np.log10(df_agg["activity_value"] * 1e-9)  # Convert to pIC50
    
    logger.info(f"Class distribution (threshold={activity_threshold_nm} nM):")
    logger.info(f"  Inhibitors (label=1): {df_agg['label'].sum()}")
    logger.info(f"  Non-inhibitors (label=0): {(df_agg['label'] == 0).sum()}")
    
    # Step 6: Rename and select final columns
    df_final = df_agg.rename(columns={"smiles_standard": "smiles"})
    df_final = df_final[["smiles", "molecule_chembl_id", "activity_value", "pIC50", "label"]]
    
    # Save
    df_final.to_parquet(output_path, index=False)
    logger.info(f"Saved processed data to {output_path}")
    
    return df_final


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess hERG data")
    parser.add_argument("--input", type=str, default="data/raw/herg_chembl.csv")
    parser.add_argument("--output", type=str, default="data/processed/herg_clean.parquet")
    parser.add_argument("--threshold", type=float, default=10000, help="IC50 threshold in nM")
    args = parser.parse_args()
    
    preprocess_herg_data(
        input_path=args.input,
        output_path=args.output,
        activity_threshold_nm=args.threshold,
    )
