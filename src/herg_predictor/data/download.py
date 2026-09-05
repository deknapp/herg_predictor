"""Download hERG bioactivity data from ChEMBL."""

import logging
from pathlib import Path

import pandas as pd

# chembl_webresource_client builds its client by fetching the API schema at
# IMPORT time, so importing it at module scope makes this whole package
# unimportable whenever ChEMBL is unreachable -- including the parts that need
# no network at all. It is imported inside download_herg_data() instead.

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_herg_data(
    target_chembl_id: str = "CHEMBL240",
    activity_types: list[str] = ["IC50", "Ki"],
    output_path: str | Path = "data/raw/herg_chembl.csv",
) -> pd.DataFrame:
    """
    Download hERG bioactivity data from ChEMBL.
    
    Args:
        target_chembl_id: ChEMBL target ID for hERG (CHEMBL240)
        activity_types: Activity measurement types to include
        output_path: Where to save the raw data
        
    Returns:
        DataFrame with columns: molecule_chembl_id, smiles, activity_type, 
        activity_value, activity_units, assay_chembl_id
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Fetching hERG data from ChEMBL (target: {target_chembl_id})")

    try:
        from chembl_webresource_client.new_client import new_client
    except Exception as exc:  # noqa: BLE001 - the client raises bare Exception
        raise ConnectionError(
            "Could not reach the ChEMBL API. The client fetches its schema on "
            "import, so this fails before any query is sent. Check "
            "https://www.ebi.ac.uk/chembl/api/data/status.json and retry."
        ) from exc

    activity = new_client.activity
    
    # Query for hERG activities
    results = activity.filter(
        target_chembl_id=target_chembl_id,
        standard_type__in=activity_types,
    ).only([
        "molecule_chembl_id",
        "canonical_smiles", 
        "standard_type",
        "standard_value",
        "standard_units",
        "standard_relation",
        "assay_chembl_id",
        "assay_type",
        "pchembl_value",
    ])
    
    # Convert to dataframe
    logger.info("Converting results to DataFrame...")
    records = list(results)
    logger.info(f"Retrieved {len(records)} activity records")
    
    df = pd.DataFrame(records)
    
    # Rename columns for clarity
    df = df.rename(columns={
        "canonical_smiles": "smiles",
        "standard_type": "activity_type",
        "standard_value": "activity_value",
        "standard_units": "activity_units",
        "standard_relation": "activity_relation",
    })
    
    # Basic filtering
    df = df.dropna(subset=["smiles", "activity_value"])
    
    logger.info(f"After removing nulls: {len(df)} records")
    logger.info(f"Activity types: {df['activity_type'].value_counts().to_dict()}")
    
    # Save raw data
    df.to_csv(output_path, index=False)
    logger.info(f"Saved raw data to {output_path}")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download hERG data from ChEMBL")
    parser.add_argument("--output", type=str, default="data/raw/herg_chembl.csv")
    args = parser.parse_args()
    
    download_herg_data(output_path=args.output)
