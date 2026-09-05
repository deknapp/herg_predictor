"""Data loading, cleaning and splitting.

Imported lazily. `download` pulls in chembl_webresource_client, which fetches
the ChEMBL API schema at import time -- so importing it eagerly made
`preprocess` and `splits` unavailable whenever ChEMBL was down, even though
neither touches the network.
"""

__all__ = [
    "download_herg_data",
    "preprocess_herg_data",
    "standardize_smiles",
    "scaffold_split",
    "random_split",
    "get_split",
]

_MODULES = {
    "download_herg_data": "download",
    "preprocess_herg_data": "preprocess",
    "standardize_smiles": "preprocess",
    "scaffold_split": "splits",
    "random_split": "splits",
    "get_split": "splits",
}


def __getattr__(name: str):
    module = _MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f".{module}", __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)
