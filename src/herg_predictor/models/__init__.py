"""Model architectures for hERG prediction.

These are imported lazily. The Random Forest baseline needs only scikit-learn,
while the feedforward net and the GNN pull in torch (and torch-geometric), and
XGBoost links against a system OpenMP runtime. Importing all of them eagerly
meant that a missing optional dependency broke `import herg_predictor.models`
entirely -- including for someone who only wanted the sklearn baseline.
"""

__all__ = [
    "RandomForestModel",
    "XGBoostModel",
    "FeedForwardNet",
    "FeedForwardClassifier",
    "MPNN",
    "GNNClassifier",
]

_MODULES = {
    "RandomForestModel": "baseline",
    "XGBoostModel": "baseline",
    "FeedForwardNet": "feedforward",
    "FeedForwardClassifier": "feedforward",
    "MPNN": "gnn",
    "GNNClassifier": "gnn",
}


def __getattr__(name: str):
    """PEP 562 lazy attribute access: import the submodule on first use."""
    module = _MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f".{module}", __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)
