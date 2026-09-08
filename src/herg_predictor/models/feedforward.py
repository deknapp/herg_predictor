"""Feed-forward neural network for molecular property prediction."""

# At the top, which is the whole fix. This import did exist -- at the *bottom*
# of the file, under a comment reading "Import numpy for type hints", below
# every class definition. Python evaluates annotations when it defines the
# class, so by the time that line ran the module had already raised NameError
# on `X_train: torch.Tensor | np.ndarray` a hundred lines earlier. The module
# could not be imported at all, and nothing caught it because the only test
# touching this file built a bare nn.Module and never reached the trainer.
# See tests/test_herg_predictor.py::test_feedforward_trains_on_a_learnable_signal.
import numpy as np
import torch
import torch.nn as nn


class FeedForwardNet(nn.Module):
    """
    Multi-layer feed-forward neural network.

    Takes molecular fingerprints or descriptors as input and outputs
    a probability of hERG inhibition.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [512, 256, 128],
        dropout: float = 0.3,
        batch_norm: bool = True,
        activation: str = "relu",
    ):
        """
        Args:
            input_dim: Size of input features
            hidden_dims: List of hidden layer sizes
            dropout: Dropout probability
            batch_norm: Whether to use batch normalization
            activation: Activation function ('relu', 'gelu', 'silu')
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.batch_norm = batch_norm

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "silu":
                layers.append(nn.SiLU())
            else:
                raise ValueError(f"Unknown activation: {activation}")

            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Logits of shape (batch_size, 1)
        """
        h = self.hidden_layers(x)
        return self.output_layer(h)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return probability of positive class."""
        logits = self.forward(x)
        return torch.sigmoid(logits)


class FeedForwardClassifier:
    """
    Wrapper class for training and inference with FeedForwardNet.

    Provides a scikit-learn-like interface.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [512, 256, 128],
        dropout: float = 0.3,
        batch_norm: bool = True,
        activation: str = "relu",
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = FeedForwardNet(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            batch_norm=batch_norm,
            activation=activation,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.learning_rate = learning_rate
        self.history = {"train_loss": [], "val_loss": [], "val_auroc": []}

    def fit(
        self,
        X_train: torch.Tensor | np.ndarray,
        y_train: torch.Tensor | np.ndarray,
        X_val: torch.Tensor | np.ndarray | None = None,
        y_val: torch.Tensor | np.ndarray | None = None,
        epochs: int = 100,
        batch_size: int = 64,
        early_stopping_patience: int = 10,
        class_weight: torch.Tensor | None = None,
        verbose: bool = True,
    ) -> "FeedForwardClassifier":
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            epochs: Maximum number of epochs
            batch_size: Batch size
            early_stopping_patience: Stop after this many epochs without improvement
            class_weight: Optional tensor of class weights [weight_neg, weight_pos]
            verbose: Print training progress
        """
        import numpy as np
        from torch.utils.data import DataLoader, TensorDataset

        # Convert to tensors if needed
        if isinstance(X_train, np.ndarray):
            X_train = torch.from_numpy(X_train).float()
        if isinstance(y_train, np.ndarray):
            y_train = torch.from_numpy(y_train).float()

        train_dataset = TensorDataset(X_train, y_train.unsqueeze(1))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Compute class weights if not provided
        if class_weight is None:
            n_pos = y_train.sum().item()
            n_neg = len(y_train) - n_pos
            pos_weight = torch.tensor([n_neg / n_pos], device=self.device)
        else:
            pos_weight = torch.tensor([class_weight[1] / class_weight[0]], device=self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_losses = []

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                self.optimizer.step()

                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)
            self.history["train_loss"].append(avg_train_loss)

            # Validation
            if X_val is not None and y_val is not None:
                val_loss, val_auroc = self._evaluate(X_val, y_val, criterion)
                self.history["val_loss"].append(val_loss)
                self.history["val_auroc"].append(val_auroc)

                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {avg_train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f}, "
                          f"Val AUROC: {val_auroc:.4f}")

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    def _evaluate(
        self,
        X: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
        criterion: nn.Module,
    ) -> tuple[float, float]:
        """Evaluate on validation set."""
        from sklearn.metrics import roc_auc_score

        self.model.eval()

        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()

        with torch.no_grad():
            X = X.to(self.device)
            y = y.to(self.device)

            logits = self.model(X)
            loss = criterion(logits, y.unsqueeze(1)).item()

            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            y_np = y.cpu().numpy()

            auroc = roc_auc_score(y_np, probs)

        return loss, auroc

    def predict(self, X: torch.Tensor | np.ndarray) -> np.ndarray:
        """Predict class labels."""
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)

    def predict_proba(self, X: torch.Tensor | np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        self.model.eval()

        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()

        with torch.no_grad():
            X = X.to(self.device)
            probs = self.model.predict_proba(X).cpu().numpy().flatten()

        return probs

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
        }, path)

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint["history"]

