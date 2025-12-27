"""Graph Neural Network models for molecular property prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.data import Data, Batch


class MPNNLayer(MessagePassing):
    """
    Message Passing Neural Network layer.
    
    Implements the message passing scheme from Gilmer et al. (2017)
    "Neural Message Passing for Quantum Chemistry"
    """
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int | None = None,
        hidden_dim: int = 128,
        aggr: str = "mean",
    ):
        super().__init__(aggr=aggr)
        
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        
        # Message function: combines source node and edge features
        if edge_dim is not None:
            self.message_mlp = nn.Sequential(
                nn.Linear(node_dim + edge_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.message_mlp = nn.Sequential(
                nn.Linear(node_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        
        # Update function: combines node with aggregated messages
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.edge_dim = edge_dim
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features (num_nodes, node_dim)
            edge_index: Edge indices (2, num_edges)
            edge_attr: Edge features (num_edges, edge_dim), optional
            
        Returns:
            Updated node features (num_nodes, hidden_dim)
        """
        # Propagate messages
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        # Update nodes
        out = self.update_mlp(torch.cat([x, out], dim=-1))
        
        return out
    
    def message(
        self,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute messages from source nodes."""
        if edge_attr is not None and self.edge_dim is not None:
            return self.message_mlp(torch.cat([x_j, edge_attr], dim=-1))
        else:
            return self.message_mlp(x_j)


class MPNN(nn.Module):
    """
    Message Passing Neural Network for molecular graphs.
    
    Architecture:
    1. Initial node embedding
    2. Multiple MPNN layers with residual connections
    3. Global pooling (readout)
    4. Final MLP for prediction
    """
    
    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: int | None = None,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        aggregation: str = "mean",
        readout: str = "mean",
    ):
        """
        Args:
            node_input_dim: Dimension of input node features
            edge_input_dim: Dimension of input edge features (optional)
            hidden_dim: Hidden dimension size
            num_layers: Number of message passing layers
            dropout: Dropout probability
            aggregation: Aggregation method for messages ('mean', 'sum', 'max')
            readout: Global pooling method ('mean', 'sum', 'max')
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Initial node embedding
        self.node_embedding = nn.Linear(node_input_dim, hidden_dim)
        
        # Edge embedding (if edge features provided)
        if edge_input_dim is not None:
            self.edge_embedding = nn.Linear(edge_input_dim, hidden_dim)
            edge_dim = hidden_dim
        else:
            self.edge_embedding = None
            edge_dim = None
        
        # Message passing layers
        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.conv_layers.append(
                MPNNLayer(
                    node_dim=hidden_dim,
                    edge_dim=edge_dim,
                    hidden_dim=hidden_dim,
                    aggr=aggregation,
                )
            )
        
        # Batch normalization
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        
        # Readout function
        if readout == "mean":
            self.readout = global_mean_pool
        elif readout == "sum":
            self.readout = global_add_pool
        elif readout == "max":
            self.readout = global_max_pool
        else:
            raise ValueError(f"Unknown readout: {readout}")
        
        # Output MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features (num_nodes, node_input_dim)
            edge_index: Edge indices (2, num_edges)
            edge_attr: Edge features (num_edges, edge_input_dim), optional
            batch: Batch assignment for each node (num_nodes,)
            
        Returns:
            Logits of shape (batch_size, 1)
        """
        # Initial embeddings
        h = self.node_embedding(x)
        
        if edge_attr is not None and self.edge_embedding is not None:
            edge_attr = self.edge_embedding(edge_attr)
        
        # Message passing with residual connections
        for conv, bn in zip(self.conv_layers, self.batch_norms):
            h_new = conv(h, edge_index, edge_attr)
            h_new = bn(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h + h_new  # Residual connection
        
        # Global pooling
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        h_graph = self.readout(h, batch)
        
        # Output
        return self.output_mlp(h_graph)
    
    def predict_proba(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return probability of positive class."""
        logits = self.forward(x, edge_index, edge_attr, batch)
        return torch.sigmoid(logits)


class GNNClassifier:
    """
    Wrapper class for training and inference with GNN.
    
    Provides a scikit-learn-like interface and handles batching.
    """
    
    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: int | None = None,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        aggregation: str = "mean",
        readout: str = "mean",
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = MPNN(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            aggregation=aggregation,
            readout=readout,
        ).to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        self.history = {"train_loss": [], "val_loss": [], "val_auroc": []}
    
    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        pos_weight: float | None = None,
        verbose: bool = True,
    ) -> "GNNClassifier":
        """
        Train the model.
        
        Args:
            train_loader: PyTorch Geometric DataLoader for training
            val_loader: PyTorch Geometric DataLoader for validation
            epochs: Maximum number of epochs
            early_stopping_patience: Stop after this many epochs without improvement
            pos_weight: Weight for positive class in loss function
            verbose: Print training progress
        """
        import numpy as np
        from sklearn.metrics import roc_auc_score
        
        if pos_weight is not None:
            pos_weight_tensor = torch.tensor([pos_weight], device=self.device)
        else:
            pos_weight_tensor = None
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_losses = []
            
            for batch in train_loader:
                batch = batch.to(self.device)
                
                self.optimizer.zero_grad()
                logits = self.model(
                    batch.x,
                    batch.edge_index,
                    getattr(batch, "edge_attr", None),
                    batch.batch,
                )
                loss = criterion(logits.squeeze(), batch.y.float())
                loss.backward()
                self.optimizer.step()
                
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            self.history["train_loss"].append(avg_train_loss)
            
            # Validation
            if val_loader is not None:
                val_loss, val_auroc = self._evaluate(val_loader, criterion)
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
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
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
    
    def _evaluate(self, loader, criterion) -> tuple[float, float]:
        """Evaluate on validation set."""
        import numpy as np
        from sklearn.metrics import roc_auc_score
        
        self.model.eval()
        losses = []
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                logits = self.model(
                    batch.x,
                    batch.edge_index,
                    getattr(batch, "edge_attr", None),
                    batch.batch,
                )
                loss = criterion(logits.squeeze(), batch.y.float())
                losses.append(loss.item())
                
                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs.flatten())
                all_labels.extend(batch.y.cpu().numpy())
        
        auroc = roc_auc_score(all_labels, all_preds)
        return np.mean(losses), auroc
    
    def predict_proba(self, loader) -> np.ndarray:
        """Predict probabilities for all samples in loader."""
        import numpy as np
        
        self.model.eval()
        all_preds = []
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                probs = self.model.predict_proba(
                    batch.x,
                    batch.edge_index,
                    getattr(batch, "edge_attr", None),
                    batch.batch,
                )
                all_preds.extend(probs.cpu().numpy().flatten())
        
        return np.array(all_preds)
    
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


import numpy as np
