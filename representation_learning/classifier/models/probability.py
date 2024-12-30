import torch
import torch.nn as nn
from typing import Dict

class ProbabilityClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [128, 64],
        dropout: float = 0.1
    ):
        """
        Classifier to determine probability of reaching attractor.
        
        Args:
            input_dim: Dimension of input features (raw coords or representations)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 3))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor containing concatenated source and target features
               Shape: [batch_size, input_dim]
        
        Returns:
            Probability of reachability [batch_size, 1]
        """
        return self.network(x)

def create_probability_classifier(config: Dict) -> ProbabilityClassifier:
    """Create classifier from config."""
    return ProbabilityClassifier(
        input_dim=config['input_dim'],
        hidden_dims=config.get('hidden_dims', [128, 64]),
        dropout=config.get('dropout', 0.1)
    )