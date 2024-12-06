import torch
import torch.nn as nn

class DistillationModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
def create_distillation_model(**kwargs):
    default_config = {
        'input_dim': 2,
        'output_dim': 2
    }
    config = {**default_config, **kwargs}

    return DistillationModel(**config)