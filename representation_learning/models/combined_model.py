import torch
import torch.nn as nn
from .transformer import TransformerModel, create_transformer
from .distillation import DistillationModel

class CombinedModel(nn.Module):
    def __init__(self, distillation_model: DistillationModel, transformer_model: TransformerModel):
        super().__init__()
        self.distillation_model = distillation_model
        self.transformer_output_proj = transformer_model.output_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        distilled_output = self.distillation_model(x)
        final_output = self.transformer_output_proj(distilled_output)
        return final_output

    @classmethod
    def from_pretrained(cls, config: dict, device: torch.device) -> 'CombinedModel':
        # Load the distillation model
        distillation_model = DistillationModel(
            input_dim=config['model']['input_dim'],
            output_dim=config['model']['d_model']
        )
        distillation_model.load_state_dict(
            torch.load(config['model']['distillation_path'], map_location=device)
        )
        
        # Load the transformer model
        transformer_model = create_transformer(**config['model'])
        transformer_model.load_state_dict(
            torch.load(config['model']['transformer_path'], map_location=device)
        )
        
        return cls(distillation_model, transformer_model) 