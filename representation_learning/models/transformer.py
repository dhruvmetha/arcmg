 # Copy existing code from experiments/representation_learning/transformer.py
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class TransformerModel(nn.Module):
    def __init__(self, input_dim, latent_dim, nhead, num_layers, dim_feedforward, use_positional_encoding=False, batch_first=True):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.pos_encoder = PositionalEncoding(latent_dim)
        encoder_layers = nn.TransformerEncoderLayer(latent_dim, nhead, dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.output_proj = nn.Sequential(nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, input_dim))

    def forward(self, src, src_key_padding_mask=None):
        src = self.input_proj(src)
        if self.use_positional_encoding:
            src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        return self.output_proj(output)

    def get_representation(self, src, src_mask=None):
        src = self.input_proj(src)
        if self.use_positional_encoding:
            src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=src_mask)
        return output

def create_transformer(**kwargs):
    default_config = {
        'input_dim': 2,
        'latent_dim': 64,
        'nhead': 4,
        'num_layers': 3,
        'dim_feedforward': 256,
        'use_positional_encoding': False
    }
    config = {**default_config, **kwargs}

    return TransformerModel(**config)