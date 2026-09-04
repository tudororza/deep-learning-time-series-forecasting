"""Compact residual LSTM used by both project experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    input_size: int
    future_size: int
    n_series: int
    output_size: int
    hidden_size: int = 64
    embedding_dim: int = 8
    dropout: float = 0.1


class ResidualLSTM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.series_embedding = nn.Embedding(config.n_series, config.embedding_dim)
        decoder_input = config.hidden_size + config.embedding_dim + config.future_size
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.output_size),
        )

    def forward(
        self,
        history: torch.Tensor,
        future_covariates: torch.Tensor,
        series_index: torch.Tensor,
        baseline: torch.Tensor,
        residual_scale: torch.Tensor,
    ) -> torch.Tensor:
        _, (hidden, _) = self.encoder(history)
        context = self.layer_norm(hidden[-1])
        horizon = baseline.shape[1]
        repeated_context = context.unsqueeze(1).expand(-1, horizon, -1)
        embedding = self.series_embedding(series_index).unsqueeze(1).expand(-1, horizon, -1)
        decoder_input = torch.cat([repeated_context, embedding, future_covariates], dim=-1)
        normalized_residual = self.decoder(decoder_input)
        return baseline + normalized_residual * residual_scale.unsqueeze(1)

    def config_dict(self) -> dict[str, int | float]:
        return asdict(self.config)

