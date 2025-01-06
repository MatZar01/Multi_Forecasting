import torch
from torch import nn


class MLP_base(nn.Module):
    def __init__(self, sample_input, store_size, sku_size, embedding_dim):
        super().__init__()

        self.store_embedder = nn.Embedding(store_size, embedding_dim)
        self.sku_embedder = nn.Embedding(sku_size, embedding_dim)

        input_dim = sample_input[2].shape[0] + embedding_dim + embedding_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, store_in, sku_in, feature_vector):
        store_embedding = self.store_embedder(store_in).squeeze(1)
        sku_embedding = self.sku_embedder(sku_in).squeeze(1)

        concatenated = torch.concatenate([store_embedding, sku_embedding, feature_vector], dim=-1)

        logits = self.model(concatenated)

        return logits
