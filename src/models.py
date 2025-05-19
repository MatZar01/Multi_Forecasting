import torch
from torch import nn
from copy import deepcopy


class MLP_base(nn.Module):
    def __init__(self, sample_input, store_size, sku_size, embedding_dim, device):
        super().__init__()
        self.device = device

        self.store_embedder = nn.Embedding(store_size, embedding_dim)
        self.sku_embedder = nn.Embedding(sku_size, embedding_dim)

        input_dim = sample_input[2].nelement() + \
                    sample_input[0].nelement() * embedding_dim + \
                    sample_input[1].nelement() * embedding_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5)
        )

        self.heads = nn.ModuleDict()

    def forward(self, store_in, sku_in, feature_vector, task):
        store_embedding = self.store_embedder(store_in)
        sku_embedding = self.sku_embedder(sku_in)

        concatenated = torch.concatenate([store_embedding.squeeze(), sku_embedding.squeeze(), feature_vector.squeeze()], dim=1)

        features = self.model(concatenated)

        logits = self.heads[str(task)](features)

        return logits

    def add_head(self, task):
        if task == -1:
            self.heads[str(task)] = nn.Linear(64, 1).to(self.device)
        else:
            if str(task) not in self.heads.keys():
                new_head = deepcopy(self.heads[str(-1)])
                self.heads[str(task)] = new_head

    def freeze_model_layers(self):
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.training = False
        self.sku_embedder.requires_grad = False
        self.store_embedder.requires_grad = False
