#!/usr/bin/env python3
"""
BEST RESULTS:
Epoch 008 | train RMSE: 28.219994 | val RMSE: 24.236523
"""

import math
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src import Forecasting_Dataset_Nddr, get_dataloader_nddr
from src import get_args
# ----------------------------
# Loss
# ----------------------------
class RMSELoss(nn.Module):
    def __init__(self, eps: float = 1e-8, reduction: str = "mean"):
        super().__init__()
        self.eps = eps
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be one of: mean, sum, none")
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred/target shapes must be broadcast-compatible
        mse = (pred - target) ** 2
        if self.reduction == "mean":
            mse = mse.mean()
        elif self.reduction == "sum":
            mse = mse.sum()
        # else "none": keep elementwise
        return torch.sqrt(mse + self.eps)


# ----------------------------
# Small encoders for each input stream
# ----------------------------
class StreamEncoder1D(nn.Module):
    """
    Encodes a 1D vector input x of shape [B, L] into a feature vector [B, D].
    Internally uses Conv1d + AdaptiveAvgPool1d(1) to get a fixed-size embedding.
    """

    def __init__(self, in_channels: int = 1, d: int = 128):
        super().__init__()
        # lightweight conv stack
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, d, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, L]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)  # [B, D]
        return x


# ----------------------------
# NDDR mixing (feature-level)
# ----------------------------
class NDDRMixer(nn.Module):
    """
    NDDR-style mixing across S streams at the feature-vector level.

    Given S feature vectors [B, D] each, we concatenate to [B, S*D],
    then for each stream produce a mixed output via a Linear(S*D -> D).

    Initialization follows the NDDR idea:
      - strong identity on its own stream (alpha)
      - small identity contributions from other streams (beta)
    """

    def __init__(self, num_streams: int, d: int, alpha: float = 0.9):
        super().__init__()
        if num_streams < 2:
            raise ValueError("NDDRMixer needs at least 2 streams.")
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be between 0 and 1.")

        self.num_streams = num_streams
        self.d = d
        self.alpha = alpha
        self.beta = (1.0 - alpha) / (num_streams - 1)

        self.proj = nn.ModuleList([nn.Linear(num_streams * d, d, bias=False) for _ in range(num_streams)])
        self._init_weights()

    def _init_weights(self):
        # Weight shape: [D, S*D]. We set block-diagonal identity-like initialization.
        with torch.no_grad():
            for s, layer in enumerate(self.proj):
                w = torch.zeros(self.d, self.num_streams * self.d)
                for k in range(self.num_streams):
                    start = k * self.d
                    end = (k + 1) * self.d
                    scale = self.alpha if (k == s) else self.beta
                    w[:, start:end] = torch.eye(self.d) * scale
                layer.weight.copy_(w)

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        # feats: list of [B, D]
        if len(feats) != self.num_streams:
            raise ValueError(f"Expected {self.num_streams} streams, got {len(feats)}")
        x = torch.cat(feats, dim=-1)  # [B, S*D]
        out = [layer(x) for layer in self.proj]  # each [B, D]
        return out


# ----------------------------
# Model
# ----------------------------
class NDDRMultiTaskRegressor(nn.Module):
    """
    3-stream encoder + repeated (MLP -> NDDR mix) blocks + multitask regression head.
    """

    def __init__(
        self,
        num_tasks: int = 1155,
        d: int = 128,
        num_nddr_blocks: int = 2,
        dropout: float = 0.1,
        alpha: float = 0.9,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.d = d
        self.num_streams = 3

        # Encoders for the three inputs (L=1, L=1, L=15)
        self.enc1 = StreamEncoder1D(in_channels=1, d=d)
        self.enc2 = StreamEncoder1D(in_channels=1, d=d)
        self.enc3 = StreamEncoder1D(in_channels=1, d=d)

        # Per-stream MLP blocks + NDDR mixing
        self.stream_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, d),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d, d),
                nn.ReLU(),
            )
            for _ in range(self.num_streams)
        ])

        self.nddr_blocks = nn.ModuleList([NDDRMixer(num_streams=self.num_streams, d=d, alpha=alpha)
                                          for _ in range(num_nddr_blocks)])

        # Final multitask head
        self.head = nn.Sequential(
            nn.Linear(self.num_streams * d, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_tasks),
        )

    def forward(self, inputs: Union[List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        if not (isinstance(inputs, (list, tuple)) and len(inputs) == 3):
            raise ValueError("inputs must be a list/tuple of 3 tensors: [x1, x2, x3]")

        x1, x2, x3 = inputs
        x3 = torch.squeeze(x3)
        # Expect shapes: [B,1], [B,1], [B,15]  -> convert to [B,1,L]
        if x1.ndim != 2 or x2.ndim != 2 or x3.ndim != 2:
            raise ValueError("Expected x1,x2,x3 each to have shape [B, L] (2D tensors).")

        x1 = x1.unsqueeze(1)  # [B,1,1]
        x2 = x2.unsqueeze(1)  # [B,1,1]
        x3 = x3.unsqueeze(1)  # [B,1,15]

        f1 = self.enc1(x1)  # [B,D]
        f2 = self.enc2(x2)  # [B,D]
        f3 = self.enc3(x3)  # [B,D]
        feats = [f1, f2, f3]

        for nddr in self.nddr_blocks:
            feats = [mlp(feat) for mlp, feat in zip(self.stream_mlps, feats)]
            feats = nddr(feats)

        fused = torch.cat(feats, dim=-1)   # [B, 3D]
        out = self.head(fused)             # [B, num_tasks]
        return out


# ----------------------------
# Train / Eval helpers
# ----------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion: nn.Module, device: torch.device, num_tasks: int) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        # supports (inputs, y) or (inputs, y, task_ids)
        if len(batch) == 2:
            inputs, y = batch
            task_ids = None
        elif len(batch) == 3:
            inputs, y, task_ids = batch
        else:
            raise ValueError("Batch must be (inputs, y) or (inputs, y, task_ids)")

        inputs = [t.to(device) for t in inputs]
        y = y.to(device)

        pred = model(inputs)

        if task_ids is None:
            # Expect y as [B, T] (or [B, num_tasks])
            if y.ndim == 1:
                y = y.unsqueeze(1)
            if y.shape[-1] != num_tasks:
                raise ValueError(
                    f"When task_ids is not provided, y must have shape [B, {num_tasks}] "
                    f"(got {tuple(y.shape)})."
                )
            loss = criterion(pred, y)
        else:
            # Expect scalar label per sample + task_ids specifying which output dim is supervised
            task_ids = task_ids.to(device).long().view(-1, 1)  # [B,1]
            if y.ndim == 2 and y.shape[1] == 1:
                y_scalar = y
            elif y.ndim == 1:
                y_scalar = y.view(-1, 1)
            else:
                raise ValueError("With task_ids provided, y must be [B] or [B,1].")

            pred_scalar = pred.gather(1, task_ids)  # [B,1]
            loss = criterion(pred_scalar, y_scalar)

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(n_batches, 1)


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_tasks: int,
    grad_clip: Optional[float] = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        if len(batch) == 2:
            inputs, y = batch
            task_ids = None
        elif len(batch) == 3:
            inputs, y, task_ids = batch
        else:
            raise ValueError("Batch must be (inputs, y) or (inputs, y, task_ids)")

        inputs = [t.to(device) for t in inputs]
        y = y.to(device)

        pred = model(inputs)

        if task_ids is None:
            if y.ndim == 1:
                y = y.unsqueeze(1)
            if y.shape[-1] != num_tasks:
                raise ValueError(
                    f"When task_ids is not provided, y must have shape [B, {num_tasks}] "
                    f"(got {tuple(y.shape)})."
                )
            loss = criterion(pred, y)
        else:
            task_ids = task_ids.to(device).long().view(-1, 1)
            if y.ndim == 2 and y.shape[1] == 1:
                y_scalar = y
            elif y.ndim == 1:
                y_scalar = y.view(-1, 1)
            else:
                raise ValueError("With task_ids provided, y must be [B] or [B,1].")

            pred_scalar = pred.gather(1, task_ids)
            loss = criterion(pred_scalar, y_scalar)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_tasks", type=int, default=1155)
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--num_nddr_blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NDDRMultiTaskRegressor(
        num_tasks=args.num_tasks,
        d=args.d,
        num_nddr_blocks=args.num_nddr_blocks,
        dropout=args.dropout,
        alpha=args.alpha,
    ).to(device)

    criterion = RMSELoss(eps=1e-8, reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    config = get_args()

    train_loader, val_loader, data_info, train_data, test_data = get_dataloader_nddr(config=config, year=1,
                                                                                      matches=None)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_tasks=args.num_tasks,
            grad_clip=args.grad_clip,
        )

        if val_loader is not None:
            val_loss = evaluate(model, val_loader, criterion, device, args.num_tasks)
            print(f"Epoch {epoch:03d} | train RMSE: {train_loss:.6f} | val RMSE: {val_loss:.6f}")
        else:
            print(f"Epoch {epoch:03d} | train RMSE: {train_loss:.6f}")


if __name__ == "__main__":
    main()
