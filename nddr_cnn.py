#!/usr/bin/env python3
"""
DS1:
Epoch 010 | train RMSELoss=28.594654 | val RMSE=42.966392 | val min_task_RMSE=3.226013 | val max_task_RMSE=414.038484
DS2:
Epoch 010 | train RMSELoss=6.962429 | val RMSE=22.333512 | val min_task_RMSE=4.978598 | val max_task_RMSE=45.025564
DS3:
Epoch 004 | train RMSELoss=9031.883219 | val RMSE=24656.353769 | val min_task_RMSE=6.128714 | val max_task_RMSE=118891.334322
"""

from __future__ import annotations

import math
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src import Forecasting_Dataset_Nddr, get_dataloader_nddr
from src import get_args
# -----------------------------
# Loss
# -----------------------------
class RMSELoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred/target: (B, 1) or (B,)
        mse = F.mse_loss(pred, target, reduction="mean")
        return torch.sqrt(mse + self.eps)


# -----------------------------
# Task-specific linear head (one linear per task, implemented efficiently)
# -----------------------------
class TaskLinear(nn.Module):
    """
    Per-task linear layer:
      y = x @ W_task + b_task
    Implemented as embeddings:
      W: (num_tasks, in_dim)
      b: (num_tasks,)
    """
    def __init__(self, in_dim: int, num_tasks: int):
        super().__init__()
        self.in_dim = in_dim
        self.num_tasks = num_tasks
        self.weight = nn.Embedding(num_tasks, in_dim)
        self.bias = nn.Embedding(num_tasks, 1)

        # Init: small weights, zero bias
        nn.init.normal_(self.weight.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.bias.weight)

    def forward(self, x: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_dim)
        task_id: (B,) long
        returns: (B, 1)
        """
        w = self.weight(task_id)          # (B, in_dim)
        b = self.bias(task_id)            # (B, 1)
        y = (x * w).sum(dim=1, keepdim=True) + b
        return y


# -----------------------------
# NDDR block (1D)
# -----------------------------
class NDDRBlock1D(nn.Module):
    """
    Classic NDDR idea:
      [Fs', Ft'] = Conv1x1( concat(Fs, Ft) )
    where Fs and Ft have same channel count.
    """
    def __init__(self, channels: int, init_identity: bool = True):
        super().__init__()
        self.channels = channels
        self.mix = nn.Conv1d(2 * channels, 2 * channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(2 * channels)

        if init_identity:
            self._init_as_block_identity()

    def _init_as_block_identity(self):
        # Initialize mix weights to (approximately) identity:
        # output_shared <- input_shared
        # output_task   <- input_task
        with torch.no_grad():
            self.mix.weight.zero_()
            # weight shape: (out_ch, in_ch, 1)
            for i in range(self.channels):
                self.mix.weight[i, i, 0] = 1.0  # shared->shared
                self.mix.weight[self.channels + i, self.channels + i, 0] = 1.0  # task->task
            # small noise can help symmetry breaking (optional)
            self.mix.weight.add_(0.001 * torch.randn_like(self.mix.weight))

    def forward(self, fs: torch.Tensor, ft: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # fs, ft: (B, C, L)
        x = torch.cat([fs, ft], dim=1)          # (B, 2C, L)
        x = self.bn(self.mix(x))                # (B, 2C, L)
        x = F.relu(x, inplace=True)
        fs2, ft2 = x[:, : self.channels], x[:, self.channels :]
        return fs2, ft2


# -----------------------------
# Small Conv block (1D)
# -----------------------------
class ConvBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: int = 1, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=p, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)
        return x


# -----------------------------
# NDDR-CNN for your input format
# -----------------------------
class NDDRCNNMultitaskRegressor(nn.Module):
    def __init__(
        self,
        num_tasks: int = 1155,
        in_len: int = 17,           # x1(1) + x2(1) + x3(15)
        base_ch: int = 64,
        num_stages: int = 3,
        dropout: float = 0.1,
        head_hidden: int = 128,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.in_len = in_len

        # Two parallel streams: shared and task streams
        self.shared_stem = ConvBlock1D(1, base_ch, k=3, p=1, dropout=dropout)
        self.task_stem = ConvBlock1D(1, base_ch, k=3, p=1, dropout=dropout)

        shared_blocks = []
        task_blocks = []
        nddr_blocks = []

        ch = base_ch
        for _ in range(num_stages):
            shared_blocks.append(ConvBlock1D(ch, ch, k=3, p=1, dropout=dropout))
            task_blocks.append(ConvBlock1D(ch, ch, k=3, p=1, dropout=dropout))
            nddr_blocks.append(NDDRBlock1D(ch, init_identity=True))

        self.shared_blocks = nn.ModuleList(shared_blocks)
        self.task_blocks = nn.ModuleList(task_blocks)
        self.nddr_blocks = nn.ModuleList(nddr_blocks)

        self.pool = nn.AdaptiveAvgPool1d(1)

        # Fusion MLP before task-specific head
        self.fuse = nn.Sequential(
            nn.Linear(2 * ch, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.head = TaskLinear(head_hidden, num_tasks=num_tasks)

        # (Optional) better init for fuse
        for m in self.fuse:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    @staticmethod
    def _pack_inputs(inputs_list: List[torch.Tensor]) -> torch.Tensor:
        """
        inputs_list: [x1(B,1), x2(B,1), x3(B,15)]
        returns: x(B,1,17)
        """
        x1, x2, x3 = inputs_list
        x3 = torch.squeeze(x3)
        # Ensure float dtype
        x1 = x1.float()
        x2 = x2.float()
        x3 = x3.float()

        x = torch.cat([x1, x2, x3], dim=1)      # (B, 17)
        x = x.unsqueeze(1)                      # (B, 1, 17)  -> (B, C_in=1, L=17)
        return x

    def forward(self, inputs_list: List[torch.Tensor], task_id: torch.Tensor) -> torch.Tensor:
        """
        task_id: (B,) or (B,1) long
        returns: (B,1)
        """
        if task_id.dim() == 2:
            task_id = task_id.squeeze(1)
        task_id = task_id.long()

        x = self._pack_inputs(inputs_list)      # (B,1,17)

        fs = self.shared_stem(x)
        ft = self.task_stem(x)

        for sb, tb, nb in zip(self.shared_blocks, self.task_blocks, self.nddr_blocks):
            fs = sb(fs)
            ft = tb(ft)
            fs, ft = nb(fs, ft)

        fs = self.pool(fs).squeeze(-1)          # (B, ch)
        ft = self.pool(ft).squeeze(-1)          # (B, ch)

        fused = torch.cat([fs, ft], dim=1)      # (B, 2ch)
        fused = self.fuse(fused)                # (B, head_hidden)

        yhat = self.head(fused, task_id)        # (B, 1)
        return yhat


# -----------------------------
# Minimal training / evaluation utilities
# -----------------------------
@torch.no_grad()
def evaluate_rmse(model: torch.nn.Module, loader, device: torch.device):
    """
    Computes:
      - overall RMSE across ALL val samples (tasks combined)
      - per-task RMSE (only for tasks that appear in val loader)
      - min and max per-task RMSE across tasks (based on tasks present)

    Returns:
      overall_rmse: float
      min_task_rmse: float
      max_task_rmse: float
      per_task_rmse: dict[int,float]  (optional, useful for debugging)
    """
    model.eval()

    # Overall
    se_sum_all = 0.0
    n_all = 0

    # Per-task accumulators
    # task_id -> (se_sum, n)
    task_se = {}
    task_n = {}

    for inputs_list, y, task_id in loader:
        inputs_list = [t.to(device) for t in inputs_list]
        y = y.to(device).float()

        if task_id.dim() == 2:
            task_id = task_id.squeeze(1)
        task_id = task_id.to(device).long()

        pred = model(inputs_list, task_id).float()
        err = (pred - y).view(-1)          # (B,)
        tid = task_id.view(-1)             # (B,)

        # Overall
        se = (err * err)
        se_sum_all += se.sum().item()
        n_all += err.numel()

        # Per-task (loop is ok; val typically smaller. If you want, we can vectorize.)
        for t, e2 in zip(tid.tolist(), se.tolist()):
            task_se[t] = task_se.get(t, 0.0) + e2
            task_n[t] = task_n.get(t, 0) + 1

    overall_rmse = math.sqrt(se_sum_all / max(n_all, 1))

    # Per-task RMSE for tasks that appear in val
    per_task_rmse = {}
    for t in task_se:
        per_task_rmse[t] = math.sqrt(task_se[t] / max(task_n[t], 1))

    if len(per_task_rmse) == 0:
        # No samples in loader
        return overall_rmse, float("nan"), float("nan"), per_task_rmse

    min_task_rmse = min(per_task_rmse.values())
    max_task_rmse = max(per_task_rmse.values())

    return overall_rmse, min_task_rmse, max_task_rmse, per_task_rmse


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    grad_clip: Optional[float] = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0

    for inputs_list, y, task_id in loader:
        inputs_list = [t.to(device) for t in inputs_list]
        y = y.to(device).float()
        task_id = task_id.to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(inputs_list, task_id)
        loss = loss_fn(pred, y)
        loss.backward()

        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_tasks", type=int, default=93)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--base_ch", type=int, default=64)
    parser.add_argument("--num_stages", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--head_hidden", type=int, default=128)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--use_dummy_data", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    model = NDDRCNNMultitaskRegressor(
        num_tasks=args.num_tasks,
        in_len=17,
        base_ch=args.base_ch,
        num_stages=args.num_stages,
        dropout=args.dropout,
        head_hidden=args.head_hidden,
    ).to(device)

    loss_fn = RMSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    config = get_args()

    train_loader, val_loader, data_info, train_data, test_data = get_dataloader_nddr(config=config, year=1, matches=None)


    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            grad_clip=args.grad_clip,
        )

        val_rmse, val_min_rmse, val_max_rmse, _ = evaluate_rmse(model, val_loader, device)

        print(
            f"Epoch {epoch:03d} | "
            f"train RMSELoss={train_loss:.6f} | "
            f"val RMSE={val_rmse:.6f} | "
            f"val min_task_RMSE={val_min_rmse:.6f} | "
            f"val max_task_RMSE={val_max_rmse:.6f}"
        )

    print("Done.")


if __name__ == "__main__":
    main()
