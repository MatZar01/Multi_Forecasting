#!/usr/bin/env python3
"""
BEST RESULTS:
DS1:
epoch 009 | train RMSE 28.825640 | val RMSE 43.759022 | val task RMSE min 3.156615 | val task RMSE max 414.138153 | best 43.759022
DS2:
epoch 002 | train RMSE 7.393369 | val RMSE 11.302721 | val task RMSE min 4.173947 | val task RMSE max 22.889055 | best 11.302721
DS3:
epoch 004 | train RMSE 8850.887728 | val RMSE 22823.150391 | val task RMSE min 9.824359 | val task RMSE max 103518.203125 | best 22823.150391
"""

from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src import Forecasting_Dataset_Cross, get_dataloader_cross
from src import get_args
# -------------------------
# Utilities
# -------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RMSELoss(nn.Module):
    def __init__(self, eps: float = 1e-8, reduction: str = "mean"):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Invalid reduction: {reduction}")
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred/target must be broadcastable to same shape
        mse = (pred - target) ** 2
        if self.reduction == "mean":
            mse = mse.mean()
        elif self.reduction == "sum":
            mse = mse.sum()
        # else "none": keep elementwise
        return torch.sqrt(mse + self.eps)


# -------------------------
# Low-rank Cross-Stitch Unit
# -------------------------

class LowRankCrossStitch(nn.Module):
    """
    Mix activations across tasks using A = I + U V^T (rank r).
    Input:  x of shape [B, T, H]
    Output: y of shape [B, T, H]
    """

    def __init__(self, num_tasks: int, rank: int = 8, init_scale: float = 1e-3):
        super().__init__()
        self.num_tasks = num_tasks
        self.rank = rank

        # U, V initialized small so A ≈ I at start (stable).
        self.U = nn.Parameter(init_scale * torch.randn(num_tasks, rank))
        self.V = nn.Parameter(init_scale * torch.randn(num_tasks, rank))

        # Optional learnable per-task identity scaling (kept near 1.0)
        self.log_diag = nn.Parameter(torch.zeros(num_tasks))  # diag = exp(log_diag)

    def forward(self, x: torch.Tensor, task_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B, T, H] (if task_idx is None)
           [B, A, H] where A=len(task_idx) (if task_idx provided as unique active tasks)
        task_idx: optional 1D LongTensor listing which tasks are present in x's task dimension.
        """
        if x.ndim != 3:
            raise ValueError(f"Expected x [B,T,H], got shape {tuple(x.shape)}")

        B, T_eff, H = x.shape

        if task_idx is None:
            U = self.U                      # [T, r]
            V = self.V                      # [T, r]
            diag = torch.exp(self.log_diag)  # [T]
        else:
            # task_idx is unique active tasks (shape [A])
            U = self.U.index_select(0, task_idx)                      # [A, r]
            V = self.V.index_select(0, task_idx)                      # [A, r]
            diag = torch.exp(self.log_diag.index_select(0, task_idx)) # [A]

        # Low-rank mixing:
        # s_r = sum_j V[j,r] * x[:,j,:]      => s: [B, r, H]
        s = torch.einsum("bth,tr->brh", x, V)

        # mix_i = sum_r U[i,r] * s_r         => mix: [B, T_eff, H]
        mix = torch.einsum("tr,brh->bth", U, s)

        # Add identity path with per-task scaling
        y = x * diag.view(1, T_eff, 1) + mix
        return y


# -------------------------
# Cross-Stitch Network for Many Tasks
# -------------------------

class CrossStitchMTLRegressor(nn.Module):
    """
    - Builds per-task representations (same shared MLP applied across tasks)
    - Applies low-rank cross-stitch after each hidden layer
    - Produces one scalar per task

    Inputs:
      x1: [B,1], x2: [B,1], x3: [B,15] => concatenated [B,17]
    Outputs:
      - If task_ids is None: returns [B, T]
      - If task_ids is provided (per-sample): returns [B, 1] predictions for each sample's task
        (computed using only active tasks in the batch for speed)
    """

    def __init__(
        self,
        num_tasks: int = 1155,
        input_dim: int = 17,
        hidden_dim: int = 128,
        num_layers: int = 3,
        rank: int = 8,
        dropout: float = 0.0,
        use_active_tasks_only: bool = True,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.num_tasks = num_tasks
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rank = rank
        self.dropout = dropout
        self.use_active_tasks_only = use_active_tasks_only

        # Task-specific input offsets (a lightweight way to break symmetry across tasks)
        self.task_embed = nn.Embedding(num_tasks, input_dim)
        nn.init.normal_(self.task_embed.weight, std=0.01)

        # Shared MLP layers (applied across tasks dimension)
        layers: List[nn.Module] = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers = nn.ModuleList(layers)

        # Cross-stitch after each hidden layer
        self.stitches = nn.ModuleList(
            [LowRankCrossStitch(num_tasks=num_tasks, rank=rank) for _ in range(num_layers)]
        )

        self.head = nn.Linear(hidden_dim, 1)

    @staticmethod
    def _concat_inputs(x_list: List[torch.Tensor]) -> torch.Tensor:
        if not isinstance(x_list, (list, tuple)) or len(x_list) != 3:
            raise ValueError("Expected inputs as [x1,x2,x3].")

        x1, x2, x3 = x_list
        x3 = torch.squeeze(x3)
        # Ensure shapes: [B,1], [B,1], [B,15]
        if x1.ndim != 2 or x2.ndim != 2 or x3.ndim != 2:
            raise ValueError("All input tensors must be rank-2: [B,features].")
        return torch.cat([x1, x2, x3], dim=1)  # [B,17]

    def forward(
        self,
        x_list: List[torch.Tensor],
        task_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_in = self._concat_inputs(x_list)  # [B,17]
        B = x_in.shape[0]

        if task_ids is None:
            # Compute all tasks
            task_idx = torch.arange(self.num_tasks, device=x_in.device, dtype=torch.long)  # [T]
            x = x_in.unsqueeze(1) + self.task_embed(task_idx).unsqueeze(0)  # [B,T,17]
        else:
            if task_ids.dtype != torch.long:
                task_ids = task_ids.long()
            if task_ids.ndim != 1 or task_ids.shape[0] != B:
                raise ValueError(f"task_ids must be [B], got {tuple(task_ids.shape)}")

            if self.use_active_tasks_only:
                # Only compute tasks that appear in this batch (unique)
                task_idx, inv = torch.unique(task_ids, sorted=True, return_inverse=True)  # task_idx [A], inv [B]
                x = x_in.unsqueeze(1) + self.task_embed(task_idx).unsqueeze(0)  # [B,A,17]
            else:
                # Compute all tasks, then gather (slower)
                task_idx = torch.arange(self.num_tasks, device=x_in.device, dtype=torch.long)
                x = x_in.unsqueeze(1) + self.task_embed(task_idx).unsqueeze(0)  # [B,T,17]
                inv = task_ids  # used later as direct indices into T

        # Shared MLP + cross-stitch
        for layer, stitch in zip(self.layers, self.stitches):
            x = layer(x)          # linear on last dim: [B,T_eff,H]
            x = F.relu(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = stitch(x, task_idx=task_idx if (task_ids is not None and self.use_active_tasks_only) else None)

        # Head per task
        y_all = self.head(x).squeeze(-1)  # [B, T_eff]

        if task_ids is None:
            return y_all  # [B,T]
        else:
            # Return per-sample prediction for its task
            if self.use_active_tasks_only:
                # inv maps each sample to its position in task_idx
                y = y_all[torch.arange(B, device=y_all.device), inv]  # [B]
            else:
                y = y_all[torch.arange(B, device=y_all.device), task_ids]  # [B]
            return y.unsqueeze(-1)  # [B,1]


# -------------------------
# Training / Evaluation
# -------------------------

@dataclass
class BatchA:
    x_list: List[torch.Tensor]
    y: torch.Tensor            # [B,1]
    task_ids: torch.Tensor     # [B]

@dataclass
class BatchB:
    x_list: List[torch.Tensor]
    y_all: torch.Tensor        # [B,T] or [B,T,1]


def parse_batch(batch) -> Union[BatchA, BatchB]:
    """
    Supports:
      ([x1,x2,x3], y, task_ids)
      ([x1,x2,x3], y_all)
    """
    if not isinstance(batch, (list, tuple)):
        raise ValueError("Batch must be tuple/list.")

    if len(batch) == 3:
        x_list, y, task_ids = batch
        return BatchA(x_list=x_list, y=y, task_ids=task_ids)
    elif len(batch) == 2:
        x_list, y_all = batch
        return BatchB(x_list=x_list, y_all=y_all)
    else:
        raise ValueError(f"Unsupported batch format with length {len(batch)}")


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: Iterable,
    device: torch.device,
    num_tasks: int,
) -> Dict[str, float]:
    """
    Returns:
      overall_rmse: RMSE over *all samples* in val (micro-average)
      min_task_rmse: min RMSE across tasks that appear in val
      max_task_rmse: max RMSE across tasks that appear in val

    Supports both batch formats:
      A) ([x1,x2,x3], y, task_ids)     y: [B,1], task_ids: [B]
      B) ([x1,x2,x3], y_all)          y_all: [B,T] or [B,T,1]
    """
    model.eval()

    # Per-task accumulators
    sse_task = torch.zeros(num_tasks, device=device)   # sum squared error per task
    n_task = torch.zeros(num_tasks, device=device)     # sample count per task

    # Global accumulators
    sse_total = torch.tensor(0.0, device=device)
    n_total = torch.tensor(0.0, device=device)

    for batch in loader:
        b = parse_batch(batch)

        if isinstance(b, BatchA):
            x_list = [t.to(device) for t in b.x_list]
            y = b.y.to(device).view(-1, 1)                 # [B,1]
            task_ids = b.task_ids.to(device).view(-1)      # [B]

            pred = model(x_list, task_ids=task_ids)        # [B,1]
            se = (pred - y).squeeze(-1).pow(2)             # [B]

            # global
            sse_total += se.sum()
            n_total += se.numel()

            # per-task
            sse_task.scatter_add_(0, task_ids, se)
            n_task.scatter_add_(0, task_ids, torch.ones_like(se))

        else:
            x_list = [t.to(device) for t in b.x_list]
            y_all = b.y_all.to(device)
            if y_all.ndim == 3 and y_all.shape[-1] == 1:
                y_all = y_all.squeeze(-1)                  # [B,T]

            pred_all = model(x_list, task_ids=None)         # [B,T]
            se_all = (pred_all - y_all).pow(2)              # [B,T]

            # global
            sse_total += se_all.sum()
            n_total += se_all.numel()

            # per-task: sum over batch dimension
            sse_task += se_all.sum(dim=0)                   # [T]
            n_task += torch.tensor(se_all.shape[0], device=device).repeat(num_tasks) \
                if se_all.shape[1] == num_tasks else torch.ones_like(sse_task) * se_all.shape[0]
            # NOTE: the above line assumes T == num_tasks. If not, adjust accordingly.

    overall_rmse = torch.sqrt(sse_total / torch.clamp(n_total, min=1.0)).item()

    # task RMSE only for tasks that appear (n_task > 0)
    mask = n_task > 0
    if mask.any():
        task_rmse = torch.sqrt(sse_task[mask] / n_task[mask])
        min_task_rmse = task_rmse.min().item()
        max_task_rmse = task_rmse.max().item()
    else:
        min_task_rmse = float("nan")
        max_task_rmse = float("nan")

    return {
        "overall_rmse": float(overall_rmse),
        "min_task_rmse": float(min_task_rmse),
        "max_task_rmse": float(max_task_rmse),
    }


def train_one_epoch(
    model: nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: Optional[float] = None,
) -> float:
    model.train()
    loss_fn = RMSELoss(reduction="mean")
    losses = []

    for batch in loader:
        b = parse_batch(batch)

        optimizer.zero_grad(set_to_none=True)

        if isinstance(b, BatchA):
            x_list = [t.to(device) for t in b.x_list]
            y = b.y.to(device).view(-1, 1)
            task_ids = b.task_ids.to(device).view(-1)
            pred = model(x_list, task_ids=task_ids)  # [B,1]
            loss = loss_fn(pred, y)
        else:
            x_list = [t.to(device) for t in b.x_list]
            y_all = b.y_all.to(device)
            if y_all.ndim == 3 and y_all.shape[-1] == 1:
                y_all = y_all.squeeze(-1)
            pred_all = model(x_list, task_ids=None)  # [B,T]
            loss = loss_fn(pred_all, y_all)

        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        losses.append(loss.item())

    return float(sum(losses) / max(1, len(losses)))


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_tasks", type=int, default=93) # ds1 1155, ds2 500 ds3 93
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # If your batches include per-sample task_ids, active-tasks-only is usually much faster:
    parser.add_argument("--use_active_tasks_only", action="store_true", default=True)
    parser.add_argument("--no_active_tasks_only", dest="use_active_tasks_only", action="store_false")

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    model = CrossStitchMTLRegressor(
        num_tasks=args.num_tasks,
        input_dim=17,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        rank=args.rank,
        dropout=args.dropout,
        use_active_tasks_only=args.use_active_tasks_only,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    config = get_args()

    train_loader, val_loader, data_info, train_data, test_data = get_dataloader_cross(config=config, year=1, matches=None)

    best_val = math.inf
    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, device, grad_clip=args.grad_clip)
        if val_loader is not None:
            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                num_tasks=args.num_tasks,
            )
            va = val_metrics["overall_rmse"]
            best_val = min(best_val, va)

            print(
                f"epoch {epoch:03d} | train RMSE {tr:.6f} | "
                f"val RMSE {va:.6f} | "
                f"val task RMSE min {val_metrics['min_task_rmse']:.6f} | "
                f"val task RMSE max {val_metrics['max_task_rmse']:.6f} | "
                f"best {best_val:.6f}"
            )
        else:
            print(f"epoch {epoch:03d} | train RMSE {tr:.6f}")


if __name__ == "__main__":
    main()
