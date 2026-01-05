#!/usr/bin/env python3
"""
epoch 010 | train_rmse=46.716514 | val_rmse=43.677451 | val_rmse_min=3.621685 | val_rmse_max=409.664398 | tasks_present=1155
"""

from __future__ import annotations

import argparse
import math
from typing import Any, Dict, List, Sequence, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src import Forecasting_Dataset_Nddr, get_dataloader_nddr
from src import get_args
# -------------------------
# Loss
# -------------------------
class RMSELoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(pred, target, reduction="mean")
        return torch.sqrt(mse + self.eps)


# -------------------------
# Helpers
# -------------------------
def ensure_2d(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(1) if x.dim() == 1 else x


def build_mlp(
    in_dim: int,
    hidden_dims: Sequence[int],
    out_dim: int,
    dropout: float = 0.0,
    activation: str = "gelu",
    layernorm: bool = True,
) -> nn.Sequential:
    act_map = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "tanh": nn.Tanh}
    if activation not in act_map:
        raise ValueError(f"activation must be one of {list(act_map.keys())}")

    layers: List[nn.Module] = []
    d = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(d, h))
        if layernorm:
            layers.append(nn.LayerNorm(h))
        layers.append(act_map[activation]())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


# -------------------------
# Batch parser (matches YOUR dataloader)
# -------------------------
def parse_batch(batch: Any, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        task_id: [B] long
        x:       [B, d_in] float
        y:       [B, 1] float

    Supports batches:
      A) (vectors, y, task_id)  where vectors is list/tuple of tensors
      B) (v1, v2, v3, y, task_id)
      C) dict with keys like {"vectors":..., "y":..., "task_id":...} or {"x":..., "label":..., "task_id":...}
    """
    vectors = None
    y = None
    task_id = None

    if isinstance(batch, (tuple, list)):
        if len(batch) == 3:
            vectors, y, task_id = batch
        elif len(batch) == 5:
            v1, v2, v3, y, task_id = batch
            vectors = [v1, v2, v3]
        else:
            raise ValueError(f"Unsupported tuple/list batch length {len(batch)}. Expected 3 or 5.")
    elif isinstance(batch, dict):
        # Try common naming
        for k in ("vectors", "x", "inputs", "features"):
            if k in batch:
                vectors = batch[k]
                break
        for k in ("y", "label", "labels", "target", "targets"):
            if k in batch:
                y = batch[k]
                break
        for k in ("task_id", "task", "taskids"):
            if k in batch:
                task_id = batch[k]
                break
        if vectors is None or y is None or task_id is None:
            raise ValueError("Dict batch missing one of vectors/x, y/label, task_id.")
    else:
        raise ValueError("Unsupported batch type. Expected tuple/list or dict.")

    if not isinstance(vectors, (list, tuple)) or len(vectors) != 3:
        raise ValueError("Expected vectors as [v1, v2, v3] list/tuple of length 3.")

    v1, v2, v3 = vectors
    v3 = torch.squeeze(v3)
    v1 = ensure_2d(v1.to(device).float())  # [B,1]
    v2 = ensure_2d(v2.to(device).float())  # [B,1]
    v3 = ensure_2d(v3.to(device).float())  # [B,15]

    x = torch.cat([v1, v2, v3], dim=1)     # [B, 17]

    y = ensure_2d(torch.as_tensor(y).to(device).float())
    if y.size(1) != 1:
        raise ValueError(f"Expected y shape [B,1] (or [B]); got {tuple(y.shape)}")

    task_id = torch.as_tensor(task_id).to(device).view(-1).long()  # [B]

    return task_id, x, y


# -------------------------
# MGMoE Model
# -------------------------
class MGMoERegressor(nn.Module):
    """
    Task-conditioned MGMoE:
      - shared experts: x -> expert repr
      - gate: [x; task_emb] -> softmax over experts
      - per-task linear head: (mix * W_task).sum + b_task
    """

    def __init__(
        self,
        n_tasks: int,
        d_in: int,
        n_experts: int = 8,
        d_expert: int = 64,
        expert_hidden: Sequence[int] = (128, 128),
        gate_hidden: Sequence[int] = (64, 64),
        task_emb_dim: int = 32,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.n_tasks = n_tasks
        self.d_in = d_in
        self.n_experts = n_experts
        self.d_expert = d_expert

        self.task_emb = nn.Embedding(n_tasks, task_emb_dim)

        self.experts = nn.ModuleList(
            [
                build_mlp(
                    in_dim=d_in,
                    hidden_dims=expert_hidden,
                    out_dim=d_expert,
                    dropout=dropout,
                    activation=activation,
                    layernorm=True,
                )
                for _ in range(n_experts)
            ]
        )

        self.gate = build_mlp(
            in_dim=d_in + task_emb_dim,
            hidden_dims=gate_hidden,
            out_dim=n_experts,
            dropout=dropout,
            activation=activation,
            layernorm=True,
        )

        self.head_W = nn.Embedding(n_tasks, d_expert)
        self.head_b = nn.Embedding(n_tasks, 1)
        self._init_head()

    def _init_head(self) -> None:
        nn.init.normal_(self.head_W.weight, mean=0.0, std=1.0 / math.sqrt(self.d_expert))
        nn.init.zeros_(self.head_b.weight)

    def forward(self, task_id: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        task_id = task_id.long()
        x = x.float()

        t = self.task_emb(task_id)                 # [B, task_emb_dim]
        gate_in = torch.cat([x, t], dim=1)         # [B, d_in+task_emb_dim]
        w = F.softmax(self.gate(gate_in), dim=1)   # [B, n_experts]

        E = torch.stack([e(x) for e in self.experts], dim=1)  # [B, n_experts, d_expert]
        mix = torch.bmm(w.unsqueeze(1), E).squeeze(1)         # [B, d_expert]

        Wt = self.head_W(task_id)  # [B, d_expert]
        bt = self.head_b(task_id)  # [B, 1]
        y_hat = (mix * Wt).sum(dim=1, keepdim=True) + bt      # [B,1]
        return y_hat


# -------------------------
# Train / Eval
# -------------------------
def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    grad_clip: Optional[float] = 1.0,
) -> Dict[str, float]:
    model.train()
    total_sse = 0.0
    total_n = 0

    for batch in loader:
        task_id, x, y = parse_batch(batch, device=device)
        pred = model(task_id, x)

        loss = loss_fn(pred, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_sse += float((pred.detach() - y).pow(2).sum().item())
        total_n += int(y.numel())

    return {"rmse": math.sqrt(total_sse / max(total_n, 1))}


@torch.no_grad()
def evaluate_with_minmax(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    n_tasks: int,
) -> Dict[str, float]:
    model.eval()

    total_sse = 0.0
    total_n = 0

    task_sse = torch.zeros(n_tasks, dtype=torch.float64, device="cpu")
    task_n = torch.zeros(n_tasks, dtype=torch.long, device="cpu")

    for batch in loader:
        task_id, x, y = parse_batch(batch, device=device)
        pred = model(task_id, x)

        se_vec = (pred - y).pow(2).view(-1).detach().cpu().to(torch.float64)  # [B]
        tid = task_id.view(-1).detach().cpu()                                  # [B]

        total_sse += float(se_vec.sum().item())
        total_n += int(se_vec.numel())

        task_sse.index_add_(0, tid, se_vec)
        task_n.index_add_(0, tid, torch.ones_like(tid, dtype=torch.long))

    overall_rmse = math.sqrt(total_sse / max(total_n, 1))

    present = task_n > 0
    tasks_present = int(present.sum().item())
    if tasks_present == 0:
        return {"rmse": float("nan"), "rmse_min": float("nan"), "rmse_max": float("nan"), "tasks_present": 0}

    per_task_rmse = torch.sqrt(task_sse[present] / task_n[present].to(torch.float64))
    return {
        "rmse": overall_rmse,
        "rmse_min": float(per_task_rmse.min().item()),
        "rmse_max": float(per_task_rmse.max().item()),
        "tasks_present": tasks_present,
    }


# -------------------------
# Wire in your loaders
# -------------------------
def get_dataloaders(args: argparse.Namespace):
    """
    Replace with your real dataloaders and return:
        train_loader, val_loader
    """
    raise NotImplementedError("Implement get_dataloaders(args) for your dataset.")


# -------------------------
# Main
# -------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_tasks", type=int, default=1155)
    p.add_argument("--n_experts", type=int, default=8)
    p.add_argument("--d_expert", type=int, default=64)
    p.add_argument("--task_emb_dim", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)

    config = get_args()

    train_loader, val_loader, data_info, train_data, test_data = get_dataloader_nddr(config=config, year=1,
                                                                                     matches=None)

    # Your features are [B,1] + [B,1] + [B,15] => d_in = 17
    # (If that ever changes, you can infer from one batch, but here it's fixed.)
    d_in = 17

    model = MGMoERegressor(
        n_tasks=args.n_tasks,
        d_in=d_in,
        n_experts=args.n_experts,
        d_expert=args.d_expert,
        expert_hidden=(128, 128),
        gate_hidden=(64, 64),
        task_emb_dim=args.task_emb_dim,
        dropout=args.dropout,
        activation="gelu",
    ).to(device)

    loss_fn = RMSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            grad_clip=args.grad_clip,
        )

        msg = f"epoch {epoch:03d} | train_rmse={tr['rmse']:.6f}"

        if val_loader is not None:
            va = evaluate_with_minmax(
                model=model,
                loader=val_loader,
                device=device,
                n_tasks=args.n_tasks,
            )
            msg += (
                f" | val_rmse={va['rmse']:.6f}"
                f" | val_rmse_min={va['rmse_min']:.6f}"
                f" | val_rmse_max={va['rmse_max']:.6f}"
                f" | tasks_present={va['tasks_present']}"
            )

        print(msg)


if __name__ == "__main__":
    main()
