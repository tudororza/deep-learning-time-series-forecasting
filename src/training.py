"""Shared training utilities."""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual_values = np.asarray(actual, dtype=np.float64)
    predicted_values = np.asarray(predicted, dtype=np.float64)
    if actual_values.shape != predicted_values.shape:
        raise ValueError("WAPE inputs must have identical shapes")
    denominator = np.abs(actual_values).sum()
    if denominator == 0:
        raise ValueError("WAPE is undefined when all targets are zero")
    return float(np.abs(actual_values - predicted_values).sum() / denominator)


@dataclass
class TrainingResult:
    state_dict: dict[str, torch.Tensor]
    best_epoch: int
    best_metric: float
    elapsed_seconds: float
    batch_size: int


def _move_batch(
    batch: tuple[torch.Tensor, ...], device: torch.device
) -> tuple[torch.Tensor, ...]:
    return tuple(tensor.to(device, non_blocking=device.type == "cuda") for tensor in batch)


def choose_batch_size(
    model: nn.Module,
    dataset: Dataset[tuple[torch.Tensor, ...]],
    requested: int,
    device: torch.device,
) -> int:
    """Probe planned sizes on GPU; CPU keeps the requested size."""
    if device.type != "cuda":
        return requested
    sizes = [size for size in (requested, 128, 64) if size <= requested]
    for size in dict.fromkeys(sizes):
        loader = DataLoader(dataset, batch_size=size, shuffle=False, num_workers=0)
        try:
            history, future, series, baseline, scale, truth = _move_batch(next(iter(loader)), device)
            model.zero_grad(set_to_none=True)
            prediction = model(history, future, series, baseline, scale)
            torch.nn.functional.l1_loss(prediction, truth).backward()
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            return size
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    raise RuntimeError("The model does not fit even with batch size 64")


def train_model(
    model: nn.Module,
    dataset: Dataset[tuple[torch.Tensor, ...]],
    *,
    device: torch.device,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    gradient_clip: float,
    seed: int,
    evaluate: Callable[[nn.Module], float],
    num_workers: int = 0,
) -> TrainingResult:
    set_seed(seed)
    model.to(device)
    selected_batch_size = choose_batch_size(model, dataset, batch_size, device)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=selected_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_metric = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    started = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        examples = 0
        for batch in loader:
            history, future, series, baseline, scale, truth = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(history, future, series, baseline, scale)
            loss = torch.nn.functional.l1_loss(prediction, truth)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            total_loss += float(loss.detach()) * len(history)
            examples += len(history)

        metric = evaluate(model)
        mean_loss = total_loss / max(examples, 1)
        print(f"epoch={epoch:02d} train_l1={mean_loss:.6f} validation_metric={metric:.6f}")
        if metric < best_metric - 1e-8:
            best_metric = metric
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"early stopping after epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return TrainingResult(
        state_dict=copy.deepcopy(best_state),
        best_epoch=best_epoch,
        best_metric=best_metric,
        elapsed_seconds=time.perf_counter() - started,
        batch_size=selected_batch_size,
    )


def fit_fixed_epochs(
    model: nn.Module,
    dataset: Dataset[tuple[torch.Tensor, ...]],
    *,
    device: torch.device,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    gradient_clip: float,
    seed: int,
    num_workers: int = 0,
) -> TrainingResult:
    """Retrain on all public data for the selected number of epochs."""
    set_seed(seed)
    model.to(device)
    selected_batch_size = choose_batch_size(model, dataset, batch_size, device)
    loader = DataLoader(
        dataset,
        batch_size=selected_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    started = time.perf_counter()
    last_loss = float("nan")
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        examples = 0
        for batch in loader:
            history, future, series, baseline, scale, truth = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(history, future, series, baseline, scale)
            loss = torch.nn.functional.l1_loss(prediction, truth)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            total += float(loss.detach()) * len(history)
            examples += len(history)
        last_loss = total / max(examples, 1)
        print(f"retrain_epoch={epoch:02d} train_l1={last_loss:.6f}")
    state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    return TrainingResult(
        state_dict=state,
        best_epoch=epochs,
        best_metric=last_loss,
        elapsed_seconds=time.perf_counter() - started,
        batch_size=selected_batch_size,
    )

