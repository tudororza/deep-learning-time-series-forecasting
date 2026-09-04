"""Device selection and a real PyTorch compute preflight."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass
class DeviceReport:
    selected: str
    torch_version: str
    cuda_available: bool
    hip_version: str | None
    device_name: str | None
    tests_passed: bool
    error: str | None = None


def preflight(prefer_accelerator: bool = True) -> tuple[torch.device, DeviceReport]:
    """Run allocation, matmul, and LSTM training checks before using an accelerator."""
    cuda_available = bool(torch.cuda.is_available())
    candidate = torch.device("cuda" if prefer_accelerator and cuda_available else "cpu")
    name: str | None = None
    try:
        if candidate.type == "cuda":
            name = torch.cuda.get_device_name(0)
        x = torch.randn(32, 32, device=candidate)
        loss = (x @ x.T).square().mean()
        lstm = torch.nn.LSTM(4, 8, batch_first=True).to(candidate)
        optimizer = torch.optim.AdamW(lstm.parameters(), lr=1e-3)
        sequence = torch.randn(2, 12, 4, device=candidate)
        output, _ = lstm(sequence)
        total = loss + output.square().mean()
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        if candidate.type == "cuda":
            torch.cuda.synchronize()
        report = DeviceReport(
            selected=candidate.type,
            torch_version=torch.__version__,
            cuda_available=cuda_available,
            hip_version=getattr(torch.version, "hip", None),
            device_name=name,
            tests_passed=True,
        )
        return candidate, report
    except Exception as exc:  # pragma: no cover - depends on local driver
        report = DeviceReport(
            selected="cpu",
            torch_version=torch.__version__,
            cuda_available=cuda_available,
            hip_version=getattr(torch.version, "hip", None),
            device_name=name,
            tests_passed=False,
            error=f"{type(exc).__name__}: {exc}",
        )
        return torch.device("cpu"), report


def write_report(report: DeviceReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2) + "\n")

