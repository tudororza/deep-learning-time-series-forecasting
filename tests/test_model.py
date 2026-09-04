from __future__ import annotations

import numpy as np
import pytest
import torch

from src.device import preflight
from src.model import ModelConfig, ResidualLSTM


@pytest.mark.parametrize(
    ("input_size", "future_size", "output_size"), [(6, 4, 1), (10, 4, 2)]
)
def test_model_shapes_and_optimizer_update(
    input_size: int, future_size: int, output_size: int
) -> None:
    torch.manual_seed(42)
    model = ResidualLSTM(
        ModelConfig(input_size, future_size, n_series=3, output_size=output_size)
    )
    history = torch.randn(5, 168, input_size)
    future = torch.randn(5, 24, future_size)
    series = torch.tensor([0, 1, 2, 0, 1])
    baseline = torch.randn(5, 24, output_size)
    scale = torch.ones(5, output_size)
    before = model.decoder[-1].weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    prediction = model(history, future, series, baseline, scale)
    assert prediction.shape == (5, 24, output_size)
    loss = prediction.square().mean()
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, model.decoder[-1].weight)


def test_zero_model_returns_baseline() -> None:
    model = ResidualLSTM(ModelConfig(1, 0, 2, 1))
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    baseline = torch.arange(48, dtype=torch.float32).reshape(2, 24, 1)
    result = model(
        torch.zeros(2, 168, 1),
        torch.empty(2, 24, 0),
        torch.tensor([0, 1]),
        baseline,
        torch.ones(2, 1),
    )
    torch.testing.assert_close(result, baseline)


def test_device_preflight_completes() -> None:
    device, report = preflight()
    assert device.type in {"cpu", "cuda"}
    assert report.tests_passed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No working CUDA/ROCm device")
def test_cpu_gpu_forward_agreement() -> None:
    torch.manual_seed(42)
    cpu_model = ResidualLSTM(ModelConfig(3, 2, 1, 1)).eval()
    gpu_model = ResidualLSTM(ModelConfig(3, 2, 1, 1)).eval().cuda()
    gpu_model.load_state_dict(cpu_model.state_dict())
    inputs = (
        torch.randn(2, 12, 3),
        torch.randn(2, 4, 2),
        torch.zeros(2, dtype=torch.long),
        torch.randn(2, 4, 1),
        torch.ones(2, 1),
    )
    with torch.no_grad():
        cpu = cpu_model(*inputs)
        gpu = gpu_model(*(value.cuda() for value in inputs)).cpu()
    np.testing.assert_allclose(cpu.numpy(), gpu.numpy(), rtol=1e-4, atol=1e-5)

