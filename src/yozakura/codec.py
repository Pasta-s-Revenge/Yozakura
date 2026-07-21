from __future__ import annotations

import torch


def quantize_symmetric(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.detach().float().cpu()
    scale = torch.clamp(x.abs().amax() / 127.0, min=1e-12)
    q = torch.clamp(torch.round(x / scale), -127, 127).to(torch.int8)
    return q, scale.reshape(1).float()


def dequantize_symmetric(q: torch.Tensor, scale: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return (q.to(device=device, dtype=torch.float32) * scale.to(device=device, dtype=torch.float32)).to(dtype)
