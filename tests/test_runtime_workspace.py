from __future__ import annotations

import pytest
import torch

from yozakura.runtime import _workspace_rows


def test_workspace_rows_caps_large_projection() -> None:
    rows = _workspace_rows(
        output_rows=1_000_000,
        rank=128,
        dtype=torch.float32,
        workspace_mib=1,
    )

    assert rows == 1024


def test_workspace_rows_never_exceeds_projection() -> None:
    rows = _workspace_rows(
        output_rows=64,
        rank=8,
        dtype=torch.float16,
        workspace_mib=256,
    )

    assert rows == 64


@pytest.mark.parametrize("workspace_mib", [0, -1])
def test_workspace_rows_rejects_non_positive_limit(workspace_mib: int) -> None:
    with pytest.raises(ValueError, match="workspace_mib must be positive"):
        _workspace_rows(
            output_rows=64,
            rank=8,
            dtype=torch.float16,
            workspace_mib=workspace_mib,
        )
