# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tests for ``bioemu.extract_internal_features`` (poolers + request validation).

These tests need neither a model checkpoint nor network access: the poolers run
on random tensors, and the validation checks in
:func:`extract_bioemu_internal_features` fire before any checkpoint download.

The module imports ``torch_geometric`` at import time, which segfaults under the
base interpreter, so the whole file is skipped unless pyg is importable. Run with
the SE3nv env. ``--noconftest`` avoids the repo's ``tests/conftest.py``, which
hard-imports an optional vendored dependency (``modelcif``) this file does not
need::

    PYTHONPATH=/data1/hanqun/bioemu/src \\
        /data1/hanqun/miniconda3/envs/SE3nv/bin/python \\
        -m pytest tests/test_internal_features.py -v --noconftest
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch_geometric")

import torch  # noqa: E402 - imported after the pyg guard

from bioemu.extract_internal_features import (  # noqa: E402 - imported after the pyg guard
    EXPECTED_D_MODEL,
    EXPECTED_NUM_LAYERS,
    AttentionPooler,
    LearnedQueryPooler,
    MeanPooler,
    ResiduePooler,
    extract_bioemu_internal_features,
    make_pooler,
)

D = EXPECTED_D_MODEL  # 512


def _poolers() -> list[ResiduePooler]:
    return [MeanPooler(), AttentionPooler(D), LearnedQueryPooler(D)]


@pytest.mark.parametrize("L", [1, 7, 53, 128])
def test_batched_pooling_shape_is_width_only(L: int) -> None:
    """A batched [1, L, D] input pools to [1, D] for every pooler and every L."""
    x = torch.randn(1, L, D)
    for pooler in _poolers():
        out = pooler(x)
        assert out.shape == (1, D), f"{type(pooler).__name__} gave {tuple(out.shape)} for L={L}"


@pytest.mark.parametrize("L", [1, 7, 53, 128])
def test_unbatched_pooling_shape_is_width_only(L: int) -> None:
    """An unbatched [L, D] input pools to [D] for every pooler and every L."""
    x = torch.randn(L, D)
    for pooler in _poolers():
        out = pooler(x)
        assert out.shape == (D,), f"{type(pooler).__name__} gave {tuple(out.shape)} for L={L}"


def test_pooled_width_invariant_to_length() -> None:
    """The pooled width stays D regardless of the residue count (key invariance)."""
    for pooler in _poolers():
        widths = {pooler(torch.randn(L, D)).shape[-1] for L in (1, 7, 53, 128)}
        assert widths == {D}


def test_make_pooler_types() -> None:
    """make_pooler returns the right concrete pooler type for each name."""
    assert isinstance(make_pooler("mean"), MeanPooler)
    assert isinstance(make_pooler("attention"), AttentionPooler)
    assert isinstance(make_pooler("learned"), LearnedQueryPooler)


def test_make_pooler_rejects_unknown() -> None:
    """An unrecognised pooler name raises ValueError."""
    with pytest.raises(ValueError):
        make_pooler("bogus")


def test_mean_pooler_has_no_parameters() -> None:
    """MeanPooler is parameter-free; the parametric poolers expose parameters."""
    assert sum(p.numel() for p in MeanPooler().parameters()) == 0
    assert sum(p.numel() for p in AttentionPooler(D).parameters()) > 0
    assert sum(p.numel() for p in LearnedQueryPooler(D).parameters()) > 0


def test_attention_weights_sum_to_one() -> None:
    """AttentionPooler's implicit residue weights form a softmax (sum to 1)."""
    L = 17
    x = torch.randn(1, L, D)
    pooler = AttentionPooler(D)
    scores = pooler.score(x)  # [1, L, 1]
    weights = torch.softmax(scores, dim=-2)  # [1, L, 1]
    total = weights.sum(dim=-2)  # [1, 1]
    assert torch.allclose(total, torch.ones_like(total), atol=1e-5)


def test_invalid_layer_index_raises_before_download() -> None:
    """Layer index 8 is out of range (0..7) and must raise before any download."""
    with pytest.raises(ValueError):
        extract_bioemu_internal_features("AAAA", layers=[8])


def test_invalid_diffusion_time_raises_before_download() -> None:
    """Diffusion time 1.5 is outside [0, 1] and must raise before any download."""
    with pytest.raises(ValueError):
        extract_bioemu_internal_features("AAAA", layers=[-1], diffusion_times=[1.5])


def test_architecture_constants() -> None:
    """The runtime guards match BioEmu's expected encoder shape."""
    assert EXPECTED_NUM_LAYERS == 8
    assert EXPECTED_D_MODEL == 512
