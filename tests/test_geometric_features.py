# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Backward-compat tests for the existing geometric featurizer.

Exercises :mod:`bioemu.extract_md_features` (the mdtraj-based MD featurizer) on a
tiny real structure to prove the geometric path still works after the internal-
features module was added. This file deliberately does NOT import
``torch_geometric``; ``extract_md_features`` depends on ``mdtraj`` only, so run it
under whichever interpreter provides mdtraj (e.g. the base env). ``--noconftest``
avoids the repo's ``tests/conftest.py``, which hard-imports ``hydra`` /
``torch_geometric`` (absent under the base interpreter) and is unused here::

    PYTHONPATH=/data1/hanqun/bioemu/src \\
        /data1/hanqun/miniconda3/bin/python \\
        -m pytest tests/test_geometric_features.py -v --noconftest
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mdtraj")

import mdtraj  # noqa: E402 - imported after the mdtraj guard

from bioemu.extract_md_features import (  # noqa: E402 - imported after the mdtraj guard
    BLOCK_ORDER,
    _contact_block,
    _detect_topology,
    _dihedral_block,
    _distance_block,
    extract_features,
)

# A 10-residue backbone-only chignolin structure shipped with the repo.
CHIGNOLIN_PDB = Path(__file__).parent / "training" / "chignolin.pdb"


def test_extract_features_end_to_end_on_chignolin() -> None:
    """The public extract_features path returns finite blocks of expected shape."""
    result, topo = extract_features(str(CHIGNOLIN_PDB))

    assert result.n_frames == 1
    assert topo.n_residues == 10
    assert topo.sequence == "GYDPETGTWG"
    assert topo.sidechains_present is False  # backbone-only input

    for block in BLOCK_ORDER:
        assert block in result.blocks, f"missing block {block}"
        mat = result.blocks[block]
        assert mat.ndim == 2 and mat.shape[0] == result.n_frames
        assert np.isfinite(mat).all(), f"non-finite values in {block}"

    # Per-residue Ca RMSF: one value per residue, finite.
    assert result.rmsf_nm is not None
    assert result.rmsf_nm.shape == (topo.n_residues,)
    assert np.isfinite(result.rmsf_nm).all()


def _tiny_trajectory(n_frames: int = 3) -> mdtraj.Trajectory:
    """Build a small multi-frame trajectory from the chignolin topology."""
    base = mdtraj.load(str(CHIGNOLIN_PDB))
    rng = np.random.default_rng(0)
    # Stack the single frame and add small jitter so frames differ.
    xyz = np.repeat(base.xyz, n_frames, axis=0)
    xyz[1:] += rng.normal(scale=0.01, size=xyz[1:].shape).astype(np.float32)
    return mdtraj.Trajectory(xyz, base.topology)


def test_pure_blocks_on_synthetic_trajectory() -> None:
    """The dihedral / distance / contact blocks run on an in-memory trajectory."""
    traj = _tiny_trajectory(n_frames=3)
    topo = _detect_topology(traj.topology, periodic=False)

    dih, dih_names = _dihedral_block(traj)
    assert dih.shape[0] == traj.n_frames
    assert dih.shape[1] == len(dih_names)
    assert dih.shape[1] > 0  # phi/psi/omega present even backbone-only
    assert np.isfinite(dih).all()
    # cos/sin encoding stays in [-1, 1].
    assert dih.min() >= -1.0 - 1e-5 and dih.max() <= 1.0 + 1e-5

    dist = _distance_block(traj, topo)
    assert dist.shape == (traj.n_frames, topo.ca_pairs.shape[0])
    assert np.isfinite(dist).all()
    assert (dist >= 0).all()  # distances are non-negative

    contacts = _contact_block(dist)
    assert contacts.shape == dist.shape
    assert np.isfinite(contacts).all()
    # Soft logistic contact scores live in [0, 1].
    assert contacts.min() >= 0.0 and contacts.max() <= 1.0
