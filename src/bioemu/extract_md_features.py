# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Extract molecular-dynamics featurizations from protein trajectories.

Works on BioEmu outputs (backbone-only or side-chain-reconstructed XTC/PDB pairs
produced by ``sidechain_relax.py``) as well as any mdtraj-loadable trajectory.
Topology is auto-detected: chi dihedrals and side-chain-dependent quantities are
included only when side chains are present.

Per-frame feature blocks
  * dihedrals  - phi/psi/omega + chi1-4, encoded as (cos, sin) pairs.
  * distances  - Ca-Ca distances for residue pairs with sequence separation > 3 (nm).
  * contacts   - soft logistic contact scores for the same Ca-Ca pairs.
  * global     - radius of gyration, end-to-end distance, RMSD to reference,
                 fraction of native contacts, DSSP helix/sheet/coil fractions.
  * solvation  - total + per-residue SASA, backbone H-bond count.
Per-residue (static): Ca RMSF and mean SASA.

Outputs (written to ``--out-dir``)
  * <prefix>_features.npz   - raw per-frame blocks + concatenated X (TICA/MSM).
  * <prefix>_features_ml.pt - z-scored torch tensors + normalisation stats.
  * <prefix>_summary.csv    - per-block statistics.
  * <prefix>_per_residue.csv- per-residue RMSF / mean SASA.
  * <prefix>_metadata.json  - run configuration and topology description.

Example
  python -m bioemu.extract_md_features --traj samples_md_equil.xtc \\
      --top samples_md_equil.pdb --out-dir features/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from itertools import combinations

import mdtraj
import numpy as np

logger = logging.getLogger(__name__)

# --- featurization constants -------------------------------------------------
SEQUENCE_SEPARATION = 3  # only Ca-Ca pairs with |i - j| > this are kept
CONTACT_CUTOFF_NM = 1.0  # midpoint of the soft contact switch (matches foldedness.py)
CONTACT_WIDTH_NM = 0.1  # steepness of the soft contact switching function
HBOND_ENERGY_CUTOFF = -0.5  # kcal/mol; Kabsch-Sander H-bond threshold
LARGE_PAIR_WARNING = 50_000  # warn when the distance block exceeds this width

BACKBONE_ATOM_NAMES = {"N", "CA", "C", "O", "CB", "OXT"}
DIHEDRALS = ("phi", "psi", "omega", "chi1", "chi2", "chi3", "chi4")
# Atom column whose residue labels the dihedral (mdtraj atom ordering).
_DIHEDRAL_RES_COL = {"phi": 2, "psi": 1, "omega": 2, "chi1": 1, "chi2": 1, "chi3": 1, "chi4": 1}
BLOCK_ORDER = ("dihedrals", "distances", "contacts", "global", "solvation")


@dataclass
class TopologyInfo:
    """Static description of the trajectory topology."""

    n_residues: int
    sequence: str
    sidechains_present: bool
    has_hydrogens: bool
    periodic: bool
    ca_atom_indices: np.ndarray  # CA atom index per residue, shape (n_residues,)
    ca_pairs: np.ndarray  # CA atom-index pairs, shape (n_pairs, 2)
    pair_labels: list[str]  # human-readable label per CA pair
    residue_resseq: np.ndarray
    residue_resname: list[str]


@dataclass
class FeatureResult:
    """Accumulated featurization output."""

    blocks: dict[str, np.ndarray] = field(default_factory=dict)  # block -> (n_frames, dim)
    block_names: dict[str, list[str]] = field(default_factory=dict)  # block -> column names
    rmsf_nm: np.ndarray | None = None  # per-residue Ca RMSF
    mean_sasa_nm2: np.ndarray | None = None  # per-residue mean SASA
    n_frames: int = 0


# --- topology handling -------------------------------------------------------
def _detect_topology(top: mdtraj.Topology, periodic: bool) -> TopologyInfo:
    """Inspect a topology and decide which featurizers are applicable."""
    ca_atom_indices = top.select("name CA")
    n_residues = len(ca_atom_indices)
    if n_residues == 0:
        raise ValueError("No CA atoms found; this does not look like a protein topology.")

    sidechains_present = any(a.name not in BACKBONE_ATOM_NAMES for a in top.atoms)
    has_hydrogens = any(a.element is not None and a.element.symbol == "H" for a in top.atoms)

    residues = [top.atom(int(i)).residue for i in ca_atom_indices]
    sequence = "".join(r.code if r.code is not None else "X" for r in residues)
    residue_resseq = np.array([r.resSeq for r in residues], dtype=np.int64)
    residue_resname = [r.name for r in residues]

    # CA-CA residue pairs separated by more than SEQUENCE_SEPARATION in sequence.
    pairs, labels = [], []
    for i, j in combinations(range(n_residues), 2):
        if j - i > SEQUENCE_SEPARATION:
            pairs.append((ca_atom_indices[i], ca_atom_indices[j]))
            labels.append(f"{residue_resname[i]}{residue_resseq[i]}-"
                          f"{residue_resname[j]}{residue_resseq[j]}")
    ca_pairs = np.array(pairs, dtype=np.int64).reshape(-1, 2)

    if len(ca_pairs) > LARGE_PAIR_WARNING:
        logger.warning(
            "%d CA-CA pairs: distance/contact blocks will be large. "
            "Consider --stride to subsample frames.", len(ca_pairs)
        )

    return TopologyInfo(
        n_residues=n_residues,
        sequence=sequence,
        sidechains_present=sidechains_present,
        has_hydrogens=has_hydrogens,
        periodic=periodic,
        ca_atom_indices=np.asarray(ca_atom_indices, dtype=np.int64),
        ca_pairs=ca_pairs,
        pair_labels=labels,
        residue_resseq=residue_resseq,
        residue_resname=residue_resname,
    )


# --- per-frame feature blocks ------------------------------------------------
def _dihedral_block(traj: mdtraj.Trajectory) -> tuple[np.ndarray, list[str]]:
    """phi/psi/omega + chi1-4 encoded as (cos, sin) pairs.

    Encoding angles as cos/sin removes the 2*pi discontinuity, which is required
    for TICA/MSM and for stable ML training.
    """
    arrays: list[np.ndarray] = []
    names: list[str] = []
    for angle in DIHEDRALS:
        indices, values = getattr(mdtraj, f"compute_{angle}")(traj)
        if values.shape[1] == 0:  # e.g. chi on a backbone-only topology
            continue
        col = _DIHEDRAL_RES_COL[angle]
        res = [traj.topology.atom(int(idx[col])).residue for idx in indices]
        labels = [f"{r.name}{r.resSeq}" for r in res]
        cos = np.cos(values).astype(np.float32)
        sin = np.sin(values).astype(np.float32)
        # Interleave cos/sin per dihedral so columns stay grouped by residue.
        interleaved = np.empty((values.shape[0], 2 * values.shape[1]), dtype=np.float32)
        interleaved[:, 0::2] = cos
        interleaved[:, 1::2] = sin
        arrays.append(interleaved)
        for lab in labels:
            names.append(f"{angle}_cos_{lab}")
            names.append(f"{angle}_sin_{lab}")
    if not arrays:
        return np.zeros((traj.n_frames, 0), dtype=np.float32), []
    return np.concatenate(arrays, axis=1), names


def _distance_block(traj: mdtraj.Trajectory, topo: TopologyInfo) -> np.ndarray:
    """Ca-Ca distances (nm) for the precomputed residue pairs."""
    if topo.ca_pairs.shape[0] == 0:
        return np.zeros((traj.n_frames, 0), dtype=np.float32)
    return mdtraj.compute_distances(traj, topo.ca_pairs, periodic=topo.periodic).astype(np.float32)


def _contact_block(
    distances_nm: np.ndarray,
    cutoff_nm: float = CONTACT_CUTOFF_NM,
    width_nm: float = CONTACT_WIDTH_NM,
) -> np.ndarray:
    """Soft logistic contact score in [0, 1] from Ca-Ca distances."""
    # expit((cutoff - d) / width): ->1 when close, ->0 when far.
    z = (cutoff_nm - distances_nm) / width_nm
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def _global_block(
    traj: mdtraj.Trajectory,
    topo: TopologyInfo,
    reference: mdtraj.Trajectory,
    native_mask: np.ndarray,
    contacts: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Scalar per-frame descriptors: Rg, end-to-end, RMSD, FNC, DSSP fractions."""
    rg = mdtraj.compute_rg(traj).astype(np.float32)

    ca = topo.ca_atom_indices
    end_to_end = np.linalg.norm(
        traj.xyz[:, ca[0], :] - traj.xyz[:, ca[-1], :], axis=-1
    ).astype(np.float32)

    # mdtraj.rmsd superposes internally and does not mutate `traj`.
    rmsd = mdtraj.rmsd(traj, reference, atom_indices=ca).astype(np.float32)

    # Fraction of native contacts relative to the reference frame.
    if native_mask.any():
        fnc = contacts[:, native_mask].mean(axis=1).astype(np.float32)
    else:
        fnc = np.zeros(traj.n_frames, dtype=np.float32)

    columns = [rg, end_to_end, rmsd, fnc]
    names = ["radius_of_gyration", "end_to_end_distance", "rmsd_to_reference",
             "fraction_native_contacts"]
    try:
        dssp = mdtraj.compute_dssp(traj, simplified=True)
        for code, label in (("H", "helix"), ("E", "sheet"), ("C", "coil")):
            columns.append((dssp == code).mean(axis=1).astype(np.float32))
            names.append(f"dssp_{label}_fraction")
    except Exception as err:  # pragma: no cover - DSSP can fail on odd topologies
        logger.warning("Skipping DSSP fractions: %s", err)
    return np.stack(columns, axis=1), names


def _solvation_block(traj: mdtraj.Trajectory) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Total + per-residue SASA and backbone H-bond count.

    Returns the per-frame block, its column names, and the per-residue SASA
    array (kept separately so the caller can average it over all frames).
    """
    sasa_res = mdtraj.shrake_rupley(traj, mode="residue").astype(np.float32)
    total_sasa = sasa_res.sum(axis=1, keepdims=True)

    hbond_mats = mdtraj.kabsch_sander(traj)
    n_hbonds = np.array(
        [int((m.data < HBOND_ENERGY_CUTOFF).sum()) for m in hbond_mats], dtype=np.float32
    ).reshape(-1, 1)

    block = np.concatenate([total_sasa, n_hbonds, sasa_res], axis=1)
    names = ["total_sasa", "n_hydrogen_bonds"]
    names += [f"sasa_res{i}" for i in range(sasa_res.shape[1])]
    return block, names, sasa_res


# --- driver ------------------------------------------------------------------
def _resolve_reference(
    traj_path: str, top_path: str | None, reference_path: str | None, protein_idx: np.ndarray
) -> mdtraj.Trajectory:
    """Load the RMSD / native-contact reference (user-supplied or first frame)."""
    if reference_path is not None:
        ref = mdtraj.load(reference_path)
        ref = ref.atom_slice(ref.top.select("protein") if len(ref.top.select("protein")) else
                             np.arange(ref.n_atoms))
        return ref[0]
    ref = mdtraj.load_frame(traj_path, 0, top=top_path)
    return ref.atom_slice(protein_idx)


def extract_features(
    traj_path: str,
    top_path: str | None = None,
    reference_path: str | None = None,
    selection: str = "protein",
    stride: int = 1,
    chunk: int = 0,
    features: tuple[str, ...] = BLOCK_ORDER,
    contact_cutoff_nm: float = CONTACT_CUTOFF_NM,
) -> tuple[FeatureResult, TopologyInfo]:
    """Featurize a trajectory, streaming it in chunks to bound memory use.

    Args:
        traj_path: trajectory file (xtc/dcd/h5/pdb/...).
        top_path: topology file; if None, ``traj_path`` is used as its own topology.
        reference_path: structure for RMSD / native contacts; defaults to frame 0.
        selection: mdtraj atom-selection string for the atoms to featurize.
        stride: keep every ``stride``-th frame.
        chunk: frames per I/O chunk; 0 loads the whole trajectory at once.
        features: which feature blocks to compute.
        contact_cutoff_nm: Ca-Ca distance defining a contact / native contact.

    Returns:
        (FeatureResult, TopologyInfo)
    """
    full_top = mdtraj.load_topology(top_path or traj_path)
    protein_idx = full_top.select(selection)
    if len(protein_idx) == 0:
        logger.warning("Selection '%s' is empty; falling back to all atoms.", selection)
        protein_idx = np.arange(full_top.n_atoms)
    sub_top = full_top.subset(protein_idx)

    # Periodicity: only treat the box as periodic if the trajectory carries one.
    probe = mdtraj.load_frame(traj_path, 0, top=top_path)
    periodic = probe.unitcell_lengths is not None
    topo = _detect_topology(sub_top, periodic=periodic)

    reference = _resolve_reference(traj_path, top_path, reference_path, protein_idx)
    ref_dist = mdtraj.compute_distances(reference, topo.ca_pairs, periodic=False)[0]
    native_mask = ref_dist < contact_cutoff_nm

    block_buf: dict[str, list[np.ndarray]] = {b: [] for b in features}
    names: dict[str, list[str]] = {}
    ca_xyz_chunks: list[np.ndarray] = []
    sasa_sum: np.ndarray | None = None
    n_frames = 0

    # NOTE: mdtraj.iterload(chunk=0) silently ignores `stride`, so the
    # whole-trajectory case must go through mdtraj.load instead.
    if chunk and chunk > 0:
        iterator: object = mdtraj.iterload(
            traj_path, top=top_path, chunk=chunk, stride=stride, atom_indices=protein_idx
        )
    else:
        iterator = [mdtraj.load(traj_path, top=top_path, stride=stride,
                                atom_indices=protein_idx)]
    for traj_c in iterator:
        n_frames += traj_c.n_frames
        ca_xyz_chunks.append(traj_c.xyz[:, topo.ca_atom_indices, :].copy())

        distances = None
        if "distances" in features or "contacts" in features or "global" in features:
            distances = _distance_block(traj_c, topo)
        contacts = (_contact_block(distances, contact_cutoff_nm)
                    if distances is not None else None)

        if "dihedrals" in features:
            arr, nm = _dihedral_block(traj_c)
            block_buf["dihedrals"].append(arr)
            names.setdefault("dihedrals", nm)
        if "distances" in features:
            block_buf["distances"].append(distances)
            names.setdefault("distances", topo.pair_labels)
        if "contacts" in features:
            block_buf["contacts"].append(contacts)
            names.setdefault("contacts", [f"contact_{l}" for l in topo.pair_labels])
        if "global" in features:
            arr, nm = _global_block(traj_c, topo, reference, native_mask, contacts)
            block_buf["global"].append(arr)
            names.setdefault("global", nm)
        if "solvation" in features:
            arr, nm, sasa_res = _solvation_block(traj_c)
            block_buf["solvation"].append(arr)
            names.setdefault("solvation", nm)
            sasa_sum = sasa_res.sum(0) if sasa_sum is None else sasa_sum + sasa_res.sum(0)

    if n_frames == 0:
        raise RuntimeError("Trajectory contained no frames after applying stride.")

    result = FeatureResult(n_frames=n_frames)
    for block in features:
        merged = np.concatenate(block_buf[block], axis=0)
        result.blocks[block] = merged
        result.block_names[block] = names.get(block, [f"{block}_{i}" for i in range(merged.shape[1])])

    # Per-residue Ca RMSF: align to frame 0, fluctuate about the mean structure.
    ca_xyz = np.concatenate(ca_xyz_chunks, axis=0)
    ca_traj = mdtraj.Trajectory(ca_xyz, sub_top.subset(topo.ca_atom_indices))
    ca_traj.superpose(ca_traj, frame=0)
    mean_xyz = ca_traj.xyz.mean(axis=0)
    result.rmsf_nm = np.sqrt(
        np.mean(np.sum((ca_traj.xyz - mean_xyz) ** 2, axis=-1), axis=0)
    ).astype(np.float32)

    if sasa_sum is not None:
        result.mean_sasa_nm2 = (sasa_sum / n_frames).astype(np.float32)

    return result, topo


# --- output ------------------------------------------------------------------
def _concatenate(result: FeatureResult) -> tuple[np.ndarray, list[str], dict[str, list[int]]]:
    """Concatenate all blocks into one matrix, tracking column ranges."""
    mats, names, slices, cursor = [], [], {}, 0
    for block in BLOCK_ORDER:
        if block not in result.blocks:
            continue
        mat = result.blocks[block]
        if mat.shape[1] == 0:
            continue
        mats.append(mat)
        names.extend(result.block_names[block])
        slices[block] = [cursor, cursor + mat.shape[1]]
        cursor += mat.shape[1]
    X = np.concatenate(mats, axis=1) if mats else np.zeros((result.n_frames, 0), np.float32)
    return X.astype(np.float32), names, slices


def write_outputs(
    result: FeatureResult,
    topo: TopologyInfo,
    out_dir: str,
    prefix: str,
    output_mode: str,
    config: dict,
) -> None:
    """Write npz / pt / csv / json outputs according to ``output_mode``."""
    os.makedirs(out_dir, exist_ok=True)
    X, feature_names, slices = _concatenate(result)
    name_arr = np.array(feature_names, dtype="U")

    if output_mode in ("tica", "both"):
        npz: dict[str, np.ndarray] = {"X": X, "feature_names": name_arr}
        for block, mat in result.blocks.items():
            npz[block] = mat
            npz[f"{block}_names"] = np.array(result.block_names[block], dtype="U")
        npz["residue_resseq"] = topo.residue_resseq
        npz["residue_resname"] = np.array(topo.residue_resname, dtype="U")
        npz["rmsf_nm"] = result.rmsf_nm
        if result.mean_sasa_nm2 is not None:
            npz["mean_sasa_nm2"] = result.mean_sasa_nm2
        path = os.path.join(out_dir, f"{prefix}_features.npz")
        np.savez_compressed(path, **npz)
        logger.info("Wrote %s  (X shape %s)", path, X.shape)

    if output_mode in ("ml", "both"):
        import torch

        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std_safe = np.where(std < 1e-8, 1.0, std)  # leave constant columns unscaled
        X_norm = ((X - mean) / std_safe).astype(np.float32)
        payload = {
            "X": torch.from_numpy(X_norm),
            "X_raw": torch.from_numpy(X),
            "mean": torch.from_numpy(mean.astype(np.float32)),
            "std": torch.from_numpy(std_safe.astype(np.float32)),
            "feature_names": feature_names,
            "block_slices": slices,
            "rmsf_nm": torch.from_numpy(result.rmsf_nm),
            "sequence": topo.sequence,
            "metadata": config,
        }
        path = os.path.join(out_dir, f"{prefix}_features_ml.pt")
        torch.save(payload, path)
        logger.info("Wrote %s  (X shape %s)", path, tuple(X_norm.shape))

    # Per-block summary CSV.
    summary_path = os.path.join(out_dir, f"{prefix}_summary.csv")
    with open(summary_path, "w") as f:
        f.write("block,n_dims,mean,std,min,max\n")
        for block, mat in result.blocks.items():
            if mat.size == 0:
                f.write(f"{block},0,nan,nan,nan,nan\n")
                continue
            f.write(f"{block},{mat.shape[1]},{mat.mean():.6g},{mat.std():.6g},"
                    f"{mat.min():.6g},{mat.max():.6g}\n")

    # Per-residue CSV.
    res_path = os.path.join(out_dir, f"{prefix}_per_residue.csv")
    with open(res_path, "w") as f:
        f.write("res_index,resseq,resname,rmsf_nm,mean_sasa_nm2\n")
        for i in range(topo.n_residues):
            sasa = result.mean_sasa_nm2[i] if result.mean_sasa_nm2 is not None else float("nan")
            f.write(f"{i},{topo.residue_resseq[i]},{topo.residue_resname[i]},"
                    f"{result.rmsf_nm[i]:.6g},{sasa:.6g}\n")

    # Metadata JSON.
    meta = dict(config)
    meta.update(
        n_frames=result.n_frames,
        n_residues=topo.n_residues,
        sequence=topo.sequence,
        sidechains_present=topo.sidechains_present,
        has_hydrogens=topo.has_hydrogens,
        periodic=topo.periodic,
        total_feature_dim=int(X.shape[1]),
        block_dims={b: int(m.shape[1]) for b, m in result.blocks.items()},
        block_slices=slices,
        units={"distance": "nm", "sasa": "nm^2", "hbond_energy": "kcal/mol",
               "dihedral": "cos/sin"},
    )
    meta_path = os.path.join(out_dir, f"{prefix}_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Wrote %s, %s, %s", summary_path, res_path, meta_path)


# --- CLI ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract MD featurizations from a protein trajectory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--traj", required=True, help="Trajectory file (xtc/dcd/h5/pdb/...).")
    parser.add_argument("--top", default=None, help="Topology file (e.g. PDB). "
                        "If omitted, --traj is used as its own topology.")
    parser.add_argument("--reference", default=None,
                        help="Reference structure for RMSD / native contacts (default: frame 0).")
    parser.add_argument("--out-dir", default=".", help="Output directory.")
    parser.add_argument("--prefix", default=None, help="Output file prefix "
                        "(default: trajectory file stem).")
    parser.add_argument("--selection", default="protein",
                        help="mdtraj atom-selection string for atoms to featurize.")
    parser.add_argument("--stride", type=int, default=1, help="Keep every Nth frame.")
    parser.add_argument("--chunk", type=int, default=0,
                        help="Frames per I/O chunk; 0 loads the whole trajectory at once.")
    parser.add_argument("--features", default="all",
                        help=f"Comma-separated subset of {','.join(BLOCK_ORDER)}, or 'all'.")
    parser.add_argument("--contact-cutoff", type=float, default=CONTACT_CUTOFF_NM,
                        help="Ca-Ca distance (nm) defining a contact / native contact.")
    parser.add_argument("--output", choices=("tica", "ml", "both"), default="both",
                        help="Which outputs to write.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    features = (BLOCK_ORDER if args.features == "all"
                else tuple(f.strip() for f in args.features.split(",")))
    unknown = set(features) - set(BLOCK_ORDER)
    if unknown:
        parser.error(f"Unknown feature blocks: {sorted(unknown)}. Choose from {BLOCK_ORDER}.")

    prefix = args.prefix or os.path.splitext(os.path.basename(args.traj))[0]
    config = {
        "traj": args.traj, "top": args.top, "reference": args.reference,
        "selection": args.selection, "stride": args.stride, "chunk": args.chunk,
        "features": list(features), "contact_cutoff_nm": args.contact_cutoff,
    }

    result, topo = extract_features(
        traj_path=args.traj, top_path=args.top, reference_path=args.reference,
        selection=args.selection, stride=args.stride, chunk=args.chunk, features=features,
        contact_cutoff_nm=args.contact_cutoff,
    )
    write_outputs(result, topo, args.out_dir, prefix, args.output, config)
    logger.info("Done: %d frames, %d residues, sidechains=%s.",
                result.n_frames, topo.n_residues, topo.sidechains_present)


if __name__ == "__main__":
    main()
