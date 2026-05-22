# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Demo: extract BioEmu internal per-residue activations for chignolin.

Taps the diffusion score model's per-residue representation ``x1d`` (width 512)
via forward hooks at a couple of encoder layers and diffusion times, pools over
residues, and prints the resulting shapes.

Requirements
  * An SE3nv-style environment where ``torch_geometric`` imports cleanly
    (the base py3.13 / torch2.9 interpreter segfaults on the pyg import).
  * Network access: the first run downloads the ``bioemu-v1.1`` checkpoint from
    HuggingFace and queries the ColabFold MSA server to embed the sequence.
    If either is unavailable the demo prints a message and exits non-fatally.

Run
  PYTHONPATH=/data1/hanqun/bioemu/src \\
      /data1/hanqun/miniconda3/envs/SE3nv/bin/python \\
      examples/extract_internal_features_demo.py
"""

from __future__ import annotations

import logging

from bioemu.extract_internal_features import extract_bioemu_internal_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Chignolin: a 10-residue mini-protein (CLN025 variant).
SEQUENCE = "GYDPETGTWG"
LAYERS = (-1, 7)  # -1 = pre-encoder x1d; 7 = last encoder layer output
DIFFUSION_TIMES = (0.99, 0.1)  # in [0, 1]; snapped to nearest denoiser step


def main() -> None:
    """Extract internal features for chignolin and print per-(layer, time) shapes."""
    try:
        features = extract_bioemu_internal_features(
            sequence=SEQUENCE,
            layers=LAYERS,
            diffusion_times=DIFFUSION_TIMES,
            pooling="mean",
        )
    except Exception as err:  # noqa: BLE001 - the heavy call fetches a model + MSA
        logger.error(
            "Could not extract internal features (model download or MSA server "
            "likely unavailable): %s: %s",
            type(err).__name__,
            err,
        )
        logger.error("Exiting without an end-to-end result; the call wiring is unchanged.")
        return

    # features is keyed by (layer, requested_time) -> InternalFeature.
    print(f"\nInternal features for '{SEQUENCE}' (L={len(SEQUENCE)})")
    print(f"{'layer':>6} {'req_t':>8} {'snap_t':>8} {'per_residue':>16} {'pooled':>10}")
    for layer in LAYERS:
        for t in DIFFUSION_TIMES:
            feat = features[(layer, t)]
            print(
                f"{feat.layer:>6} {feat.diffusion_time:>8.4f} {feat.snapped_time:>8.4f} "
                f"{str(tuple(feat.per_residue.shape)):>16} {str(tuple(feat.pooled.shape)):>10}"
            )
    # Expected: per_residue (10, 512), pooled (512,) for every (layer, time) pair.


if __name__ == "__main__":
    main()
