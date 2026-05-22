# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Extract BioEmu's internal per-residue activations via forward hooks.

BioEmu's diffusion score model (:class:`bioemu.models.DiGConditionalScoreModel`)
processes a protein through a stack of eight :class:`SAEncoderLayer` blocks that
operate on a per-residue representation ``x1d`` of width 512. This module taps
that representation -- without modifying the model -- by registering forward
hooks on the encoder layers while the denoiser runs, then pools the captured
per-residue activations into a fixed-width vector with a configurable pooler.

Captured representations
  * layer ``-1``  - the pre-encoder ``x1d`` fed into layer 0 (single-repr
                    projection + time embedding), captured via a forward-pre hook.
  * layer ``i`` (0..7) - the output ``x1d`` of encoder layer ``i``.

For each requested ``(layer, diffusion_time)`` pair the hook performs an online
nearest-snap: across all denoiser timesteps it keeps only the activation whose
visited time is closest to the requested time. This bounds memory to
``num_layers * num_times`` tensors regardless of the number of denoiser steps.

Tensor shapes
  * per-residue activations: ``[L, 512]`` (system-size dependent; ``L`` residues).
  * pooled features: ``[512]`` (fixed width, pooled over the residue axis).

Example
  python -m bioemu.extract_internal_features --sequence GSHMKERAERA \\
      --layers -1,3,7 --times 0.99,0.5,0.1 --pooling mean --out feats.pt
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.data.batch import Batch

from bioemu.denoiser import dpm_solver, heun_denoiser
from bioemu.models import DiGConditionalScoreModel
from bioemu.model_utils import load_model, load_sdes, maybe_download_checkpoint

logger = logging.getLogger(__name__)

# --- runtime guards (fail loudly if BioEmu's architecture changes) -----------
EXPECTED_NUM_LAYERS = 8
EXPECTED_D_MODEL = 512

# Default denoiser integration parameters, matching config/denoiser/{dpm,heun}.yaml.
_DEFAULT_MAX_T = 0.99
_DEFAULT_EPS_T = 0.001
_HEUN_NOISE = 1.0


# --- residue pooling ---------------------------------------------------------
class ResiduePooler(nn.Module):
    """Abstract base class for pooling over the residue axis.

    A pooler maps a per-residue representation ``[..., L, D]`` to a fixed-width
    representation ``[..., D]`` by reducing the residue (second-to-last) axis.
    """

    def forward(self, x: Tensor) -> Tensor:
        """Pool over the residue axis.

        Args:
            x: Per-residue activations of shape ``[..., L, D]``.

        Returns:
            Pooled features of shape ``[..., D]``.
        """
        raise NotImplementedError


class MeanPooler(ResiduePooler):
    """Parameter-free mean pooling over the residue axis."""

    def forward(self, x: Tensor) -> Tensor:
        """Average ``[..., L, D]`` over residues to ``[..., D]``."""
        return x.mean(dim=-2)


class AttentionPooler(ResiduePooler):
    """Single-head additive attention pooling over the residue axis.

    A linear layer scores each residue; scores are softmax-normalised over the
    residue axis and used to take a weighted sum. Parameters are untrained by
    default but the module is fully functional.
    """

    def __init__(self, d_model: int = EXPECTED_D_MODEL):
        """Args:
        d_model: Feature dimension ``D`` of the per-residue representation.
        """
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Attention-weighted sum of ``[..., L, D]`` over residues to ``[..., D]``."""
        scores = self.score(x)  # [..., L, 1]
        weights = torch.softmax(scores, dim=-2)  # [..., L, 1]
        return (weights * x).sum(dim=-2)  # [..., D]


class LearnedQueryPooler(ResiduePooler):
    """Learned-query scaled dot-product attention pooling over residues.

    A trainable query vector attends over the residue axis using scaled
    dot-product attention after projecting the residues to a hidden space.
    Exposes trainable parameters intended for downstream fine-tuning.
    """

    def __init__(self, d_model: int = EXPECTED_D_MODEL, hidden: int = EXPECTED_D_MODEL):
        """Args:
        d_model: Feature dimension ``D`` of the per-residue representation.
        hidden: Width of the key/value projection used for attention.
        """
        super().__init__()
        self.key = nn.Linear(d_model, hidden)
        self.value = nn.Linear(d_model, d_model)
        self.query = nn.Parameter(torch.randn(hidden) / hidden**0.5)
        self._scale = hidden**0.5

    def forward(self, x: Tensor) -> Tensor:
        """Learned-query attention over ``[..., L, D]`` to ``[..., D]``."""
        keys = self.key(x)  # [..., L, hidden]
        values = self.value(x)  # [..., L, D]
        scores = (keys @ self.query) / self._scale  # [..., L]
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # [..., L, 1]
        return (weights * values).sum(dim=-2)  # [..., D]


def make_pooler(method: str, d_model: int = EXPECTED_D_MODEL) -> ResiduePooler:
    """Construct a residue pooler by name.

    Args:
        method: One of ``"mean"``, ``"attention"``, ``"learned"``.
        d_model: Feature dimension ``D`` (used by parametric poolers).

    Returns:
        A :class:`ResiduePooler` instance.

    Raises:
        ValueError: If ``method`` is not recognised.
    """
    if method == "mean":
        return MeanPooler()
    if method == "attention":
        return AttentionPooler(d_model)
    if method == "learned":
        return LearnedQueryPooler(d_model)
    raise ValueError(f"Unknown pooling method {method!r}. Choose from 'mean', 'attention', 'learned'.")


# --- result container --------------------------------------------------------
@dataclass
class InternalFeature:
    """One captured internal representation for a (layer, time) request.

    Attributes:
        layer: Requested layer index; ``-1`` denotes the pre-encoder ``x1d``.
        diffusion_time: Requested diffusion time in ``[0, 1]``.
        snapped_time: Nearest denoiser timestep actually visited and captured.
        per_residue: Per-residue activations, shape ``[L, 512]`` (CPU tensor).
        pooled: Residue-pooled features, shape ``[512]`` (CPU tensor).
    """

    layer: int
    diffusion_time: float
    snapped_time: float
    per_residue: Tensor
    pooled: Tensor


# --- hook bookkeeping --------------------------------------------------------
@dataclass
class _Capture:
    """Best (nearest-snap) capture for one (layer, requested_time) pair."""

    per_residue: Tensor  # [B, L, 512] on CPU
    snapped_time: float
    distance: float  # |snapped_time - requested_time|


# --- public API --------------------------------------------------------------
def extract_bioemu_internal_features(
    sequence: str,
    layers: Sequence[int] = (-1, 3, 7),
    diffusion_times: Sequence[float] = (0.99, 0.5, 0.1),
    pooling: str = "mean",
    *,
    model_name: str = "bioemu-v1.1",
    batch_size: int = 1,
    num_denoiser_steps: int = 50,
    denoiser_type: str = "dpm",
    device: str | torch.device | None = None,
    cache_embeds_dir: str | None = None,
    msa_file: str | None = None,
    seed: int | None = None,
    pooler: ResiduePooler | None = None,
    return_pooler: bool = False,
) -> (
    dict[tuple[int, float], InternalFeature]
    | tuple[dict[tuple[int, float], InternalFeature], ResiduePooler]
):
    """Extract BioEmu's internal per-residue activations via forward hooks.

    Runs the BioEmu denoiser once on ``sequence`` while hooks capture the
    encoder's per-residue representation ``x1d`` (width 512) at the requested
    layers. For each ``(layer, time)`` pair the activation visited at the
    denoiser timestep closest to the requested time is kept (online
    nearest-snap), then pooled over residues into a fixed-width vector.

    Args:
        sequence: Amino-acid sequence (single-letter codes).
        layers: Layer indices to capture; each must be in ``{-1, 0..7}`` where
            ``-1`` is the pre-encoder ``x1d`` (input to layer 0).
        diffusion_times: Diffusion times in ``[0, 1]`` to snap to. Each is
            matched to the nearest denoiser timestep actually visited.
        pooling: Pooler name when ``pooler`` is not given
            (``"mean"`` | ``"attention"`` | ``"learned"``).
        model_name: Pretrained BioEmu checkpoint name to download/load.
        batch_size: Number of prior samples to run together. With
            ``batch_size > 1`` captured activations are averaged over the batch
            so returned shapes remain ``[L, 512]`` / ``[512]``.
        num_denoiser_steps: Number of denoiser integration steps ``N``.
        denoiser_type: ``"dpm"`` or ``"heun"``.
        device: Torch device; defaults to CUDA if available, else CPU.
        cache_embeds_dir: Directory for cached ColabFold embeddings.
        msa_file: Optional path to a precomputed MSA A3M file.
        seed: If given, seeds ``torch`` before prior sampling for reproducibility.
        pooler: Pre-built pooler to reuse for all outputs; overrides ``pooling``.
        return_pooler: If ``True``, also return the pooler instance.

    Returns:
        A dict keyed by ``(layer, requested_time)`` mapping to
        :class:`InternalFeature`. If ``return_pooler`` is ``True``, returns
        ``(dict, pooler)``.

    Raises:
        ValueError: On invalid layer indices, diffusion times, or denoiser type.
        RuntimeError: If the loaded model's architecture does not match the
            expected number of encoder layers or feature dimension.
    """
    layers = list(layers)
    diffusion_times = list(diffusion_times)

    # --- validate requests ---------------------------------------------------
    for layer in layers:
        if layer != -1 and not (0 <= layer < EXPECTED_NUM_LAYERS):
            raise ValueError(
                f"Layer index {layer} invalid; must be -1 or in 0..{EXPECTED_NUM_LAYERS - 1}."
            )
    for t in diffusion_times:
        if not (0.0 <= t <= 1.0):
            raise ValueError(f"Diffusion time {t} invalid; must be in [0, 1].")
    if denoiser_type not in ("dpm", "heun"):
        raise ValueError(f"denoiser_type must be 'dpm' or 'heun', got {denoiser_type!r}.")

    device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # --- load model + SDEs ---------------------------------------------------
    ckpt_path, model_config_path = maybe_download_checkpoint(model_name=model_name)
    score_model: DiGConditionalScoreModel = load_model(ckpt_path, model_config_path)
    sdes = load_sdes(model_config_path=model_config_path)
    score_model = score_model.to(device).eval()

    # --- architecture guards -------------------------------------------------
    encoder = score_model.model_nn.st_module.encoder
    encoder_layers = encoder.layers
    if len(encoder_layers) != EXPECTED_NUM_LAYERS:
        raise RuntimeError(
            f"Expected {EXPECTED_NUM_LAYERS} encoder layers but found "
            f"{len(encoder_layers)} at score_model.model_nn.st_module.encoder.layers; "
            "BioEmu's architecture may have changed."
        )
    if score_model.model_nn.d_model != EXPECTED_D_MODEL:
        raise RuntimeError(
            f"Expected d_model={EXPECTED_D_MODEL} but model reports "
            f"{score_model.model_nn.d_model}; BioEmu's architecture may have changed."
        )

    # --- pooler --------------------------------------------------------------
    if pooler is None:
        pooler = make_pooler(pooling, EXPECTED_D_MODEL)
    pooler = pooler.to(device)

    # --- build input batch ---------------------------------------------------
    # Imported lazily: bioemu.sample pulls in the PDB-conversion / openfold stack,
    # which is unrelated to feature extraction and need not be present to use the
    # poolers or import this module.
    from bioemu.sample import get_context_chemgraph

    if seed is not None:
        torch.manual_seed(seed)
    context_chemgraph = get_context_chemgraph(
        sequence=sequence, cache_embeds_dir=cache_embeds_dir, msa_file=msa_file
    )
    batch = Batch.from_data_list([context_chemgraph] * batch_size).to(device)

    # --- shared hook state ---------------------------------------------------
    # current_time holds the scalar diffusion time of the in-flight score call.
    current_time: dict[str, float] = {"t": float("nan")}
    # captures[(layer, requested_time)] = best _Capture so far.
    captures: dict[tuple[int, float], _Capture] = {}

    def _snap(layer: int, activation: Tensor) -> None:
        """Online nearest-snap of one [B, L, D] activation across all requested times."""
        t_now = current_time["t"]
        act = activation.detach().to("cpu")
        for req_t in diffusion_times:
            dist = abs(t_now - req_t)
            prev = captures.get((layer, req_t))
            if prev is None or dist < prev.distance:
                captures[(layer, req_t)] = _Capture(
                    per_residue=act, snapped_time=t_now, distance=dist
                )

    def _time_pre_hook(_module, args):  # noqa: ANN001 - torch hook signature
        """Stash the scalar diffusion time of the current score call (t is args[1])."""
        t = args[1]
        current_time["t"] = float(t.reshape(-1)[0])

    def _make_layer_hook(layer: int):
        def hook(_module, _inp, output):  # noqa: ANN001 - torch hook signature
            _snap(layer, output)

        return hook

    def _pre_encoder_hook(_module, args):  # noqa: ANN001 - torch hook signature
        _snap(-1, args[0])

    # --- register hooks, run denoiser, always clean up -----------------------
    handles: list[torch.utils.hooks.RemovableHandle] = []
    try:
        handles.append(score_model.register_forward_pre_hook(_time_pre_hook))
        for layer in layers:
            if layer == -1:
                handles.append(encoder_layers[0].register_forward_pre_hook(_pre_encoder_hook))
            else:
                handles.append(encoder_layers[layer].register_forward_hook(_make_layer_hook(layer)))

        with torch.no_grad():
            if denoiser_type == "dpm":
                dpm_solver(
                    sdes=sdes,
                    batch=batch,
                    N=num_denoiser_steps,
                    score_model=score_model,
                    max_t=_DEFAULT_MAX_T,
                    eps_t=_DEFAULT_EPS_T,
                    device=device,
                )
            else:
                heun_denoiser(
                    sdes=sdes,
                    N=num_denoiser_steps,
                    eps_t=_DEFAULT_EPS_T,
                    max_t=_DEFAULT_MAX_T,
                    device=device,
                    batch=batch,
                    score_model=score_model,
                    noise=_HEUN_NOISE,
                )
    finally:
        for handle in handles:
            handle.remove()

    # --- assemble results ----------------------------------------------------
    results: dict[tuple[int, float], InternalFeature] = {}
    for layer in layers:
        for req_t in diffusion_times:
            capture = captures.get((layer, req_t))
            if capture is None:
                raise RuntimeError(
                    f"No activation captured for layer {layer} at time {req_t}; "
                    "the denoiser may not have called the score model."
                )
            per_residue = capture.per_residue  # [B, L, 512]
            if per_residue.shape[-1] != EXPECTED_D_MODEL:
                raise RuntimeError(
                    f"Captured channel dim {per_residue.shape[-1]} != expected "
                    f"{EXPECTED_D_MODEL} for layer {layer}; architecture may have changed."
                )
            # Reduce the batch axis: [B, L, 512] -> [L, 512] (average over batch).
            per_residue = per_residue.mean(dim=0)  # [L, 512]
            with torch.no_grad():
                pooled = pooler(per_residue.to(device)).to("cpu")  # [512]
            results[(layer, req_t)] = InternalFeature(
                layer=layer,
                diffusion_time=req_t,
                snapped_time=capture.snapped_time,
                per_residue=per_residue,
                pooled=pooled,
            )

    if return_pooler:
        return results, pooler
    return results


# --- CLI ---------------------------------------------------------------------
def _parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract BioEmu internal per-residue activations via forward hooks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sequence", required=True, help="Amino-acid sequence (single-letter).")
    parser.add_argument("--layers", default="-1,3,7",
                        help="Comma-separated layer indices; -1 = pre-encoder x1d, 0..7 = layers.")
    parser.add_argument("--times", default="0.99,0.5,0.1",
                        help="Comma-separated diffusion times in [0, 1] to snap to.")
    parser.add_argument("--pooling", default="mean", choices=("mean", "attention", "learned"),
                        help="Residue pooling method.")
    parser.add_argument("--model-name", default="bioemu-v1.1", help="Pretrained checkpoint name.")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Number of prior samples; activations are averaged over the batch.")
    parser.add_argument("--steps", type=int, default=50, help="Number of denoiser steps N.")
    parser.add_argument("--denoiser", default="dpm", choices=("dpm", "heun"), help="Denoiser type.")
    parser.add_argument("--device", default=None, help="Torch device (default: cuda if available).")
    parser.add_argument("--cache-embeds-dir", default=None,
                        help="Directory for cached ColabFold embeddings.")
    parser.add_argument("--msa-file", default=None, help="Optional precomputed MSA A3M file.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for prior sampling.")
    parser.add_argument("--out", default=None,
                        help="Optional .pt path to torch.save the captured features.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    layers = _parse_int_list(args.layers)
    times = _parse_float_list(args.times)

    features = extract_bioemu_internal_features(
        sequence=args.sequence,
        layers=layers,
        diffusion_times=times,
        pooling=args.pooling,
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_denoiser_steps=args.steps,
        denoiser_type=args.denoiser,
        device=args.device,
        cache_embeds_dir=args.cache_embeds_dir,
        msa_file=args.msa_file,
        seed=args.seed,
    )

    # Shape table.
    print(f"{'layer':>6} {'req_t':>8} {'snap_t':>8} {'per_residue':>16} {'pooled':>10}")
    for layer in layers:
        for t in times:
            feat = features[(layer, t)]
            print(
                f"{feat.layer:>6} {feat.diffusion_time:>8.4f} {feat.snapped_time:>8.4f} "
                f"{str(tuple(feat.per_residue.shape)):>16} {str(tuple(feat.pooled.shape)):>10}"
            )

    if args.out is not None:
        payload = {
            f"layer{feat.layer}_t{feat.diffusion_time}": {
                "per_residue": feat.per_residue,
                "pooled": feat.pooled,
                "snapped_time": feat.snapped_time,
            }
            for feat in features.values()
        }
        torch.save(payload, args.out)
        logger.info("Wrote %s (%d feature entries).", args.out, len(payload))


if __name__ == "__main__":
    main()
