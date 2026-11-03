"""The auxiliary frequency loss over an action chunk: a band-weighted DCT-II.

After VLANeXt, arXiv 2602.18532, whose transform is src/models/VLANeXt.py
lines 1730 to 1772 and whose flow hook is lines 1622 to 1632. It takes an
orthonormal DCT-II along the chunk axis of the predicted clean action and
weights the coefficients in two bands split at a configured index.

Default off. Nothing here is reached until a run names
model.action.dct.enabled=true, and FlowDecoder.loss then returns exactly what
it returned before this existed, key set and generator draws included.

resolve() turns a band that would contribute nothing into a raise, and makes a
band that reduces to something simpler say so in the line it hands the log.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# What `similarity` may say. The reference also offers "mae" and "cosine",
# VLANeXt.py lines 1762 to 1770; no config it ships selects either, and
# neither is tested here, so naming one raises rather than putting an
# unmeasured code path under a headline run.
SIMILARITIES = ("mse",)


@dataclass(frozen=True)
class FreqLossConfig:
    """What model.action.dct says, before a chunk length resolves it.

    enabled is the only gate. Every other field is read only when it is true,
    so the shipped values may be the recommended ones without arming anything.

    Exactly one of split_index and split states where the low band ends.
    split_index is a count of coefficients and cannot be misread. split is the
    fraction the reference configures, kept so its numbers carry over, and
    resolved by the reference's own rule, int(chunk * split) -- except that a
    fraction needing the reference's max(1, ...) clamp raises here instead of
    being rounded up into a band nobody asked for.
    """

    enabled: bool = False
    weight: float = 0.0
    split_index: int | None = None
    split: float | None = None
    low_weight: float = 0.0
    high_weight: float = 0.0
    similarity: str = "mse"


@dataclass(frozen=True)
class Resolved:
    """A FreqLossConfig against a known chunk: the band, and what it means."""

    chunk: int
    weight: float
    split_index: int
    low_weight: float
    high_weight: float
    similarity: str
    report: str

    def band_weights(self, device: torch.device) -> torch.Tensor:
        """[chunk] float32: low_weight below split_index, high_weight above."""
        weights = torch.full(
            (self.chunk,), self.high_weight, device=device, dtype=torch.float32
        )
        weights[: self.split_index] = self.low_weight
        return weights


def dct_matrix(chunk: int, device: torch.device) -> torch.Tensor:
    """[K, K] float32 orthonormal DCT-II: row k is basis k, column n is time n.

    M[k, n] = c_k cos(pi (n + 1/2) k / K), with c_0 = sqrt(1/K) and c_k =
    sqrt(2/K) above it. Orthonormal, so M M^T is the identity and Parseval is
    exact: the squared coefficients sum to the squared samples. Row 0 is
    sqrt(1/K) times the sum, which is sqrt(K) times the mean, and that identity
    is the whole of what the reference's shipped band penalises.

    Built in float64 and cast down so the basis does not carry the rounding of
    a cosine evaluated in float32 into every coefficient.
    """
    if chunk < 1:
        raise ValueError(f"the chunk is {chunk}, expected at least one step")
    time = torch.arange(chunk, device=device, dtype=torch.float64) + 0.5
    basis = torch.arange(chunk, device=device, dtype=torch.float64)
    matrix = torch.cos((math.pi / chunk) * basis[:, None] * time[None, :])
    matrix[0] *= math.sqrt(1.0 / chunk)
    matrix[1:] *= math.sqrt(2.0 / chunk)
    return matrix.to(torch.float32)


def check_weights(weight: float, low: float, high: float) -> None:
    """Raise on any weight triple that would train and contribute nothing."""
    for name, value in (("weight", weight), ("low_weight", low), ("high_weight", high)):
        if not math.isfinite(value):
            raise ValueError(f"model.action.dct.{name} is {value}, expected finite")
    if weight <= 0.0:
        raise ValueError(
            f"model.action.dct is enabled and its weight is {weight}, so the "
            "term is multiplied by zero and changes nothing.\n"
            "Expected a positive weight on an enabled term.\n"
            "Set model.action.dct.weight above zero, or enabled=false."
        )
    if low < 0.0 or high < 0.0:
        raise ValueError(
            f"model.action.dct band weights are low={low} and high={high}; a "
            "negative weight pays the model to diverge in that band.\n"
            "Expected both at or above zero."
        )
    if low == 0.0 and high == 0.0:
        raise ValueError(
            "model.action.dct is enabled and both band weights are zero, so "
            "every coefficient is multiplied by zero and the term is "
            "identically zero at every step.\n"
            "Expected one band above zero.\n"
            "Set low_weight or high_weight, or enabled=false."
        )


def band_edge(config: FreqLossConfig, chunk: int) -> tuple[int, str]:
    """Where the low band ends, and the phrase a message uses to name it."""
    named = [
        name
        for name, value in (("split_index", config.split_index), ("split", config.split))
        if value is not None
    ]
    if len(named) != 1:
        raise ValueError(
            f"model.action.dct names {named or ['neither split_index nor split']} "
            "for the band edge; exactly one is required.\n"
            "split_index is a count of coefficients. split is the fraction "
            "VLANeXt configures, 0.125, resolved as int(chunk * split)."
        )
    if config.split_index is not None:
        return int(config.split_index), f"split_index={int(config.split_index)}"
    fraction = float(config.split)
    raw = chunk * fraction
    index = int(raw)
    stated = f"split={fraction} over chunk {chunk} gives int({raw:.6g})={index}"
    if index < 1:
        raise ValueError(
            f"model.action.dct.split={fraction} over a chunk of {chunk} puts "
            f"int({raw:.6g})=0 coefficients in the low band.\n"
            "VLANeXt clamps that to 1, VLANeXt.py line 1742, and so trains a "
            "band nobody configured. State the intent here instead.\n"
            f"Use split_index=1, or a split at or above {1.0 / chunk:.6g}."
        )
    return index, stated


def verdict(index: int, chunk: int, low: float, high: float, scale: float) -> str:
    """The tail of the report: what this band reduces to, when it reduces."""
    if index == 1 and high == 0.0:
        return (
            " -- DEGENERATE: only the k=0 coefficient is weighted, and "
            "orthonormal DCT-II's k=0 term is sqrt(K) times the chunk mean, so "
            f"this term is exactly {scale:g} x MSE on the chunk-mean action. "
            "It carries no frequency structure; it is chunk-mean matching, "
            "which is the reduction every VLANeXt config ships."
        )
    if low == high:
        return (
            " -- DEGENERATE: both bands carry one weight, so by Parseval this "
            f"term is exactly {scale:g} x the masked x0-space mean squared "
            "error and carries no frequency structure. An ablation control."
        )
    if low == 0.0:
        return (
            f" -- high-pass: the low band k<{index} carries no weight, so the "
            "chunk mean is not constrained by this term at all."
        )
    return f" -- low band is {index} of {chunk} coefficients, {index / chunk:.1%}"


def resolve(config: FreqLossConfig, chunk: int) -> Resolved:
    """A config plus a chunk length: the band, or a raise that names why not.

    Every raise below is a setting that would have trained without complaint
    and contributed nothing, or contributed something other than what it reads
    as. A dead auxiliary loss that looks armed is the failure this guards, so
    such a setting is never quietly accepted and never quietly repaired.

    Two things about the reference are worth having written down. Every config
    it ships, config/libero_train_config.yaml lines 163 to 168 and its seven
    identical siblings, sets split 0.125 over a chunk of 8, so the band edge is
    int(8 * 0.125) = 1 with high_freq_weight 0.0: only k=0 is weighted, and the
    loss is chunk-mean matching. And 0.1 x 5.0 = 0.5, the weight Table 8 of the
    paper states against the 0.1 its yaml sets; under one weighted band the
    outer weight and the band weight are redundant, since only their product
    reaches the loss, so the two numbers agree rather than conflict.
    """
    if not config.enabled:
        raise ValueError(
            "resolve() was called on a disabled dct config.\n"
            "Expected the caller to check enabled first; a disabled config "
            "has no band to resolve."
        )
    if config.similarity not in SIMILARITIES:
        raise ValueError(
            f"model.action.dct.similarity is {config.similarity!r}; this tree "
            f"implements {', '.join(map(repr, SIMILARITIES))}.\n"
            "VLANeXt offers 'mae' and 'cosine' too and ships neither."
        )
    weight = float(config.weight)
    low = float(config.low_weight)
    high = float(config.high_weight)
    check_weights(weight, low, high)
    index, stated = band_edge(config, chunk)
    if index < 1:
        raise ValueError(
            f"model.action.dct {stated}: the low band holds no coefficients "
            f"and the whole chunk takes high_weight={high}.\n"
            "Expected a band edge of at least one.\n"
            "Say that as low_weight instead, so the log reads as what it does."
        )
    if index > chunk:
        raise ValueError(
            f"model.action.dct {stated} is past the chunk of {chunk}; the band "
            "edge cannot sit beyond the last coefficient."
        )
    if index == chunk and high != low:
        raise ValueError(
            f"model.action.dct {stated} equals the chunk of {chunk}, so the "
            f"high band holds no coefficients and high_weight={high} is read "
            "and then multiplied by nothing.\n"
            f"Expected a band edge below {chunk}, or high_weight={low}."
        )
    head = (
        f"dct aux loss ON: chunk {chunk}, {stated}, coefficients k<{index} "
        f"weighted {low} and k>={index} weighted {high}, outer weight {weight} "
        f"(effective {weight * low:g} low, {weight * high:g} high), "
        f"similarity {config.similarity}"
    )
    return Resolved(
        chunk=chunk,
        weight=weight,
        split_index=index,
        low_weight=low,
        high_weight=high,
        similarity=config.similarity,
        report=head + verdict(index, chunk, low, high, weight * low),
    )


def band_loss(
    residual: torch.Tensor,
    matrix: torch.Tensor,
    band: torch.Tensor,
    denominator: torch.Tensor,
) -> torch.Tensor:
    """A masked residual [B, S, K, A] to the scalar band-weighted DCT loss.

    residual is the predicted clean action minus the target, with the validity
    mask already multiplied in, so a padded step and an invalid column are
    exactly zero. The transform is linear, so transforming the masked
    difference is the difference of the masked transforms, and a masked entry
    contributes zero to every coefficient rather than contributing a padded
    value to all of them.

    denominator is [B]: the per-sample count of valid slot, step and column
    triples, which is the flow loss's own denominator. Sharing it is what keeps
    the two terms comparable in size, and what makes the Parseval identity in
    verdict() exact rather than approximate.

    The transform runs with autocast off. The training step is wrapped in bf16
    autocast, engine/loop.py line 192, and einsum is on the list autocast
    lowers, so this one call would run in bf16 while the flow loss beside it
    stays float32: that loss is built from elementwise squares and sums, which
    autocast leaves alone. A k=0 coefficient is a sum of K terms, and in bf16
    the accumulation error on it would be a large share of the chunk-mean
    residual this term exists to read.
    """
    if residual.dim() != 4:
        raise ValueError(f"residual is {tuple(residual.shape)}, expected [B, S, K, A]")
    chunk = residual.shape[2]
    if tuple(matrix.shape) != (chunk, chunk):
        raise ValueError(
            f"the dct matrix is {tuple(matrix.shape)} and the residual carries "
            f"{chunk} steps; build the matrix for the batch's chunk"
        )
    if tuple(band.shape) != (chunk,):
        raise ValueError(
            f"the band weights are {tuple(band.shape)}, expected ({chunk},)"
        )
    with torch.autocast(device_type=residual.device.type, enabled=False):
        coefficients = torch.einsum(
            "kn,bsna->bska", matrix.float(), residual.float()
        )
    per_sample = ((coefficients * coefficients) * band.view(1, 1, chunk, 1)).sum(
        dim=(1, 2, 3)
    ) / denominator
    return per_sample.mean()
