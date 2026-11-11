"""History at H = 1: one earlier step p per draw, the no-history form, and the state noise.

A draw at step t takes p uniformly on 0 .. t - MIN_GAP, or the no-history
form when t is too early, its episode is blocked, the dropout uniform fires,
or (decided by the dataset, which sees the cameras) no view has a picture at
p. That form masks the earlier views and feeds the start token a zero state.

Every choice spends uniforms from mixture.extra_uniforms at the positions
neo_vla.constants names, keyed by the draw's stream number and the seed, so a
resumed run makes the same choices and a replayed window redraws them. The noise is
training-only and lives on measured tokens' valid dims.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..constants import (
    BODY_DIM,
    BODY_NOISE_WIDTH,
    EXTRA_BODY_NOISE_FIRST,
    EXTRA_DROPOUT,
    EXTRA_HISTORY_STEP,
    EXTRA_NOISE_FIRST,
    EXTRA_NOISE_SWITCH,
    EXTRA_UNIFORMS,
    HISTORY_SIZE,
    MIN_GAP,
    N_ROLES,
    NO_HISTORY_FORM,
    NOISE_APERTURE,
    NOISE_HAND,
    NOISE_JOINTS,
    NOISE_POSITION,
    NOISE_ROTATION,
    NOISE_WIDTH,
    SAMPLING,
    STATE_APERTURE,
    STATE_DIM,
    STATE_HAND,
    STATE_JOINTS,
    STATE_MEASURED,
    STATE_POSITION,
    STATE_ROTATION,
    UNIFORM_CEIL,
    UNIFORM_FLOOR,
)
from .mixture import extra_uniforms

__all__ = [
    "HISTORY_SIZE", "MIN_GAP", "SAMPLING", "NO_HISTORY_FORM", "EXTRA_UNIFORMS",
    "EXTRA_HISTORY_STEP", "EXTRA_DROPOUT", "EXTRA_NOISE_SWITCH", "EXTRA_NOISE_FIRST",
    "UNIFORM_FLOOR", "UNIFORM_CEIL", "NO_HISTORY_REASONS", "HistoryDraw",
    "draw_history_step", "history_uniforms", "decide_history", "noise_normals",
    "body_noise_normals", "noise_dims", "noise_scale", "state_noise", "body_state_noise",
    "check_history_config",
]

# Why a draw carries no history, in the order decide_history tests them; the
# dataset adds no_camera after reading which views have a picture at p.
NO_HISTORY_REASONS = ("short", "blocked", "dropout", "no_camera")

# The three state_noise settings a training data group states, and nothing else.
NOISE_KEYS = ("sigma", "rot_sigma_rad", "prob")


@dataclass(frozen=True)
class HistoryDraw:
    """The history step a draw uses, or None with the reason it has none."""

    step: int | None
    reason: str | None

    def __post_init__(self) -> None:
        if (self.step is None) == (self.reason is None):
            raise ValueError(
                f"history draw has step {self.step} and reason {self.reason!r}.\n"
                "Expected exactly one: a step, or the reason there is none.\n"
                "Build it with decide_history."
            )
        if self.reason is not None and self.reason not in NO_HISTORY_REASONS:
            raise ValueError(
                f"history reason is {self.reason!r}, expected one of {NO_HISTORY_REASONS}.\n"
                "Each no-history draw names why it has none.\n"
                "Add the reason to NO_HISTORY_REASONS first."
            )


def draw_history_step(t: int, u: float) -> int | None:
    """The history step for a draw at t: None when t < MIN_GAP, else uniform on 0 .. t - MIN_GAP.

    p = min(floor(u (t - MIN_GAP + 1)), t - MIN_GAP), the same rule mixture
    uses for a start inside an episode, so the gap t - p is uniform on
    MIN_GAP .. t and never shorter than the smallest deployment horizon.
    """
    step, uniform = int(t), float(u)
    if not 0.0 <= uniform < 1.0:
        raise ValueError(
            f"the history uniform is {uniform}.\n"
            "Expected a value in [0, 1), as torch.rand gives.\n"
            "Pass mixture.extra_uniforms at EXTRA_HISTORY_STEP."
        )
    if step < 0:
        raise ValueError(
            f"the draw's step is {step}.\n"
            "Expected a step inside the episode, at or above 0.\n"
            "Pass the draw's start."
        )
    if step < MIN_GAP:
        return None
    last = step - MIN_GAP
    return min(int(math.floor(uniform * (last + 1))), last)


def history_uniforms(index: int, *, seed: int) -> torch.Tensor:
    """One draw's EXTRA_UNIFORMS extra uniforms, float64, reference positions first.

    index is the draw's stream number, pass * epoch_samples + window index
    (RobotDataset.stream_number), so a replayed window draws a new history
    step, dropout, noise switch and noise, and a resumed run the same ones.
    """
    return extra_uniforms(int(index), seed=int(seed), count=EXTRA_UNIFORMS)


def decide_history(t: int, uniforms: torch.Tensor, *, dropout: float,
                   blocked: bool) -> HistoryDraw:
    """Rules 1 to 4 of the history draw, bar the camera check at p.

    short when t < MIN_GAP, blocked when the episode takes the no-history form
    on every draw, dropout when the dropout uniform falls below the rate, and
    otherwise the step draw_history_step gives.
    """
    if int(t) < MIN_GAP:
        return HistoryDraw(None, "short")
    if blocked:
        return HistoryDraw(None, "blocked")
    if float(uniforms[EXTRA_DROPOUT]) < float(dropout):
        return HistoryDraw(None, "dropout")
    return HistoryDraw(draw_history_step(int(t), float(uniforms[EXTRA_HISTORY_STEP])), None)


def noise_normals(uniforms: torch.Tensor, *, n_slots: int) -> torch.Tensor:
    """The draw's arm normals [N_ROLES, n_slots, NOISE_WIDTH], float64.

    Role-major, then slot, then the 26 noise columns, from the uniforms at
    EXTRA_NOISE_FIRST up to the body's at EXTRA_BODY_NOISE_FIRST; each is
    clamped away from 0 and 1 before ndtri, which is infinite at both.
    """
    count = N_ROLES * int(n_slots) * NOISE_WIDTH
    if EXTRA_NOISE_FIRST + count > EXTRA_BODY_NOISE_FIRST:
        raise ValueError(
            f"{n_slots} slots need {count} arm normals from position {EXTRA_NOISE_FIRST}, "
            f"past the body's at {EXTRA_BODY_NOISE_FIRST}.\n"
            "Expected two slots, the count the uniform layout is written for.\n"
            "RobotDataset refuses any other count; change the contract first."
        )
    if EXTRA_NOISE_FIRST + count > int(uniforms.shape[0]):
        raise ValueError(
            f"{int(uniforms.shape[0])} uniforms hold no {count} normals after position "
            f"{EXTRA_NOISE_FIRST}.\n"
            f"Expected at least {EXTRA_NOISE_FIRST + count}, as history_uniforms gives.\n"
            "Draw them with history_uniforms."
        )
    block = uniforms[EXTRA_NOISE_FIRST:EXTRA_NOISE_FIRST + count].to(torch.float64)
    normals = torch.special.ndtri(block.clamp(UNIFORM_FLOOR, UNIFORM_CEIL))
    return normals.reshape(N_ROLES, int(n_slots), NOISE_WIDTH)


def body_noise_normals(uniforms: torch.Tensor) -> torch.Tensor:
    """The draw's body normals [N_ROLES, BODY_NOISE_WIDTH], float64, role-major.

    From EXTRA_BODY_NOISE_FIRST, after every arm normal, so the arms' noise
    of a draw is the same whether its key has a body or not.
    """
    count = N_ROLES * BODY_NOISE_WIDTH
    if EXTRA_BODY_NOISE_FIRST + count > int(uniforms.shape[0]):
        raise ValueError(
            f"{int(uniforms.shape[0])} uniforms hold no {count} body normals after position "
            f"{EXTRA_BODY_NOISE_FIRST}.\n"
            f"Expected {EXTRA_UNIFORMS}, as history_uniforms gives.\n"
            "Draw them with history_uniforms."
        )
    block = uniforms[EXTRA_BODY_NOISE_FIRST:EXTRA_BODY_NOISE_FIRST + count].to(torch.float64)
    normals = torch.special.ndtri(block.clamp(UNIFORM_FLOOR, UNIFORM_CEIL))
    return normals.reshape(N_ROLES, BODY_NOISE_WIDTH)


def noise_dims(state_dims: torch.Tensor) -> torch.Tensor:
    """A record's valid dims [..., STATE_DIM] -> the noise columns they make live [..., NOISE_WIDTH].

    The noise has its own order (position 3, rotation vector 3, aperture,
    joints 7, hand 12), which is not the record's (joints first): reading one
    through the other's slices is trap T8. The rotation noise is one 3-vector
    turning the 6D, live only when the whole 6D is valid.
    """
    if int(state_dims.shape[-1]) != STATE_DIM or state_dims.dtype is not torch.bool:
        raise ValueError(
            f"state dims are {tuple(state_dims.shape)} {state_dims.dtype}, "
            f"expected [..., {STATE_DIM}] torch.bool.\n"
            "Pass the sample's state_dims."
        )
    live = torch.zeros((*state_dims.shape[:-1], NOISE_WIDTH), dtype=torch.bool)
    live[..., NOISE_POSITION] = state_dims[..., STATE_POSITION]
    live[..., NOISE_ROTATION] = state_dims[..., STATE_ROTATION].all(dim=-1, keepdim=True)
    live[..., NOISE_APERTURE] = state_dims[..., STATE_APERTURE]
    live[..., NOISE_JOINTS] = state_dims[..., STATE_JOINTS]
    live[..., NOISE_HAND] = state_dims[..., STATE_HAND]
    return live


def noise_scale(*, sigma: float, rot_sigma_rad: float) -> torch.Tensor:
    """[NOISE_WIDTH] float64: rot_sigma_rad on the rotation vector, sigma everywhere else.

    D13: joints and hand reuse sigma (model-input units, as the position is);
    there are no new noise settings.
    """
    scale = torch.full((NOISE_WIDTH,), float(sigma), dtype=torch.float64)
    scale[NOISE_ROTATION] = float(rot_sigma_rad)
    return scale


def _switched(uniforms: torch.Tensor, prob: float) -> bool:
    """Whether this draw's noise is on: one switch for the arms and the body."""
    return float(uniforms[EXTRA_NOISE_SWITCH]) < float(prob)


def state_noise(uniforms: torch.Tensor, codes: torch.Tensor, dims: torch.Tensor, *,
                sigma: float, rot_sigma_rad: float, prob: float) -> torch.Tensor:
    """The pre-scaled arm state perturbations [N_ROLES, n_slots, NOISE_WIDTH] float32.

    Position, aperture, joints and hand get sigma times a normal per dim,
    rotation rot_sigma_rad times a normal per axis (a rotation vector in
    radians, applied on the left of R). All zero when the switch uniform is at
    or above prob, and zero on every (role, slot) whose code is not measured
    and on every noise column whose record dims are not valid, so the zero
    start state, an unknown record, an absent arm and an unread joint never
    move.
    """
    slots = int(codes.shape[-1])
    if tuple(codes.shape) != (N_ROLES, slots):
        raise ValueError(
            f"codes are {tuple(codes.shape)}, expected ({N_ROLES}, slots).\n"
            "One code per role and slot, as build_conditioning writes them.\n"
            "Pass the sample's state_code."
        )
    if tuple(dims.shape) != (N_ROLES, slots, STATE_DIM):
        raise ValueError(
            f"dims are {tuple(dims.shape)}, expected ({N_ROLES}, {slots}, {STATE_DIM}).\n"
            "Pass the sample's state_dims."
        )
    noise = noise_normals(uniforms, n_slots=slots) * noise_scale(
        sigma=sigma, rot_sigma_rad=rot_sigma_rad)
    live = (codes == STATE_MEASURED)[..., None] & noise_dims(dims)
    if not _switched(uniforms, prob):
        live = torch.zeros_like(live)
    return torch.where(live, noise, torch.zeros_like(noise)).to(torch.float32)


def body_state_noise(uniforms: torch.Tensor, codes: torch.Tensor, dims: torch.Tensor, *,
                     sigma: float, rot_sigma_rad: float, prob: float) -> torch.Tensor:
    """The pre-scaled body perturbations [N_ROLES, BODY_DIM] float32.

    sigma times a normal on every valid dim of a measured body record, in
    model-input units like the arms' (D13: rot_sigma_rad is the arms' rotation
    vector's and has no body column; it is taken so one settings mapping
    serves both). The arms' switch: one draw's noise is on or off as a whole.
    """
    del rot_sigma_rad
    if tuple(codes.shape) != (N_ROLES,) or tuple(dims.shape) != (N_ROLES, BODY_DIM):
        raise ValueError(
            f"body codes {tuple(codes.shape)} and dims {tuple(dims.shape)}, expected "
            f"({N_ROLES},) and ({N_ROLES}, {BODY_DIM}).\n"
            "Pass the sample's body_state_code and body_state_dims."
        )
    noise = body_noise_normals(uniforms) * float(sigma)
    live = (codes == STATE_MEASURED)[:, None] & dims.to(torch.bool)
    if not _switched(uniforms, prob):
        live = torch.zeros_like(live)
    return torch.where(live, noise, torch.zeros_like(noise)).to(torch.float32)


def check_history_config(history_dropout, state_noise) -> tuple[float, dict]:
    """The two required training knobs, checked, as (dropout, noise settings).

    A data group without either is refused rather than defaulted, so a stage
    cannot train without the context dropout or the noise and say nothing.
    """
    if history_dropout is None:
        raise ValueError(
            "data.train.history_dropout is not set.\n"
            "Expected the history dropout rate, 0.1 in every official stage.\n"
            "State it in the data group; there is no default."
        )
    if isinstance(history_dropout, bool) or not isinstance(history_dropout, (int, float)) \
            or not 0.0 <= float(history_dropout) < 1.0:
        raise ValueError(
            f"data.train.history_dropout is {history_dropout!r}.\n"
            "Expected a rate in [0, 1).\n"
            "Set it to 0.1, the rate every official stage trains with."
        )
    if state_noise is None:
        raise ValueError(
            "data.train.state_noise is not set.\n"
            f"Expected {{{', '.join(NOISE_KEYS)}}}, 0.02, 0.02 and 0.8 in every official stage.\n"
            "State it in the data group; there is no default."
        )
    settings = {str(key): value for key, value in dict(state_noise).items()}
    if sorted(settings) != sorted(NOISE_KEYS):
        raise ValueError(
            f"data.train.state_noise holds {sorted(settings)}.\n"
            f"Expected exactly {sorted(NOISE_KEYS)}.\n"
            "State the three settings and nothing else."
        )
    for key in NOISE_KEYS:
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(
                f"data.train.state_noise.{key} is {value!r}.\n"
                "Expected a finite number at or above 0.\n"
                "Set sigma 0.02, rot_sigma_rad 0.02, prob 0.8."
            )
        settings[key] = float(value)
    if settings["prob"] > 1.0:
        raise ValueError(
            f"data.train.state_noise.prob is {settings['prob']}.\n"
            "Expected a probability in [0, 1].\n"
            "Set it to 0.8: one sample in five stays clean."
        )
    return float(history_dropout), settings
