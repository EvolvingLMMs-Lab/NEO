"""Which camera each arm's suffix block is written against, drawn per sample.

Qwen-RobotManip eq (5) picks the reference per sample, and for two arms it
names two ways: both arms share one third-person camera, or each arm takes its
own wrist camera. Both are drawn here, uniformly, per sample.

The draw is a pure function of the sample index and the run seed, spending the
uniforms of that draw's own stream past the three its window spent (extra
positions 0 to 2 at most, before the history draw's), so a run that resumes
draws what it would have drawn. The pool it draws from is the usable cameras:
a picture at the step and a pose that passed its audit there, both decided by
the caller before the draw. So a sample
with any usable camera is in camera mode, and robot mode is what an arm no
usable camera can serve gets.
"""
from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

import torch

from .mixture import extra_uniforms

# How the arms of one sample share their references. SHARED is the only one a
# single-armed directory has, and the only one a directory whose arms do not
# share a base frame may take: two arms under one rotation is a statement the
# manifest has to make.
SHARED = "shared"
PER_ARM = "per_arm"
STRATEGIES = (SHARED, PER_ARM)


@dataclass(frozen=True)
class ReferenceCamera:
    """One camera an arm's block may be written against.

    key and slot are the manifest's own; arm_slot is the action slot of the arm
    the camera is mounted on, or None for a camera that does not move with an
    arm. Only a camera naming an arm can serve that arm under PER_ARM.
    """

    key: str
    slot: int
    arm_slot: int | None


@dataclass(frozen=True)
class Reference:
    """The camera drawn for each action slot, and the strategy that chose it.

    cameras is keyed by action slot. A slot with no entry is in robot frame,
    which is what a directory with no eligible camera has for every slot.
    """

    strategy: str
    cameras: dict[int, ReferenceCamera]

    def camera_for(self, slot: int) -> ReferenceCamera | None:
        """The camera one action slot is written against, or None for robot."""
        return self.cameras.get(int(slot))


def strategies_for(
    cameras: tuple[ReferenceCamera, ...],
    arm_slots: tuple[int, ...],
    *,
    common_base_frame: bool,
) -> tuple[str, ...]:
    """Which strategies this directory can offer, in STRATEGIES order.

    SHARED needs one camera and, for more than one arm, the manifest's word
    that the arms share a base frame: without it two arms would be written
    against one rotation nobody checked applies to both. PER_ARM needs every
    arm to name a camera of its own, so a directory where only one arm carries
    a wrist camera cannot take it.
    """
    offered = []
    if cameras and (len(arm_slots) == 1 or common_base_frame):
        offered.append(SHARED)
    mounted = {slot: [one for one in cameras if one.arm_slot == slot]
               for slot in arm_slots}
    if len(arm_slots) > 1 and all(mounted[slot] for slot in arm_slots):
        offered.append(PER_ARM)
    return tuple(offered)


def offered_strategies(
    pool: tuple[ReferenceCamera, ...],
    arm_slots: tuple[int, ...],
    *,
    common_base_frame: bool,
) -> tuple[str, ...]:
    """The strategies one sample's usable pool offers, in STRATEGIES order.

    strategies_for, the ones that give every arm a camera, whenever it offers
    any. When it offers none and some arm's own wrist is usable (two arms, no
    shared base frame, the other arm's wrist gone), PER_ARM alone, for the arms
    it can serve: no camera can serve the others, so they alone are in the
    robot frame. PER_ARM is never offered short of an arm beside SHARED, which
    would put that arm in the robot frame while a shared camera could serve it.
    """
    offered = strategies_for(pool, arm_slots, common_base_frame=common_base_frame)
    if offered or len(arm_slots) < 2:
        return offered
    if any(one.arm_slot in arm_slots for one in pool):
        return (PER_ARM,)
    return ()


def present_pool(
    cameras: tuple[ReferenceCamera, ...], present: Collection[str]
) -> tuple[ReferenceCamera, ...]:
    """The posed cameras that have a picture at one step, in slot order.

    present is the reader's own list of the cameras it returned a frame for at
    that step, so what comes back is the intersection of what the manifest
    poses and what the sample will actually show. A source that records no
    per-camera flag hands back every camera of the step and this returns the
    whole pool, which is what the draw took before such a flag existed.
    """
    keys = frozenset(present)
    return tuple(sorted((one for one in cameras if one.key in keys),
                        key=lambda one: one.slot))


def draw_reference(
    cameras: tuple[ReferenceCamera, ...],
    arm_slots: tuple[int, ...],
    index: int,
    *,
    seed: int,
    common_base_frame: bool,
    present: Collection[str],
    usable: Collection[str],
) -> Reference:
    """One sample index -> the camera each action slot is written against.

    Uniform over the strategies the usable pool offers (offered_strategies),
    then uniform over the cameras that strategy allows: under SHARED one
    camera for every arm, under PER_ARM each arm's own usable wrist. Uniform
    because Qwen-RobotManip states both ways and weights neither. One uniform
    for the strategy and then one per arm, spent in slot order whichever
    strategy was drawn, so the count a sample spends never depends on the
    branch, and a pool where every camera is usable draws what it drew before
    the audit moved ahead of the draw.

    present and usable are required: a caller that forgets either writes
    samples against a camera it has no picture of, or whose pose failed its
    audit, then drops that arm to the robot frame while another camera passes.
    usable is the cameras whose pose passed the audit at this step; the audit
    is per camera, so one usable camera is usable for every arm SHARED gives
    it. A pool that offers nothing puts every slot in the robot frame, which
    the caller counts. index is the draw's stream number (mixture.Draw), so a
    window the sampler hands out again on a later pass draws its cameras afresh.
    """
    pool = present_pool(present_pool(cameras, present), usable)
    offered = offered_strategies(pool, arm_slots, common_base_frame=common_base_frame)
    if not offered:
        return Reference(SHARED, {})
    uniform = extra_uniforms(index, seed=seed, count=1 + len(arm_slots))
    strategy = offered[_pick(uniform[0], len(offered))]
    if strategy == SHARED:
        drawn = pool[_pick(uniform[1], len(pool))]
        return Reference(SHARED, {slot: drawn for slot in arm_slots})
    chosen: dict[int, ReferenceCamera] = {}
    for position, slot in enumerate(sorted(arm_slots)):
        mounted = tuple(one for one in pool if one.arm_slot == slot)
        if mounted:
            chosen[slot] = mounted[_pick(uniform[position + 1], len(mounted))]
    return Reference(PER_ARM, chosen)


def _pick(uniform: torch.Tensor, count: int) -> int:
    """One uniform -> an index below count, the way mixture picks an episode."""
    return min(int(float(uniform) * count), count - 1)
