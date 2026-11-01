"""Explicit post-train frame support. Generic loaders retain their existing draw."""
from __future__ import annotations

import json
from dataclasses import replace

from .conditioning import parse_frame_ref
from .mixture import name_key


def choose(choices, *, seed, directory, episode_id, sample=None):
    """Stable across workers, row order, passes and resume; no Python hash/RNG state."""
    identity = json.dumps([int(seed), directory, int(episode_id), sample], separators=(",", ":"))
    return tuple(choices[name_key(identity) % len(choices)])


def configure(manifest, mode, frame_set, draw, pin_keeps_other_poses):
    if mode not in ("wrist", "random") or draw not in ("episode", "sample"):
        raise ValueError("frame_mode must be wrist|random; frame_draw must be episode|sample")
    if not frame_set or (mode == "wrist" and not pin_keeps_other_poses):
        raise ValueError("post-train needs a resolved frame_set; wrist needs pin_keeps_other_poses")
    entries = []
    for entry in manifest.entries:
        keys = entry.embodiment.keys
        for key in keys:
            choices = frame_set.get(key, ())
            if not choices or (mode == "wrist" and len(choices) != 1):
                raise ValueError(f"{entry.name}: no unambiguous {mode} frame_set for {key}")
            cameras = {one.slot: one for one in entry.reference_cameras()}
            for frames in choices:
                if len(frames) != len(entry.arms):
                    raise ValueError(f"{entry.name}: frame tuple must name every arm")
                for arm, spelling in zip(entry.arms, frames):
                    kind, slot = parse_frame_ref(spelling)
                    if kind == "camera":
                        if slot is None or slot not in cameras:
                            raise ValueError(f"{entry.name}: {spelling} needs an explicit posed slot")
                        camera = cameras[slot]
                        if len(entry.arms) > 1 and not entry.common_base_frame:
                            if camera.arm_slot != arm.spec.slot:
                                raise ValueError(f"{entry.name}: {spelling} belongs to another arm")
                    elif mode == "wrist":
                        raise ValueError("wrist frame_set cannot contain robot")
            if mode == "random" and tuple("robot" for _ in entry.arms) not in map(tuple, choices):
                raise ValueError("random frame_set must cover the robot frame")
        # A single-arm fixed pin also exposes its resolved spelling to loader audits.
        if mode == "wrist" and len(entry.arms) == 1:
            pins = {frame_set[k][0][0] for k in keys}
            if len(pins) != 1:
                raise ValueError(f"{entry.name}: keys {keys} state different wrist pins {sorted(pins)}")
            entry = replace(entry, frame_ref=pins.pop())
        entries.append(entry)
    return replace(manifest, entries=tuple(entries))
