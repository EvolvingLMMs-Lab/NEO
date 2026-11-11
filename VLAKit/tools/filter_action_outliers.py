"""Rule 3: mask out-of-band or non-finite action targets per group and step.

Uses the released normaliser (embodiments.json) as it is: its per-(key, slot,
column) q01/q99 bounds and band alpha. At each step, a group is out of band
when any of its valid columns is non-finite or outside
[q01 - alpha (q99 - q01), q99 + alpha (q99 - q01)]. Only that group is masked
at that step (arm: end-effector, joints, hand; body: base, posture); the other
groups train. The decision is made by neo_vla.data.embodiment, the same
functions the training loader calls, so bounds are never re-estimated here.

Targets are physical action rows in the loader's column order: 26 per arm
(j0-j6, dx dy dz, drot_x drot_y drot_z, gripper, h0-h11) or 8 for the body.

Usage:
    python tools/filter_action_outliers.py \
        --normalizer metadata/action_pretrain/embodiments.json \
        --dataset <manifest entry name> --key agibot_g1 --slot 0 \
        --targets targets.npy --out in_band.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import filter_common as common
import numpy as np

from neo_vla.data.embodiment import (
    ACTION_GROUPS,
    BODY_GROUPS,
    Embodiments,
    normalise_actions,
    normalise_body_actions,
)


def in_band(embodiments: Embodiments, dataset: str, key: str, slot: int | None,
            targets: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Physical targets -> (in_band per column, per-group flags, the dims judged)."""
    entry = embodiments.entry(key)
    alpha = embodiments.band_alpha
    if slot is None:
        dims = embodiments.body_columns(dataset, key)
        _, flags = normalise_body_actions(targets, entry, alpha=alpha, dims_valid=dims)
        groups = BODY_GROUPS
    else:
        dims = embodiments.columns(dataset, key, slot)
        _, flags = normalise_actions(targets, entry, slot, alpha=alpha, dims_valid=dims)
        groups = ACTION_GROUPS
    flags = flags.cpu().numpy()
    return flags, {name: flags[..., span.start] for name, span in groups.items()}, dims


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--normalizer", required=True, help="the released embodiments.json")
    parser.add_argument("--dataset", required=True,
                        help="the manifest entry name whose trained columns are judged")
    parser.add_argument("--key", required=True, help="the embodiment key of the targets")
    side = parser.add_mutually_exclusive_group(required=True)
    side.add_argument("--slot", type=int, help="the arm slot of [..., 26] targets")
    side.add_argument("--body", action="store_true", help="[..., 8] body targets")
    parser.add_argument("--targets", required=True, help=".npy physical targets")
    parser.add_argument("--out", required=True, help="new .npz output")
    args = parser.parse_args(argv)

    targets_path = Path(args.targets)
    out = common.new_output(args.out, [targets_path])
    embodiments = Embodiments.load(args.normalizer)
    targets = np.load(targets_path, allow_pickle=False)
    flags, groups, dims = in_band(embodiments, args.dataset, args.key,
                                  None if args.body else args.slot, targets)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as stream:
        np.savez(stream, in_band=flags, dims_valid=dims,
                 **{f"group_{name}": flag for name, flag in groups.items()})
    print(json.dumps({
        "band_alpha": embodiments.band_alpha,
        "steps": int(np.prod(flags.shape[:-1])),
        "masked_steps": {name: int((~flag).sum()) for name, flag in groups.items()},
        "out": str(out),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
