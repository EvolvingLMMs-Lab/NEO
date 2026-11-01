"""LIBERO and LIBERO-plus: joints from the camrand half's own state, the nom half masked.

The post-train layout config/layouts/libero_v7.json (not shipped here) states
the joints of libero_franka_sim per directory: the four *_no_noops_lerobot
directories (the camrand half) and the tpr6_* re-rendered legs record them in
their own parquet as observation.states.joint_state, float32 [7] rad
(meta/info.json names them joint_0..joint_6: robosuite robot0_joint_pos, which
the LIBERO client sends); the four *_nom_lerobot directories hold
observation.state only, so their joints are not recorded and are masked:
the sidecar holds NaN and says so. Which directory is which is stated here by
name and checked against the parquet both ways, so a directory stated recorded
whose parquet lacks the column, or one stated masked whose parquet has it, is
refused rather than written as the other. A reading that cannot be an angle
(common.implausible_joints) is NaN and counted. There is no hand (a gripper
key) and no body (ABSENT): both columns are NaN, not recorded.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from sidecars import common

ARMS = 1
JOINT_COLUMN = "observation.states.joint_state"
JOINTS = 7

# The directory names whose parquet records the joints, and those that do not,
# by name: <suite>_no_noops_lerobot and tpr6_<half>_<suite> record;
# <suite>_nom_lerobot does not. Any other LIBERO name is refused.
RECORDED_SUFFIX = "_no_noops_lerobot"
RECORDED_PREFIX = "tpr6_"
MASKED_SUFFIX = "_nom_lerobot"

RECORDED = (f"our parquet {JOINT_COLUMN}, float32 [{JOINTS}] rad (robosuite robot0_joint_pos); "
            "readings beyond the joint ceiling or spiking NaN")
MASKED = ("not recorded: StarVLA's nominal-camera LIBERO recordings "
          "(starvla_libero/<suite>_no_noops_1.0.0_lerobot) hold observation.state only; "
          "masked (D18)")
UNRECORDED = {
    common.COLUMN_HAND: {"arms": ARMS, "unit": "rad", "recorded": False,
                         "source": "not recorded: libero_franka_sim has a gripper, no hand"},
    common.COLUMN_BODY: {"unit": "", "recorded": False,
                         "source": "not recorded: libero_franka_sim's body is ABSENT (fixed base)"},
}


def records_joints(name: str) -> bool:
    """Whether a LIBERO directory, by its name, records the joints; refuses a name not stated."""
    if name.endswith(RECORDED_SUFFIX) or name.startswith(RECORDED_PREFIX):
        return True
    if name.endswith(MASKED_SUFFIX):
        return False
    raise ValueError(
        f"LIBERO directory {name} is neither *{RECORDED_SUFFIX}, {RECORDED_PREFIX}* nor "
        f"*{MASKED_SUFFIX}\n"
        "expected a directory config/layouts/libero_v7.json classifies\n"
        "state whether its parquet records the joints in tools/sidecars/libero.py"
    )


def described(recorded: bool) -> dict[str, dict[str, Any]]:
    """The column descriptions of one LIBERO entry."""
    joints = {"arms": ARMS, "unit": "rad", "recorded": recorded,
              "source": RECORDED if recorded else MASKED}
    return {common.COLUMN_JOINTS: joints, **UNRECORDED}


def check_column(name: str, data_file: Path, recorded: bool) -> None:
    """Refuse a parquet whose joint column contradicts the directory's statement."""
    present = JOINT_COLUMN in pq.ParquetFile(data_file).schema_arrow.names
    if present != recorded:
        said = "records the joints" if recorded else "is masked (D18)"
        found = "lacks" if recorded else "holds"
        raise common.Misaligned(
            f"{name} {said} and {data_file} {found} {JOINT_COLUMN}\n"
            f"expected the column {'in every' if recorded else 'in no'} episode parquet\n"
            "fix the directory, or its statement in tools/sidecars/libero.py and "
            "config/layouts/libero_v7.json"
        )


def build(entry: dict[str, Any], directory: Path) -> dict[str, Any]:
    """Every row of one LIBERO directory: columns, stamp parts and checks."""
    name = entry["name"]
    recorded = records_joints(name)
    episodes = common.directory_episodes(directory)
    total = sum(episode.rows for episode in episodes)
    columns = common.empty_columns(total, ARMS, command=False)
    episode_column = np.empty(total, dtype=np.int64)
    frame_column = np.empty(total, dtype=np.int64)
    digest = common.SourceDigest()
    checks = {"episodes": len(episodes), "joint_rows": 0, "implausible_written_nan": 0}
    read = [JOINT_COLUMN, common.FRAME_COLUMN] if recorded else [common.FRAME_COLUMN]
    start = 0
    for episode in episodes:
        check_column(name, episode.data_file, recorded)
        table = pq.read_table(episode.data_file, columns=read)
        stop = start + episode.rows
        episode_column[start:stop] = episode.index
        frame_column[start:stop] = table.column(common.FRAME_COLUMN).to_numpy()
        if recorded:
            values = np.stack(table.column(JOINT_COLUMN).to_numpy(zero_copy_only=False))
            if values.ndim != 2 or values.shape[1] != JOINTS:
                raise common.Misaligned(
                    f"{episode.data_file} {JOINT_COLUMN} has shape {list(values.shape)}\n"
                    f"expected [rows, {JOINTS}], libero_franka_sim's joints\n"
                    "check the directory's joint column")
            digest.text(str(episode.index))
            digest.array(values)
            joints = common.finite(values)
            implausible = common.implausible_joints(joints)
            joints[implausible] = np.nan
            columns[common.COLUMN_JOINTS][start:stop] = joints
            checks["implausible_written_nan"] += int(implausible.sum())
            checks["joint_rows"] += int(np.isfinite(joints).any(axis=1).sum())
        start = stop
    if not recorded:
        digest.text("no joint source (D18)")
    return {"episodes": episodes, "columns": columns, "episode_column": episode_column,
            "frame_column": frame_column, "described": described(recorded), "arms": ARMS,
            "source_digest": digest.hexdigest(), "checks": checks}
