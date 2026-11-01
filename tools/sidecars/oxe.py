"""Open X-Embodiment: joints for the five families whose state holds them.

Group A stores arm joint angles in rad in
observation.state[0:7] of our own parquet, under the wrong names x..yaw:
austin_buds, austin_sirius, utaustin_mutex and viola (7 joints) and
berkeley_fanuc_manipulation (6; its slot 6 is a pad, so joint 6 is NaN). A
row whose joint slots are all exactly 0 is a missing reading (mostly an
episode's first row) and is written NaN. Every other OXE directory records no
joints anywhere on the cluster: its sidecar is written all NaN with the
reason stated, so a reader can tell "not recorded" from "not built".
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from sidecars import common

ARMS = 1

# Entry name -> joints its observation.state[0:n] holds.
JOINT_STATE = {
    "austin_buds_dataset_converted_externally_to_rlds": 7,
    "austin_sirius_dataset_converted_externally_to_rlds": 7,
    "utaustin_mutex": 7,
    "viola": 7,
    "berkeley_fanuc_manipulation": 6,
}

UNRECORDED = {
    common.COLUMN_HAND: {"arms": ARMS, "unit": "rad", "recorded": False,
                         "source": "not recorded"},
    common.COLUMN_BODY: {"unit": "", "recorded": False,
                         "source": "not recorded in our parquet or on the cluster"},
}


def described(name: str) -> dict[str, dict[str, Any]]:
    """The column descriptions of one OXE entry."""
    if name in JOINT_STATE:
        joints = {"arms": ARMS, "unit": "rad", "recorded": True,
                  "source": (f"our parquet observation.state[0:{JOINT_STATE[name]}]; "
                             "all-zero rows NaN")}
    else:
        joints = {"arms": ARMS, "unit": "rad", "recorded": False,
                  "source": ("not recorded: no joint field in our parquet, and the raw "
                             "RLDS is not on the cluster (decision D12)")}
    return {common.COLUMN_JOINTS: joints, **UNRECORDED}


def joint_rows(state: np.ndarray, count: int) -> tuple[np.ndarray, int]:
    """The joint slots of a state block, NaN on rows holding no reading."""
    joints = np.full((state.shape[0], 7), np.nan, dtype=np.float64)
    values = state[:, :count].astype(np.float64)
    missing = (values == 0.0).all(axis=1)
    values[missing] = np.nan
    joints[:, :count] = values
    return joints, int(missing.sum())


def build(entry: dict[str, Any], directory: Path) -> dict[str, Any]:
    """Every row of one OXE directory: columns, stamp parts and checks."""
    name = entry["name"]
    count = JOINT_STATE.get(name)
    episodes = common.directory_episodes(directory)
    total = sum(episode.rows for episode in episodes)
    columns = common.empty_columns(total, ARMS, command=False)
    episode_column = np.empty(total, dtype=np.int64)
    frame_column = np.empty(total, dtype=np.int64)
    digest = common.SourceDigest()
    checks = {"episodes": len(episodes), "zero_rows_written_nan": 0}
    read = ["observation.state", common.FRAME_COLUMN] if count else [common.FRAME_COLUMN]
    start = 0
    for episode in episodes:
        table = pq.read_table(episode.data_file, columns=read)
        stop = start + episode.rows
        episode_column[start:stop] = episode.index
        frame_column[start:stop] = table.column(common.FRAME_COLUMN).to_numpy()
        if count:
            state = np.stack(table.column("observation.state").to_numpy(zero_copy_only=False))
            digest.text(str(episode.index))
            digest.array(state[:, :count])
            joints, missing = joint_rows(state, count)
            columns[common.COLUMN_JOINTS][start:stop] = common.finite(joints)
            checks["zero_rows_written_nan"] += missing
        start = stop
    if not count:
        digest.text("no joint source")
    return {"episodes": episodes, "columns": columns, "episode_column": episode_column,
            "frame_column": frame_column, "described": described(name), "arms": ARMS,
            "source_digest": digest.hexdigest(), "checks": checks}
