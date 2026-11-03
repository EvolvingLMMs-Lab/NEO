"""The numbers the v7 layout shares between the loader, the model, the server and the tools.

Each value is stated here once and re-exported by the module that owns what it
means: neo_vla.data.history, neo_vla.data.actions, neo_vla.data.embodiment,
neo_vla.model.tokenizers, neo_vla.model.encoder, neo_vla.data.prompt.template
and neo_vla.engine.checkpoint. A change to any value here changes the data and
checkpoint contract of the v7 layout. A v6 value is not kept beside its v7
value: the OLD_ and RETIRED_ tables name it, so a reader refuses a v6 artefact
by its name.

Nothing in this module imports torch or numpy: the eval clients import it
inside simulator environments that have neither.
"""
from __future__ import annotations

# The prefix layout and the formats that carry it. A checkpoint, a prompt cache,
# a manifest and an embodiments file each state one of these, and a reader
# refuses any other, naming the old value when it is one of the OLD_ tables.
# PREFIX_LAYOUT_ALIASES are earlier labels of this same layout, which files and
# checkpoints written before the label was settled still state.
PREFIX_LAYOUT = "native"
PREFIX_LAYOUT_ALIASES = ("v7",)
OLD_PREFIX_LAYOUTS = ("v6",)
CHECKPOINT_FORMAT = "neo_vla_step_v4"
OLD_CHECKPOINT_FORMATS = ("neo_vla_step_v1", "neo_vla_step_v2", "neo_vla_step_v3")
FORMAT_GUARD = "neo_vla_step_v4.a_tree_without_embodiments_refuses_this"
EMBODIMENTS_SCHEMA = "neo_vla_embodiments_v2"
OLD_EMBODIMENTS_SCHEMAS = ("neo_vla_embodiments_v1",)
EMBODIMENTS_FILE = "embodiments.json"
EMBODIMENT_KEYS_SCHEMA = "neo_vla_embodiment_keys_v2"
OLD_EMBODIMENT_KEYS_SCHEMAS = ("neo_vla_embodiment_keys_v1",)
MANIFEST_SCHEMA_VERSION = 3
OLD_MANIFEST_SCHEMA_VERSIONS = (1, 2)

# The action-pretraining corpus (stages 1 and 2 read it), and where every
# corpus keeps its files under the data root: the manifest at
# manifests/<corpus>.json, the stage records, normaliser and sidecars under
# METADATA_DIR/<corpus>, the numeric, frame, camera and prompt caches under
# PROCESSED_DIR/<corpus>.
PRETRAIN_CORPUS = "action_pretrain"
METADATA_DIR = "metadata"
PROCESSED_DIR = "processed"

# The stored v7 columns, one spelling for the new-data parquet and the stage-1
# sidecars alike; NaN means not recorded, never 0. A command column exists only
# where a family's target is its command (DROID joints, Galaxea base). An ingest
# output states INGEST_LAYOUT under INGEST_LAYOUT_FIELD in meta/info.json, and a
# sidecar lives at METADATA_DIR/<corpus>/SIDECAR_DIR/<entry name>.parquet.
INGEST_LAYOUT = "v7"
INGEST_LAYOUT_FIELD = "ingest_layout"
COLUMN_JOINTS = "observation.joints"
COLUMN_HAND = "observation.hand"
COLUMN_BODY = "observation.body"
COLUMN_JOINTS_COMMAND = "action.joints"
COLUMN_BODY_COMMAND = "action.body"
SIDECAR_DIR = "sidecars"

# The widest arm and hand any key has. A 6-DoF arm fills joints 0 to 5 and its
# joint 6 is invalid, never 0; a hand fills its declared order from h0.
ARM_JOINTS = 7
HAND_JOINTS = 12

# One arm's action row at chunk step j: joint targets in rad, dpos in cm per
# step and drot as a spatial rotation vector in rad per step (both in the
# frame_ref frame), the canonical gripper (0 open), and hand targets. Joints
# and hand are absolute in joint space and never turned by the camera frame.
# ACTION_EE is the 7 columns every benchmark executes; RANGE_COLUMNS are the
# q01/q99-mapped ones, every column but the gripper.
ACTION_DIM = 26
ACTION_RECORD = "j7_dpos3_drot3_g1_h12"
ACTION_JOINTS = slice(0, 7)
ACTION_DPOS = slice(7, 10)
ACTION_DROT = slice(10, 13)
ACTION_GRIPPER = slice(13, 14)
GRIPPER_COLUMN = 13
ACTION_HAND = slice(14, 26)
ACTION_EE = slice(7, 14)
RANGE_COLUMNS = tuple(range(0, 13)) + tuple(range(14, 26))
ACTION_STAMP = {"dim": 26, "record": "j7_dpos3_drot3_g1_h12"}
OLD_ACTION_DIMS = (19,)

# One arm's state record: joints in rad, position in metres in the robot base
# frame, columns 0 and 1 of R_base<-ee, the canonical aperture a in [0, 1] with
# 0 open, and the hand. The model reads [x * m, m], the record with its invalid
# dimensions zeroed and the mask of its valid ones, STATE_INPUT_DIM wide.
STATE_DIM = 29
STATE_RECORD = "j7_pos3_rot6d01_ap_h12"
STATE_JOINTS = slice(0, 7)
STATE_POSITION = slice(7, 10)
STATE_ROTATION = slice(10, 16)
STATE_APERTURE = 16
STATE_HAND = slice(17, 29)
STATE_INPUT_DIM = 58
STATE_STAMP = {"dim": 29, "record": "j7_pos3_rot6d01_ap_h12"}
OLD_STATE_RECORDS = ("pos3_rot6d01_ap",)
OLD_STATE_DIMS = (10,)

# The body block, one per sample and shared by the arms, in state and action
# alike, always in the base frame (never turned by frame_ref): base dx, dy in
# cm per step and dyaw in rad per step (state: the last step's motion; action:
# the step's own), then three torso joints and head yaw, pitch, absolute.
BODY_DIM = 8
BODY_RECORD = "base3_torso3_head2"
BODY_BASE = slice(0, 3)
BODY_TORSO = slice(3, 6)
BODY_HEAD = slice(6, 8)
BODY_POSTURE = slice(3, 8)
BODY_INPUT_DIM = 16
BODY_STAMP = {"dim": 8, "record": "base3_torso3_head2"}

# The two time roles of the state tokens, in the order the prefix lays them out.
ROLE_START = 0
ROLE_NOW = 1
N_ROLES = 2

# What each state token holds, one code per (role, arm) and per role for the
# body; which dimensions are valid is the per-dimension mask beside it. The
# producer writes the first code that applies in STATE_CODE_PRECEDENCE. Measured
# means at least one valid dimension; a start token without history is the
# zero record with an all-ones mask, even for a source whose state is unknown.
STATE_MEASURED = 0
STATE_UNKNOWN = 1
STATE_NO_HISTORY = 2
STATE_ABSENT = 3
STATE_CODE_PRECEDENCE = (STATE_ABSENT, STATE_NO_HISTORY, STATE_UNKNOWN, STATE_MEASURED)

# The placeholder id of a state column (<FAKE_PAD_12>) and the id the scale
# token held, which stays reserved and is never reused.
STATE_TOKEN_ID = 151694
RETIRED_ANCHOR_IDS = {"<scale>": 151690}

# The arm slots, and the prefix's state columns, role-major: (start, arm 0),
# (start, arm 1), (start, body), (now, arm 0), (now, arm 1), (now, body).
# The suffix holds one block per arm slot and the body's block last.
N_ARM_SLOTS = 2
N_STATE_COLUMNS = 6
SUFFIX_BLOCKS = 3
BODY_BLOCK = 2

# Training-only state noise per (role, arm), each value already multiplied by
# its sigma, in its own order, which is not the record's: position 3, rotation
# vector 3 in radians, aperture 1, joints 7, hand 12. The body's noise is per
# role, in the body's order. NOISE_ slices index the noise, STATE_ the record.
NOISE_WIDTH = 26
NOISE_POSITION = slice(0, 3)
NOISE_ROTATION = slice(3, 6)
NOISE_APERTURE = 6
NOISE_JOINTS = slice(7, 14)
NOISE_HAND = slice(14, 26)
BODY_NOISE_WIDTH = 8

# History at H = 1: one earlier chunk start p, at least MIN_GAP rows before t,
# drawn uniformly on 0 .. t - MIN_GAP. The no-history form masks the earlier
# views and feeds the start tokens the zero record with an all-ones mask.
HISTORY_SIZE = 1
MIN_GAP = 8
SAMPLING = "uniform"
NO_HISTORY_FORM = "zero_state"

# One draw's extra uniforms (neo_vla.data.mixture.extra_uniforms), by position.
# 0 to 2 belong to the reference-camera draw and are unchanged; 6 to 109 are
# the arm normals, role-major, then slot, then the 26 noise columns; 110 to 125
# are the body normals, role-major, then the 8 body columns.
EXTRA_UNIFORMS = 126
EXTRA_HISTORY_STEP = 3
EXTRA_DROPOUT = 4
EXTRA_NOISE_SWITCH = 5
EXTRA_NOISE_FIRST = 6
EXTRA_BODY_NOISE_FIRST = 110
UNIFORM_FLOOR = 2.0 ** -53
UNIFORM_CEIL = 1.0 - 2.0 ** -53

# The scaffold's own text and the fixed widths around the rendered text. The
# rendered text holds sections 2 to 5 and opens with FREQUENCY_OPEN; the state
# line (a newline and the six state columns) and the 9-id close follow it, and
# padding follows them.
HISTORY_HEADER = "Earlier:\n"
NOW_HEADER = "Now:\n"
STATE_LINE_OPEN = "\n"
FREQUENCY_OPEN = "2. Frequency information:\nFrequency: "
INSTRUCTION_OPEN = "Instruction: "
PREFIX_BEFORE_TEXT = 636
PREFIX_AFTER_TEXT = 16
PREFIX_LEN = 908

# The batch both producers emit. A geometry record is in the batch only when
# the intrinsics mode reads it; every view key and geometry key has a history_
# twin laid out the same way. dim_valid and body_dim_valid are per sample: the
# directory's trained columns and the dimensions read on every step the sample
# keeps, constant over the chunk as at serve. TRAIN_KEYS are what serving drops.
VIEW_KEYS = ("pixels", "image_shapes", "view_valid")
GEOMETRY_KEYS = ("intrinsics", "intrinsics_valid", "camera_extrinsics", "camera_extrinsics_valid")
HISTORY_PREFIX = "history_"
SAMPLE_KEYS = (
    "prompt", "embodiment", "frame_ref", "slot_valid", "history_valid",
    "state", "state_code", "state_dims", "dim_valid",
    "body_state", "body_state_code", "body_state_dims", "body_dim_valid",
)
TRAIN_KEYS = (
    "actions", "action_valid", "action_loss_dims", "state_noise",
    "body_actions", "body_action_valid", "body_loss_dims", "body_state_noise",
)
RETIRED_BATCH_KEYS = {
    "action_loss_valid": "action_loss_dims",
    "state_valid": "state_code and state_dims",
    "scale": "the frequency line of the prompt text",
}

# Refusals by number. Every refusal message starts with refusal_prefix(n), so a
# test and a log search find it by its number.
REFUSALS = {
    1: "missing_embodiments_file",
    2: "embodiments_mismatch",
    3: "unknown_embodiment_key",
    4: "record_mismatch",
    5: "two_stage_conflict",
    6: "retired_field",
    7: "untrained_prompt_head",
    8: "inherited_entry_changed",
    9: "shared_clause_or_head",
    10: "constant_or_absent_dim_requested",
    11: "old_checkpoint_format",
    12: "untrained_frame_ref",
    13: "normalised_reply",
    14: "fit_gate",
    15: "slot_not_in_entry",
    16: "episode_identity",
    17: "libero_instruction_not_eval",
    18: "history_key_missing",
    19: "history_gap_below_min",
    20: "history_or_state_shape",
    21: "state_noise_at_eval",
    22: "stamp_mismatch",
    23: "embodiment_missing",
    24: "unknown_request_field",
}


def prefix_layout_of(stated) -> str:
    """A prefix-layout label as read from a file or checkpoint, earlier labels of
    this layout read as PREFIX_LAYOUT and any other value returned as stated."""
    stated = str(stated)
    return PREFIX_LAYOUT if stated in PREFIX_LAYOUT_ALIASES else stated


def refusal_prefix(number: int) -> str:
    """R<n> and its name, the first words of every message refusing under it."""
    if number not in REFUSALS:
        raise KeyError(
            f"refusal R{number} is not in REFUSALS.\n"
            f"Expected one of R1 to R{max(REFUSALS)}.\n"
            "Add it to this table."
        )
    return f"R{number} {REFUSALS[number]}:"


def batch_keys(geometry: tuple[str, ...], *, train: bool) -> frozenset[str]:
    """The exact key set of one robot batch, for the geometry records a mode reads.

    geometry is the subset of GEOMETRY_KEYS the intrinsics mode reads; tokens
    reads all four. train adds TRAIN_KEYS; a co-train batch also carries "chat",
    which this set leaves out because only the robot half is compared.
    """
    unknown = sorted(set(geometry) - set(GEOMETRY_KEYS))
    if unknown:
        raise ValueError(
            f"geometry names {unknown}, which are not geometry records.\n"
            f"Expected a subset of {list(GEOMETRY_KEYS)}.\n"
            "Pass the records the intrinsics mode reads."
        )
    prefix = (*VIEW_KEYS, *geometry)
    keys = {*prefix, *(HISTORY_PREFIX + key for key in prefix), *SAMPLE_KEYS}
    if train:
        keys |= set(TRAIN_KEYS)
    return frozenset(keys)
