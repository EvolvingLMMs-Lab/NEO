"""What a directory is: its canonical name, its source, and its camera views.

The manifest records the answer for every directory; this module holds the
vocabulary those answers are drawn from, so the tool that writes a manifest and
the loader that reads one cannot mean different things by "rh20t" or by
"wrist". Nothing here touches the filesystem: it maps names to names.

A source id is a permanent number. Re-issuing one would make two corpora
indistinguishable in a manifest that mixes both, so the table only ever grows.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Permanent. A new corpus takes the next number; a number is never reused.
# 7-17 are the five datasets added for the gd7 stage-1 corpus, one per source
# word their layouts state.
SOURCE_IDS: dict[str, int] = {"oxe": 0, "agibot": 1, "rh20t": 2, "libero": 3,
                              "droid3cam": 4, "pika": 5, "robotwin2": 6,
                              "interndata_a1_franka_hand": 7,
                              "interndata_a1_franka_robotiq": 8,
                              "interndata_a1_genie1": 9, "interndata_a1_lift2": 10,
                              "interndata_a1_split_aloha": 11, "robocoin": 12,
                              "robocoin_hands": 13, "galaxea_owd": 14, "robomind": 15,
                              "robomind_tienkung": 16, "bimanual_yam": 17}

# What a camera is looking from. "unknown" is a member of the vocabulary and
# not an absence: a view whose role was never established renders as an
# unnamed camera rather than as a confidently wrong one.
VIEW_ROLES = ("exterior", "head", "side", "wrist", "unknown")

# Roles a camera-frame action target may be expressed in: every one of them.
# Every delta is spatial (in the robot base frame) before any camera turns it
# (CX 12), and a camera frame is R_base<-camera transposed applied to it: a
# change of basis that keeps every degree of freedom whatever the camera looks
# from. What decides whether a camera can be the reference is whether its pose
# was measured, not its role, and the manifest states that per view.
ACTION_FRAME_ROLES = frozenset(VIEW_ROLES)

# Suffixes a shard carries in its directory name. Stripped to a fixed point, so
# the order they were appended in cannot change the canonical name.
SHARD_SUFFIXES = ("_unified", "_shardNNofNN")

# The datasets of the OXE conversion, one name each however many shards the
# converter wrote. Listed rather than matched by shape: their names share no
# form, and a rule loose enough to cover them would also claim the next corpus.
OXE_DATASETS = (
    "austin_buds_dataset_converted_externally_to_rlds",
    "austin_sailor_dataset_converted_externally_to_rlds",
    "austin_sirius_dataset_converted_externally_to_rlds",
    "bc_z",
    "berkeley_autolab_ur5",
    "berkeley_cable_routing",
    "berkeley_fanuc_manipulation",
    "bridge_orig",
    "cmu_stretch",
    "dlr_edan_shared_control_converted_externally_to_rlds",
    "dobbe",
    "droid",
    "droid3cam",
    "fmb_dataset",
    "fractal20220817_data",
    "furniture_bench_dataset_converted_externally_to_rlds",
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds",
    "jaco_play",
    "kuka",
    "language_table",
    "nyu_franka_play_dataset_converted_externally_to_rlds",
    "roboturk",
    "stanford_hydra_dataset_converted_externally_to_rlds",
    "taco_play",
    "toto",
    "ucsd_kitchen_dataset_converted_externally_to_rlds",
    "utaustin_mutex",
    "viola",
)

# One whole-name rule per source. A name two rules claim raises and so does a
# name none claims: a directory that took a source by default would take that
# source's arm count and camera vocabulary with it.
SOURCE_RULES: tuple[tuple[str, str], ...] = (
    (r"task_\d+", "agibot"),
    (r"rh20t_k\d+_(?:bysa|bync)", "rh20t"),
    (r"libero_\w+_no_noops(?:_[\d.]+)?_lerobot", "libero"),
    # Both spellings of the RoboTwin conversion: the directory's own last
    # component, and the manifest name its assign rule builds from two.
    (r"(?:robotwin2_)?aloha_agilex_\w+", "robotwin2"),
    ("|".join(re.escape(name) for name in OXE_DATASETS), "oxe"),
)

_SOURCE_PATTERNS = tuple(
    (re.compile(pattern), source) for pattern, source in SOURCE_RULES
)

# A suffix pattern spells its digits as runs of N, so the table above reads as
# a directory name does. A run of N is any run of digits.
_SUFFIX_PATTERN = re.compile(
    "(?:%s)$" % "|".join(
        re.sub("N+", r"\\d+", re.escape(suffix)) for suffix in SHARD_SUFFIXES
    )
)

# A name that is all suffix would strip to the empty string and collide with
# every other degenerate name, so stripping stops one step before that.
_MAX_STRIPS = 8


@dataclass(frozen=True)
class ViewSpec:
    """One camera of one directory, as the manifest records it.

    key         the column the frames are stored under
    role        one of VIEW_ROLES
    slot        which of the model's camera slots this view occupies
    prompt_noun the words the renderer uses for it, or "" when the role is
                unknown and the renderer says nothing about the camera
    arm_slot    the action slot of the arm this camera is mounted on, or None
                for a camera that does not move with an arm. It is what lets a
                dual-arm sample give each arm its own wrist reference, and it
                is what composes a mounted camera's pose for the prefix.
    """

    key: str
    role: str
    slot: int
    prompt_noun: str
    arm_slot: int | None = None

    def __post_init__(self) -> None:
        if self.role not in VIEW_ROLES:
            raise ValueError(
                f"view {self.key!r} has role {self.role!r}.\n"
                f"Expected one of {VIEW_ROLES}.\n"
                "Add the role to VIEW_ROLES, or record this camera as unknown."
            )
        if self.slot < 0:
            raise ValueError(
                f"view {self.key!r} takes camera slot {self.slot}.\n"
                "Expected a slot at or above 0.\n"
                "Fix the slot in the manifest entry for this directory."
            )
        if self.arm_slot is not None and self.arm_slot < 0:
            raise ValueError(
                f"view {self.key!r} is mounted on action slot {self.arm_slot}.\n"
                "Expected a slot at or above 0; every action slot is an arm.\n"
                "Fix the view's arm_slot in the manifest entry for this directory."
            )


@dataclass(frozen=True)
class ArmSpec:
    """One arm of one directory: which action slot it fills and its name.

    slot is the action tensor's slot index and it starts at 0: every slot is an
    arm now, and arm identity is positional, so this number is the arm's whole
    identity to the decoder rather than a label beside one.
    """

    slot: int
    name: str

    def __post_init__(self) -> None:
        if self.slot < 0:
            raise ValueError(
                f"arm {self.name!r} takes action slot {self.slot}.\n"
                "Expected a slot at or above 0; the first arm takes slot 0.\n"
                "Fix the slot in the manifest entry for this directory."
            )


def canonical_name(directory: str) -> str:
    """A directory path or basename -> the dataset name the manifest keys on.

    Strips the shard and unification suffixes repeatedly until the name stops
    changing, so two spellings of the same dataset cannot resolve to two
    entries and split its mixture weight in half.
    """
    name = str(directory).rstrip("/").rsplit("/", 1)[-1]
    for _ in range(_MAX_STRIPS):
        stripped = _SUFFIX_PATTERN.sub("", name)
        if stripped == name or not stripped:
            return name
        name = stripped
    raise ValueError(
        f"canonical_name did not settle for {directory!r} after {_MAX_STRIPS} strips, "
        f"reaching {name!r}.\n"
        "Expected the suffixes to run out.\n"
        "Check that no pattern in SHARD_SUFFIXES matches the empty string."
    )


def source_of(name: str) -> str:
    """A canonical dataset name -> its source, a key of SOURCE_IDS.

    Raises for a name no source claims. The alternative is a default source,
    which would give an unrecognised corpus another corpus's state layout.
    """
    claimed = [
        source for pattern, source in _SOURCE_PATTERNS if pattern.fullmatch(name)
    ]
    if len(claimed) == 1:
        return claimed[0]
    if not claimed:
        raise ValueError(
            f"no source claims the dataset name {name!r}.\n"
            f"Expected a name one rule in SOURCE_RULES matches, for one of {tuple(SOURCE_IDS)}.\n"
            "Add the dataset to OXE_DATASETS or add a rule to SOURCE_RULES."
        )
    raise ValueError(
        f"the dataset name {name!r} is claimed by {sorted(claimed)}.\n"
        "Expected exactly one source to claim it.\n"
        "Narrow one of the rules in SOURCE_RULES."
    )


def source_id(source: str) -> int:
    """A source name -> its permanent number. Raises for an unregistered one."""
    if source not in SOURCE_IDS:
        raise ValueError(
            f"source {source!r} has no id.\n"
            f"Expected one of {tuple(SOURCE_IDS)}.\n"
            "Give the new corpus the next free number in SOURCE_IDS; never reuse one."
        )
    return SOURCE_IDS[source]


def reference_views(views: tuple[ViewSpec, ...]) -> tuple[ViewSpec, ...]:
    """The views a camera-frame sample may express its actions in, in slot order.

    Empty means the directory has no camera whose role admits it, which is what
    puts its samples in robot mode. Several is not a tie to settle here: the
    reference is drawn per sample, and the order is the slot order so that the
    draw is a function of the seed rather than of how the manifest was written.
    """
    eligible = [view for view in views if view.role in ACTION_FRAME_ROLES]
    return tuple(sorted(eligible, key=lambda view: view.slot))
