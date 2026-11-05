"""Requests in, action chunks out: the serving side of the robot batch.

A request is one observation in native records: the 29-wide arm state per
served slot, the 8-wide body, one RGB frame per camera slot, the per-view
calibration and camera pose, and the previous step as history. The Predictor
checks it against the step's embodiments file, builds the batch the loader
would have built for the same frame, runs policy.predict and reads the chunk
back into physical units through the same normaliser.

Which columns a request may ask for comes from the datasets record of the
directory it names. A request that names no directory, or a step whose file
carries no datasets records, is served from the embodiment entry alone: the
columns the key can decode and the state it holds measured.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy
import torch
from ..constants import (
    ACTION_DIM,
    BODY_DIM,
    BODY_RECORD,
    GEOMETRY_KEYS,
    HISTORY_PREFIX,
    N_ROLES,
    ROLE_NOW,
    ROLE_START,
    STATE_ABSENT,
    STATE_APERTURE,
    STATE_DIM,
    STATE_HAND,
    STATE_JOINTS,
    STATE_MEASURED,
    STATE_NO_HISTORY,
    STATE_POSITION,
    STATE_RECORD,
    STATE_ROTATION,
    STATE_UNKNOWN,
    batch_keys,
    refusal_prefix,
)
from ..data.conditioning import mark_unknown
from ..data.embodiment import denormalise, denormalise_body, normalise_body_state, normalise_state
from ..data.images import check_fits, place_views, resize_filter, snap_grid
from ..data.rays import CALIBRATION_COLUMNS
from ..data.reader import PIXEL_MAX
from ..model.tokenizers import EXTRINSICS_DIM, FRAME_REF_CAMERA, FRAME_REF_ROBOT, MAX_CAMERAS

logger = logging.getLogger(__name__)

FRAME_REFS = {"camera": FRAME_REF_CAMERA, "robot": FRAME_REF_ROBOT}
ROTATION_TOLERANCE = 1e-4
STATE_FIELDS = (
    ("joints", STATE_JOINTS),
    ("position", STATE_POSITION),
    ("rotation", STATE_ROTATION),
    ("aperture", slice(STATE_APERTURE, STATE_APERTURE + 1)),
    ("hand", STATE_HAND),
)
# A pose is read whole or not at all.
ATOMIC_FIELDS = ("position", "rotation")
VIEW_RECORDS = (("intrinsics", len(CALIBRATION_COLUMNS)), ("camera_extrinsics", EXTRINSICS_DIM))
TRAINING_ONLY = ("state_noise", "body_state_noise")
REQUEST_FIELDS = (
    "embodiment", "directory", "prompt", "images", "arm_slots", "state", "body",
    "intrinsics", "camera_extrinsics", "frame_ref", "dim_valid", "body_dim_valid",
    "history", "noise_seed",
)
HISTORY_FIELDS = ("images", "state", "body", "camera_extrinsics", "gap")


def refusal(number: int, what: str, expected: str, fix: str) -> ValueError:
    return ValueError(f"{refusal_prefix(number)} {what}\nExpected {expected}\n{fix}")


def frame_ref_value(spell) -> int:
    """'robot', 'camera' or 'camera:N' -> the frame row the suffix tokens read."""
    text = str(spell).strip()
    if text in FRAME_REFS:
        return FRAME_REFS[text]
    if text.startswith("camera:"):
        tail = text.split(":", 1)[1]
        if tail.isdigit():
            slot = int(tail)
            if 0 <= slot < MAX_CAMERAS:
                return FRAME_REF_CAMERA + slot
            raise ValueError(
                f"frame_ref names camera slot {slot}.\n"
                f"Expected a slot below {MAX_CAMERAS}, the camera rows the frame table has."
            )
    raise ValueError(
        f"frame_ref is {spell!r}.\n"
        "Expected 'robot', 'camera' (the same as 'camera:0'), or 'camera:N'."
    )


def canonical_frame(spell) -> str:
    row = frame_ref_value(spell)
    return "robot" if row == FRAME_REF_ROBOT else f"camera:{row - FRAME_REF_CAMERA}"


def frame_refs_of(value, arms: int) -> tuple[str, ...]:
    """One frame for every served arm, or one per arm."""
    if isinstance(value, str):
        frame_ref_value(value)
        return (value,) * int(arms)
    if isinstance(value, (list, tuple)) and len(value) == int(arms) and all(
        isinstance(one, str) for one in value
    ):
        for one in value:
            frame_ref_value(one)
        return tuple(value)
    raise ValueError(f"frame_ref is {value!r:.80} for {int(arms)} served arms.")


def check_arm_slots(slots, n_slots: int) -> tuple[int, ...]:
    read = tuple(int(slot) for slot in slots)
    if not read:
        raise ValueError("no arm slot was named.")
    if len(set(read)) != len(read):
        raise ValueError(f"the arm slots are {list(read)} and one of them repeats.")
    outside = [slot for slot in read if not 0 <= slot < int(n_slots)]
    if outside:
        raise ValueError(f"the arm slots {outside} are not in 0..{int(n_slots) - 1}.")
    return read


def required(payload: dict, name: str):
    if name not in payload:
        raise ValueError(f"the request carries no {name!r}.")
    return payload[name]


def as_record(value, name: str, width: int) -> numpy.ndarray | None:
    if value is None:
        return None
    array = numpy.asarray(value, dtype=numpy.float32).reshape(-1)
    if array.shape != (width,):
        raise ValueError(f"{name} carries {array.size} numbers, expected {width}.")
    if not bool(numpy.isfinite(array).all()):
        raise ValueError(f"{name} holds a non-finite number: {array.tolist()}.")
    return array


def pose_off(pose: numpy.ndarray) -> float:
    """How far two 3-vectors are from an orthonormal pair."""
    first, second = pose[0:3].astype(numpy.float64), pose[3:6].astype(numpy.float64)
    return max(
        abs(float(numpy.linalg.norm(first)) - 1.0),
        abs(float(numpy.linalg.norm(second)) - 1.0),
        abs(float(first @ second)),
    )


def check_pose(pose: numpy.ndarray | None, where: str, number: int | None = None):
    if pose is None:
        return None
    off = pose_off(pose)
    if off > ROTATION_TOLERANCE:
        what = f"{where} holds pose {pose.tolist()}, {off:.3e} off an orthonormal pair."
        if number is not None:
            raise refusal(number, what,
                          f"forward and up of a rotation, within {ROTATION_TOLERANCE}.",
                          "Build the pose with neo_vla.data.conditioning.extrinsics_6.")
        raise ValueError(what)
    return pose


def as_view_records(value, name: str, width: int, views: int) -> tuple:
    sent = list(value)
    if len(sent) != int(views):
        raise ValueError(f"{name} carries {len(sent)} entries for {int(views)} camera slots.")
    records = tuple(as_record(one, name, int(width)) for one in sent)
    if name == "camera_extrinsics":
        for slot, one in enumerate(records):
            check_pose(one, f"camera_extrinsics[{slot}]")
    return records


def record_entries(value, width: int, where: str, record: str) -> numpy.ndarray:
    """A list of numbers and nulls -> float64 with NaN where null."""
    try:
        sent = list(value)
    except TypeError as error:
        raise refusal(20, f"{where} is {type(value).__name__}, not a list.",
                      f"{width} entries ({record}), each a number or null, or null for the "
                      "whole record.",
                      "Send the record as a list.") from error
    if len(sent) != int(width):
        raise refusal(20, f"{where} carries {len(sent)} entries.",
                      f"{width} entries in {record} order, null for each entry not read.",
                      f"Pack the record as {record}.")
    values = numpy.full(int(width), numpy.nan, dtype=numpy.float64)
    for index, one in enumerate(sent):
        if one is None:
            continue
        if isinstance(one, bool) or not isinstance(
            one, (int, float, numpy.integer, numpy.floating)
        ):
            raise refusal(20, f"{where}[{index}] is {one!r}.",
                          "a number, or null for an entry not read.",
                          "Send numbers only.")
        number = float(one)
        if not math.isfinite(number):
            raise refusal(20, f"{where}[{index}] holds the non-finite number {number}.",
                          "a finite measurement, or null for an entry not read.",
                          "Send null where the robot did not report the entry.")
        values[index] = number
    if not bool(numpy.isfinite(values).any()):
        raise refusal(20, f"{where} holds {width} nulls.",
                      "null for a record with nothing read, not a list of nulls.",
                      "Send null for the whole record.")
    return values


def state_record(value, where: str) -> numpy.ndarray | None:
    if value is None:
        return None
    array = record_entries(value, STATE_DIM, where, STATE_RECORD)
    for name in ATOMIC_FIELDS:
        span = dict(STATE_FIELDS)[name]
        given = numpy.isfinite(array[span])
        if bool(given.any()) and not bool(given.all()):
            raise refusal(20, f"{where} holds part of its {name}: {array[span].tolist()}.",
                          f"the {name} all numbers or all null.",
                          "Send the whole pose, or null for every entry of it.")
    six = array[STATE_ROTATION]
    if bool(numpy.isfinite(six).all()):
        off = pose_off(six)
        if off > ROTATION_TOLERANCE:
            raise refusal(20, f"{where} holds 6D columns {six.tolist()}, "
                          f"{off:.3e} off an orthonormal pair.",
                          f"columns 0 and 1 of a rotation, within {ROTATION_TOLERANCE}.",
                          "Send the first two columns of the end-effector rotation matrix.")
    aperture = float(array[STATE_APERTURE])
    if math.isfinite(aperture) and not 0.0 <= aperture <= 1.0:
        raise refusal(20, f"{where} holds aperture {aperture}.",
                      "the canonical aperture in [0, 1], 0 open and 1 closed.",
                      "Map the raw gripper reading through its open and closed values.")
    return array.astype(numpy.float32)


def state_records(value, arms: int, where: str) -> tuple:
    try:
        sent = list(value)
    except TypeError as error:
        raise refusal(20, f"{where} is {type(value).__name__}, not a list.",
                      "one record per arm slot.",
                      "Send a list, null where an arm is not measured.") from error
    if len(sent) != int(arms):
        raise refusal(20, f"{where} carries {len(sent)} records for {int(arms)} arm slots.",
                      "one per slot of arm_slots, null where the robot does not measure it.",
                      "Send one entry per served arm, in the same order.")
    return tuple(state_record(one, f"{where}[{index}]") for index, one in enumerate(sent))


def body_record(value, where: str, *, present: bool) -> numpy.ndarray | None:
    if not present:
        if value is not None:
            raise refusal(20, f"{where} is a list and the key has no body (a fixed base).",
                          "null: a key without a body has no body record.",
                          "Send body null for this embodiment, in the history too.")
        return None
    if value is None:
        return None
    return record_entries(value, BODY_DIM, where, BODY_RECORD).astype(numpy.float32)


def check_image(sent_image, where: str, number: int | None = None) -> numpy.ndarray:
    image = numpy.asarray(sent_image)
    if image.dtype != numpy.uint8 or image.ndim != 3 or image.shape[2] != 3:
        what = f"{where} is {image.dtype} {image.shape}, expected uint8 [height, width, 3]."
        if number is not None:
            raise refusal(number, what, "the camera's own RGB frame.",
                          "Send the frame as captured, not a normalised tensor.")
        raise ValueError(what)
    return image


def check_fields(payload: dict) -> None:
    for name in TRAINING_ONLY:
        if name in payload:
            raise refusal(21, f"the request carries {name!r}, which only training draws.",
                          "no state noise at inference; noise_seed seeds the decode noise.",
                          f"Drop {name!r} from the request.")
    unknown = sorted(set(payload) - set(REQUEST_FIELDS))
    if unknown:
        raise refusal(24, f"the request carries {unknown}, which are not request fields.",
                      f"only {list(REQUEST_FIELDS)}.",
                      "Drop them; a field the server does not read would be ignored silently.")


def check_embodiment(payload: dict, embodiments) -> str:
    if payload.get("embodiment") in (None, ""):
        raise refusal(23, "the request names no embodiment.",
                      "the embodiment key of the robot, one of the checkpoint's entries.",
                      "Send the key with every request.")
    key = str(payload["embodiment"])
    if key not in embodiments.entries:
        raise refusal(3, f"the request names embodiment {key!r}, which this checkpoint has "
                      "no entry for.",
                      f"one of {sorted(embodiments.entries)}.",
                      "Serve a checkpoint trained on this embodiment.")
    return key


def check_directory(payload: dict, embodiments, key: str):
    """(directory, record), or (None, None) for a request served from the entry alone."""
    named = payload.get("directory")
    if named in (None, ""):
        return None, None
    record = embodiments.datasets.get(str(named))
    if record is None:
        raise refusal(4, f"the request names directory {named!r}, which this step holds no "
                      "datasets record for.",
                      f"one of the {len(embodiments.datasets)} directories in its "
                      "embodiments.json.",
                      "Send directory null to serve from the embodiment entry alone.")
    if key not in tuple(record.keys):
        raise refusal(4, f"directory {named!r} trained keys {list(record.keys)} and the "
                      f"request names {key!r}.",
                      "a key the directory's record lists.",
                      "Send the embodiment key the directory was trained under.")
    return str(named), record


@dataclass(frozen=True)
class History:
    images: tuple
    state: tuple
    body: object
    camera_extrinsics: tuple
    gap: int


def parse_history(payload: dict, *, views: int, arms: int, min_gap: int,
                  body_present: bool) -> History | None:
    if "history" not in payload:
        raise refusal(18, "the request carries no 'history'.",
                      "null on the first call of an episode, the previous step after that.",
                      "Send 'history' with every request, null included.")
    sent = payload["history"]
    if sent is None:
        return None
    if not isinstance(sent, dict) or set(sent) != set(HISTORY_FIELDS):
        found = sorted(sent) if isinstance(sent, dict) else type(sent).__name__
        raise refusal(20, f"the history is {found}.",
                      f"null or an object with exactly {list(HISTORY_FIELDS)}.",
                      "Send the previous step's observation and the rows executed since.")
    gap = sent["gap"]
    if isinstance(gap, bool) or not isinstance(gap, (int, numpy.integer)):
        raise refusal(20, f"the history gap is {gap!r}.", "an integer count of rows.",
                      "Send the number of action rows executed since the previous step.")
    if int(gap) < int(min_gap):
        raise refusal(19, f"the history gap is {int(gap)} rows.",
                      f"at least the checkpoint's min_gap, {int(min_gap)}; training never drew "
                      "an earlier step closer than that.",
                      f"Execute at least {int(min_gap)} rows of each chunk before the next call.")
    images = list(sent["images"]) if isinstance(sent["images"], (list, tuple)) else None
    if images is None or len(images) != int(views):
        count = "not a list" if images is None else f"{len(images)} entries"
        raise refusal(20, f"the history images are {count} for {int(views)} camera slots.",
                      "one entry per camera slot of the request, null where empty.",
                      "Send the previous step's images as they were sent.")
    if all(one is None for one in images):
        raise refusal(20, "every history image is null.",
                      "null for the whole history when there is no earlier step.",
                      "Send 'history': null on the first call instead.")
    checked = tuple(
        None if one is None else check_image(one, f"history camera slot {slot}", 20)
        for slot, one in enumerate(images)
    )
    poses = sent["camera_extrinsics"]
    if not isinstance(poses, (list, tuple)) or len(poses) != int(views):
        raise refusal(20, f"the history camera_extrinsics are {poses!r:.60}.",
                      f"one entry per camera slot ({int(views)}), null where not measured.",
                      "Send the previous step's camera_extrinsics.")
    extrinsics = []
    for slot, one in enumerate(poses):
        if one is None:
            extrinsics.append(None)
            continue
        record = numpy.asarray(one, dtype=numpy.float32).reshape(-1)
        if record.shape != (EXTRINSICS_DIM,) or not bool(numpy.isfinite(record).all()):
            raise refusal(20, f"history camera slot {slot} carries pose {record.tolist()}.",
                          f"{EXTRINSICS_DIM} finite numbers, forward then up, or null.",
                          "Send the previous step's camera_extrinsics.")
        extrinsics.append(check_pose(record, f"history camera slot {slot}", 20))
    return History(
        images=checked,
        state=state_records(sent["state"], arms, "the history state"),
        body=body_record(sent["body"], "the history body", present=body_present),
        camera_extrinsics=tuple(extrinsics),
        gap=int(gap),
    )


class Masks:
    """The columns one request may use: its directory's record, or the entry alone."""

    def __init__(self, embodiments, key: str, directory: str | None):
        self.embodiments = embodiments
        self.key = key
        self.directory = directory
        self.entry = embodiments.entries[key]

    def where(self) -> str:
        return f"{self.directory!r}" if self.directory is not None else "the entry alone"

    def columns(self, slot: int) -> numpy.ndarray:
        if self.directory is None:
            found = self.entry.decodable(int(slot))
        else:
            found = self.embodiments.columns(self.directory, self.key, int(slot))
        return numpy.asarray(found, dtype=bool).copy()

    def state_dims(self, slot: int) -> numpy.ndarray:
        if self.directory is None:
            return numpy.asarray(self.entry.state_valid(int(slot)), dtype=bool)
        return self.embodiments.state_dims(self.directory, self.key, int(slot))

    def body_columns(self) -> numpy.ndarray:
        if self.directory is None:
            found = self.entry.body.decodable()
        else:
            found = self.embodiments.body_columns(self.directory, self.key)
        return numpy.asarray(found, dtype=bool).copy()

    def body_state_dims(self) -> numpy.ndarray:
        if self.directory is None:
            return ~numpy.asarray(self.entry.body.state_absent, dtype=bool)
        return self.embodiments.body_state_dims(self.directory, self.key)


def check_slots(entry, record, arm_slots, *, key: str, directory: str | None) -> None:
    for slot in arm_slots:
        entry.slot_index(int(slot))
        if record is not None and int(slot) not in tuple(int(one) for one in record.slots):
            raise refusal(15, f"directory {directory!r} packs its arms into slots "
                          f"{list(record.slots)} and the request drives slot {int(slot)} "
                          f"under {key!r}.",
                          "the slots the directory's arms fill.",
                          "Serve the slots this directory was trained in.")


def check_state_fields(entry, key: str, arm_slots, records: list[tuple]) -> None:
    for where, state in records:
        for arm, slot in enumerate(arm_slots):
            record = state[arm]
            if record is None:
                continue
            absent = numpy.asarray(entry.state_absent[entry.slot_index(int(slot))], dtype=bool)
            for name, span in STATE_FIELDS:
                if bool(absent[span].all()) and bool(numpy.isfinite(record[span]).any()):
                    raise refusal(20, f"{where}[{arm}] sends the {name} of slot {slot} under "
                                  f"{key!r}, whose entry has no range for it.",
                                  f"null for the {name}: training never held it measured "
                                  "for this key.",
                                  f"Send null for the {name}, in the history too.")


def masked_record(record, dims) -> tuple:
    """Drop the entries training never held valid; count how many were sent."""
    if record is None:
        return None, 0
    sent = numpy.isfinite(record)
    keep = sent & numpy.asarray(dims, dtype=bool)
    masked = int(numpy.count_nonzero(sent & ~keep))
    if not bool(keep.any()):
        return None, masked
    return numpy.where(keep, record, numpy.nan).astype(numpy.float32), masked


def check_dim_valid(value, *, masks: Masks, arm_slots, action_dim: int):
    if value is None:
        return numpy.stack([masks.columns(int(slot)) for slot in arm_slots])
    flags = numpy.asarray(value)
    if flags.shape != (len(arm_slots), int(action_dim)):
        raise refusal(20, f"dim_valid is {tuple(flags.shape)}.",
                      f"({len(arm_slots)}, {int(action_dim)}): one row per arm slot and one "
                      "flag per action column, or null.",
                      "Send null to take every column the model was trained to decode.")
    flags = flags.astype(bool)
    for arm, slot in enumerate(arm_slots):
        trained = masks.columns(int(slot))
        asked = sorted(numpy.flatnonzero(flags[arm] & ~trained).tolist())
        if asked:
            raise refusal(10, f"dim_valid asks for columns {asked} of slot {slot} under "
                          f"{masks.key!r} for {masks.where()}.",
                          f"only the trained columns {numpy.flatnonzero(trained).tolist()}.",
                          "Send dim_valid null, or leave those columns false.")
    return flags


def check_body_dim_valid(value, *, masks: Masks):
    if value is None:
        return masks.body_columns()
    flags = numpy.asarray(value)
    if flags.shape != (BODY_DIM,):
        raise refusal(20, f"body_dim_valid is {tuple(flags.shape)}.",
                      f"({BODY_DIM},): one flag per body column, or null.",
                      "Send null to take every body column the model was trained to decode.")
    flags = flags.astype(bool)
    trained = masks.body_columns()
    asked = sorted(numpy.flatnonzero(flags & ~trained).tolist())
    if asked:
        raise refusal(10, f"body_dim_valid asks for body columns {asked} under "
                      f"{masks.key!r} for {masks.where()}.",
                      f"only the trained body columns {numpy.flatnonzero(trained).tolist()} "
                      "(none for a key without a body).",
                      "Send body_dim_valid null, or leave those columns false.")
    return flags


def check_frame_trained(frame_ref: str, *, key: str, trained) -> None:
    """A step that records its training frames refuses any other frame."""
    if trained is None:
        return
    spelled = canonical_frame(frame_ref)
    frames = dict(trained.get("keys", {}).get(key, {}).get("frames", {}))
    if float(frames.get(spelled, 0.0)) <= 0.0:
        raise refusal(12, f"frame_ref {frame_ref!r} ({spelled}) was not trained under {key!r}.",
                      "a frame the run trained, one of "
                      f"{sorted(name for name, share in frames.items() if float(share) > 0.0)}.",
                      "Send a trained frame, or serve a run that trained this one.")


def check_prompt_head(prompt: str, *, key: str, trained) -> None:
    """A step that records its prompt heads refuses a prompt that opens with no such head."""
    if trained is None:
        return
    heads = list(trained.get("keys", {}).get(key, {}).get("prompt_heads", []))
    for head, tail in heads:
        head, tail = str(head), str(tail)
        if not (prompt.startswith(head) and prompt.endswith(tail)):
            continue
        task = prompt[len(head):len(prompt) - len(tail)]
        if len(prompt) >= len(head) + len(tail) and task.strip():
            return
    shown = [str(head)[-80:] for head, _ in heads[:3]]
    raise refusal(7, f"the prompt {prompt[:120]!r} opens with no head this run trained "
                  f"under {key!r}.",
                  f"one of the {len(heads)} trained heads (ends of the first three: {shown}), "
                  "then the task.",
                  "Render the prompt with neo_vla.serve.prompt.render_prompt.")


@dataclass(frozen=True)
class Observation:
    embodiment: str
    directory: str | None
    prompt: str
    images: tuple
    arm_slots: tuple[int, ...]
    state: tuple
    body: numpy.ndarray | None
    body_present: bool
    intrinsics: tuple
    camera_extrinsics: tuple
    frame_ref: tuple[str, ...]
    dim_valid: numpy.ndarray
    body_dim_valid: numpy.ndarray
    history: History | None
    noise_seed: int | None = None
    masked: dict = field(default_factory=lambda: {"state_dims_masked": 0, "body_dims_masked": 0})

    @classmethod
    def from_payload(cls, payload: dict, *, n_views: int, action_dim: int,
                     arm_slots: tuple[int, ...], embodiments, min_gap: int) -> Observation:
        check_fields(payload)
        key = check_embodiment(payload, embodiments)
        directory, record = check_directory(payload, embodiments, key)
        asked = tuple(int(slot) for slot in required(payload, "arm_slots"))
        if asked != tuple(arm_slots):
            raise ValueError(f"the request asks for arm slots {list(asked)} and this server "
                             f"reads {list(arm_slots)}.")
        entry = embodiments.entries[key]
        check_slots(entry, record, asked, key=key, directory=directory)
        masks = Masks(embodiments, key, directory)
        arms = len(arm_slots)
        sent = list(required(payload, "images"))
        if len(sent) > int(n_views):
            raise ValueError(f"the request carries {len(sent)} camera slots, the model has "
                             f"{int(n_views)}.")
        images = tuple(
            None if one is None else check_image(one, f"camera slot {slot}")
            for slot, one in enumerate(sent)
        )
        present = bool(entry.body.present)
        history = parse_history(payload, views=len(images), arms=arms, min_gap=min_gap,
                                body_present=present)
        state = state_records(required(payload, "state"), arms, "the state")
        body = body_record(required(payload, "body"), "the body", present=present)
        check_state_fields(entry, key, asked, [("the state", state)] + (
            [] if history is None else [("the history state", history.state)]))
        dims = [masks.state_dims(int(slot)) for slot in asked]
        body_dims = masks.body_state_dims()
        counts = {"state_dims_masked": 0, "body_dims_masked": 0}

        def mask_arms(records: tuple) -> tuple:
            fed = []
            for arm, one in enumerate(records):
                kept, masked = masked_record(one, dims[arm])
                counts["state_dims_masked"] += masked
                fed.append(kept)
            return tuple(fed)

        def mask_body(one):
            kept, masked = masked_record(one, body_dims)
            counts["body_dims_masked"] += masked
            return kept

        state, body = mask_arms(state), mask_body(body)
        if history is not None:
            history = History(images=history.images, state=mask_arms(history.state),
                              body=mask_body(history.body),
                              camera_extrinsics=history.camera_extrinsics, gap=history.gap)
        frame_ref = frame_refs_of(required(payload, "frame_ref"), arms)
        flags = check_dim_valid(required(payload, "dim_valid"), masks=masks, arm_slots=asked,
                                action_dim=action_dim)
        body_flags = check_body_dim_valid(required(payload, "body_dim_valid"), masks=masks)
        for spelling in dict.fromkeys(frame_ref):
            check_frame_trained(spelling, key=key, trained=embodiments.trained)
        prompt = str(required(payload, "prompt"))
        check_prompt_head(prompt, key=key, trained=embodiments.trained)
        views = {name: as_view_records(required(payload, name), name, width, len(images))
                 for name, width in VIEW_RECORDS}
        return cls(
            embodiment=key,
            directory=directory,
            prompt=prompt,
            images=images,
            arm_slots=asked,
            state=state,
            body=body,
            body_present=present,
            intrinsics=views["intrinsics"],
            camera_extrinsics=views["camera_extrinsics"],
            frame_ref=frame_ref,
            dim_valid=flags,
            body_dim_valid=body_flags,
            history=history,
            noise_seed=None if payload.get("noise_seed") is None else int(payload["noise_seed"]),
            masked=counts,
        )


def check_body_presence(codes, body_dim_valid) -> None:
    absent = codes == STATE_ABSENT
    if not torch.equal(absent[:, ROLE_START], absent[:, ROLE_NOW]):
        raise RuntimeError("body_state_code is ABSENT in one role and not the other.")
    if bool(body_dim_valid[absent[:, ROLE_NOW]].any()):
        raise RuntimeError("body_dim_valid is True on a request whose key has no body.")


class Predictor:
    """Turn checked requests into batches and decoded chunks for one served policy."""

    def __init__(self, policy, *, arm_slots, min_gap: int, device: str | torch.device = "cpu"):
        if policy.embodiments is None:
            raise ValueError("the served policy carries no embodiments.")
        self.policy = policy
        self.embodiments = policy.embodiments
        self.n_slots = int(policy.decoder.n_slots)
        self.action_dim = int(policy.decoder.action_dim)
        self.body_dim = int(policy.decoder.body_dim)
        if (self.action_dim, self.body_dim) != (ACTION_DIM, BODY_DIM):
            raise ValueError(f"the served policy decodes {self.action_dim}-wide arm rows and a "
                             f"{self.body_dim}-wide body, expected {ACTION_DIM} and {BODY_DIM}.")
        self.n_views = int(policy.encoder.n_views)
        self.arm_slots = check_arm_slots(arm_slots, self.n_slots)
        # The canvas is the one training tiled: max_grid tokens of pixels_per_token each.
        self.token = int(policy.encoder.vision.pixels_per_token)
        self.short_side = min(int(size) for size in policy.encoder.max_grid) * self.token
        self.min_gap = int(min_gap)
        self.resample = resize_filter(policy.encoder.pixel_convention)
        self.cell = (int(policy.encoder.max_grid[0]) * self.token,
                     int(policy.encoder.max_grid[1]) * self.token)
        self.device = torch.device(device)
        logger.info("serving arm slots %s of %d; canvas %dx%d, short side %d, %d pixels per "
                    "token; history gap at least %d", list(self.arm_slots), self.n_slots,
                    self.cell[0], self.cell[1], self.short_side, self.token, self.min_gap)

    def observation(self, payload: dict) -> Observation:
        observation = Observation.from_payload(
            payload, n_views=self.n_views, action_dim=self.action_dim,
            arm_slots=self.arm_slots, embodiments=self.embodiments, min_gap=self.min_gap,
        )
        # Tokenise now, so a prompt over the text budget is refused before the batch.
        self.policy.encoder.instruction_ids(observation.prompt)
        return observation

    def views(self, images: tuple):
        frames, tiles, slots = {}, {}, {}
        for slot, image in enumerate(images):
            if image is None:
                continue
            key = f"view{slot}"
            frame = torch.from_numpy(image).permute(2, 0, 1).float() / PIXEL_MAX
            tile = snap_grid((int(frame.shape[-2]), int(frame.shape[-1])),
                             short_side=self.short_side, token=self.token, cell=self.cell)
            check_fits(tile, cell=self.cell)
            frames[key], tiles[key], slots[key] = frame, tile, slot
        return place_views(frames, tiles, slots=slots, n_views=self.n_views, cell=self.cell,
                           short_side=self.short_side, token=self.token,
                           resample=self.resample)

    def states(self, observations: list[Observation]):
        size = len(observations)
        state = torch.full((size, N_ROLES, self.n_slots, STATE_DIM), float("nan"))
        dims = torch.zeros((size, N_ROLES, self.n_slots, STATE_DIM), dtype=torch.bool)
        codes = torch.full((size, N_ROLES, self.n_slots), STATE_ABSENT, dtype=torch.long)
        for row, observation in enumerate(observations):
            for arm, slot in enumerate(self.arm_slots):
                roles = {ROLE_NOW: observation.state[arm]}
                if observation.history is None:
                    codes[row, ROLE_START, slot] = STATE_NO_HISTORY
                else:
                    roles[ROLE_START] = observation.history.state[arm]
                for role, record in roles.items():
                    if record is None:
                        codes[row, role, slot] = STATE_UNKNOWN
                        continue
                    codes[row, role, slot] = STATE_MEASURED
                    values = torch.from_numpy(numpy.asarray(record, dtype=numpy.float32))
                    state[row, role, slot] = values
                    dims[row, role, slot] = torch.isfinite(values)
        return state, codes, dims

    def bodies(self, observations: list[Observation]):
        size = len(observations)
        body = torch.full((size, N_ROLES, BODY_DIM), float("nan"))
        dims = torch.zeros((size, N_ROLES, BODY_DIM), dtype=torch.bool)
        codes = torch.full((size, N_ROLES), STATE_ABSENT, dtype=torch.long)
        for row, observation in enumerate(observations):
            if not observation.body_present:
                continue
            roles = {ROLE_NOW: observation.body}
            if observation.history is None:
                codes[row, ROLE_START] = STATE_NO_HISTORY
            else:
                roles[ROLE_START] = observation.history.body
            for role, record in roles.items():
                if record is None:
                    codes[row, role] = STATE_UNKNOWN
                    continue
                codes[row, role] = STATE_MEASURED
                values = torch.from_numpy(numpy.asarray(record, dtype=numpy.float32))
                body[row, role] = values
                dims[row, role] = torch.isfinite(values)
        return body, codes, dims

    def view_record(self, rows: list[tuple], width: int):
        data = torch.zeros(len(rows), self.n_views, int(width), dtype=torch.float32)
        valid = torch.zeros(len(rows), self.n_views, dtype=torch.bool)
        for row, records in enumerate(rows):
            for slot, values in enumerate(records):
                if values is None:
                    continue
                data[row, slot] = torch.from_numpy(values)
                valid[row, slot] = True
        return mark_unknown(data, valid), valid

    def noise(self, observations: list[Observation]):
        """Seeded decode noise when any request asks for it, else None (the policy draws)."""
        if all(one.noise_seed is None for one in observations):
            return None
        chunk = int(self.policy.decoder.chunk)
        arms, bodies = [], []
        for one in observations:
            generator = None
            if one.noise_seed is not None:
                generator = torch.Generator()
                generator.manual_seed(int(one.noise_seed))
            arms.append(torch.randn((1, self.n_slots, chunk, self.action_dim),
                                    generator=generator))
            bodies.append(torch.randn((1, chunk, self.body_dim), generator=generator))
        return torch.cat(arms, dim=0), torch.cat(bodies, dim=0)

    def batch(self, observations: list[Observation]) -> dict:
        size = len(observations)
        placed = [self.views(one.images) for one in observations]
        earlier = [self.views(() if one.history is None else one.history.images)
                   for one in observations]
        served = list(self.arm_slots)
        slot_valid = torch.zeros(size, self.n_slots, dtype=torch.bool)
        slot_valid[:, served] = True
        frame_ref = torch.full((size, self.n_slots), FRAME_REF_ROBOT, dtype=torch.long)
        dim_valid = torch.zeros(size, self.n_slots, self.action_dim, dtype=torch.bool)
        body_dim_valid = torch.zeros(size, self.body_dim, dtype=torch.bool)
        for row, observation in enumerate(observations):
            for arm, slot in enumerate(served):
                frame_ref[row, slot] = frame_ref_value(observation.frame_ref[arm])
                dim_valid[row, slot] = torch.from_numpy(
                    numpy.array(observation.dim_valid[arm], dtype=bool))
            body_dim_valid[row] = torch.from_numpy(
                numpy.array(observation.body_dim_valid, dtype=bool))
        state, codes, dims = self.states(observations)
        body, body_codes, body_dims = self.bodies(observations)
        check_body_presence(body_codes, body_dim_valid)
        history_view_valid = torch.stack([valid for _, _, valid in earlier])
        batch = {
            "pixels": torch.stack([pixels for pixels, _, _ in placed]),
            "image_shapes": torch.stack([shapes for _, shapes, _ in placed]),
            "view_valid": torch.stack([valid for _, _, valid in placed]),
            HISTORY_PREFIX + "pixels": torch.stack([pixels for pixels, _, _ in earlier]),
            HISTORY_PREFIX + "image_shapes": torch.stack([shapes for _, shapes, _ in earlier]),
            HISTORY_PREFIX + "view_valid": history_view_valid,
            "history_valid": torch.tensor([one.history is not None for one in observations],
                                          dtype=torch.bool),
            "prompt": [one.prompt for one in observations],
            "embodiment": [one.embodiment for one in observations],
            "frame_ref": frame_ref,
            "slot_valid": slot_valid,
            "state": state,
            "state_code": codes,
            "state_dims": dims,
            "dim_valid": dim_valid,
            "body_state": body,
            "body_state_code": body_codes,
            "body_state_dims": body_dims,
            "body_dim_valid": body_dim_valid,
        }
        batch.update(self.geometry_batch(observations, batch["view_valid"], history_view_valid))
        expected = batch_keys(tuple(GEOMETRY_KEYS), train=False)
        if set(batch) != expected:
            raise RuntimeError(f"the serving batch holds {sorted(set(batch) ^ expected)} out of "
                               "step with batch_keys.")
        return {key: value.to(self.device) if torch.is_tensor(value) else value
                for key, value in batch.items()}

    def geometry_batch(self, observations: list[Observation], view_valid,
                       history_view_valid) -> dict:
        found = {}
        data, measured = self.view_record([one.intrinsics for one in observations],
                                          len(CALIBRATION_COLUMNS))
        valid = measured & view_valid
        data = mark_unknown(data, valid)
        found["intrinsics"], found["intrinsics_valid"] = data, valid
        # The calibration does not change between the two steps.
        found[HISTORY_PREFIX + "intrinsics"] = data.clone()
        found[HISTORY_PREFIX + "intrinsics_valid"] = valid & history_view_valid
        data, measured = self.view_record([one.camera_extrinsics for one in observations],
                                          EXTRINSICS_DIM)
        valid = measured & view_valid
        found["camera_extrinsics"] = mark_unknown(data, valid)
        found["camera_extrinsics_valid"] = valid
        earlier, measured = self.view_record(
            [(None,) * self.n_views if one.history is None else one.history.camera_extrinsics
             for one in observations],
            EXTRINSICS_DIM,
        )
        present = measured & history_view_valid
        found[HISTORY_PREFIX + "camera_extrinsics"] = mark_unknown(earlier, present)
        found[HISTORY_PREFIX + "camera_extrinsics_valid"] = present
        return found

    @torch.inference_mode()
    def __call__(self, observations: list[Observation]) -> list[tuple]:
        """Normalised chunks, one (arms [n_arms, chunk, 26], body [chunk, 8]) per request."""
        noise = self.noise(observations)
        arms, body = self.policy.predict(
            self.batch(observations),
            noise=None if noise is None else noise[0].to(self.device),
            body_noise=None if noise is None else noise[1].to(self.device),
        )
        served = arms[:, list(self.arm_slots)].float().cpu().numpy()
        bodies = body.float().cpu().numpy()
        return [(numpy.ascontiguousarray(row), numpy.ascontiguousarray(one))
                for row, one in zip(served, bodies)]

    def decode(self, observation: Observation, chunks) -> dict:
        """One normalised chunk -> physical actions, NaN where a column is not emitted."""
        arm_chunks, body_chunk = chunks
        entry = self.embodiments.entries[observation.embodiment]
        alpha = float(self.embodiments.band_alpha)
        values = numpy.asarray(arm_chunks, dtype=numpy.float32)
        actions = numpy.full(values.shape, numpy.nan, dtype=numpy.float32)
        emitted = numpy.zeros((values.shape[0], values.shape[-1]), dtype=bool)
        clips = {
            "state_range": 0,
            "decode_range": 0,
            "gripper": 0,
            "state_dims_masked": int(observation.masked["state_dims_masked"]),
            "body_dims_masked": int(observation.masked["body_dims_masked"]),
        }
        for arm, slot in enumerate(observation.arm_slots):
            physical, decoded, counted = denormalise(
                torch.from_numpy(values[arm].copy()), entry, int(slot), alpha=alpha)
            row = entry.slot_index(int(slot))
            # A constant column is emitted at its constant whether or not it was asked for.
            constant = (numpy.asarray(entry.constant[row], dtype=bool)
                        & ~numpy.asarray(entry.absent[row], dtype=bool))
            emitted[arm] = numpy.asarray(decoded, dtype=bool) & (
                numpy.asarray(observation.dim_valid[arm], dtype=bool) | constant)
            actions[arm] = numpy.asarray(physical, dtype=numpy.float32)
            actions[arm][:, ~emitted[arm]] = numpy.nan
            clips["decode_range"] += int(counted["range"])
            clips["gripper"] += int(counted["gripper"])
            records = [observation.state[arm]]
            if observation.history is not None:
                records.append(observation.history.state[arm])
            for record in records:
                if record is None:
                    continue
                _, clipped = normalise_state(
                    torch.from_numpy(numpy.asarray(record, dtype=numpy.float32)), entry,
                    int(slot), alpha=alpha)
                clips["state_range"] += int(clipped)
        body_actions = body_emitted = None
        if observation.body_present:
            physical, decoded, counted = denormalise_body(
                torch.from_numpy(numpy.asarray(body_chunk, dtype=numpy.float32).copy()),
                entry, alpha=alpha)
            constant = entry.body.action_constant & ~entry.body.action_absent
            body_emitted = numpy.asarray(decoded, dtype=bool) & (
                numpy.asarray(observation.body_dim_valid, dtype=bool) | constant)
            body_actions = numpy.asarray(physical, dtype=numpy.float32).copy()
            body_actions[:, ~body_emitted] = numpy.nan
            clips["decode_range"] += int(counted["range"])
            records = [observation.body]
            if observation.history is not None:
                records.append(observation.history.body)
            for record in records:
                if record is None:
                    continue
                _, clipped = normalise_body_state(
                    torch.from_numpy(numpy.asarray(record, dtype=numpy.float32)), entry,
                    alpha=alpha)
                clips["state_range"] += int(clipped)
        return {
            "actions": actions,
            "dim_valid": emitted,
            "body_actions": body_actions,
            "body_dim_valid": body_emitted,
            "arm_slots": list(observation.arm_slots),
            "embodiment": observation.embodiment,
            "clips": clips,
        }
