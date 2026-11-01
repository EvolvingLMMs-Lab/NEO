"""config/embodiments.yaml under v7 (neo_vla_embodiment_keys_v2): what each key declares.

Per key the table states its clause, slots, effector, stage and frame fix, and
under v7 three declarations the stages read and every entry digest covers:
joints ({count, names, unit} or null), hand ({count, order, unit,
polarity_note} or null) and body (absent, or {base, torso, head} with none
where the source records no such part). A v1 table states none of them and is
refused by name. build_manifest holds every directory to its keys'
declarations; build_stats copies them into embodiments.json.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from neo_vla.constants import (
    ARM_JOINTS,
    EMBODIMENT_KEYS_SCHEMA,
    HAND_JOINTS,
    OLD_EMBODIMENT_KEYS_SCHEMAS,
)
from neo_vla.data.body import BASE_SOURCES, HEAD_JOINTS, TORSO_JOINTS
from neo_vla.data.embodiment import HAND_UNITS, JOINT_UNITS, inherited_groups

# The three declarations a v2 key states. Each may be null (joints, hand) or
# the word absent (body), but never left out: silence is how a v1 key reads.
DECLARATIONS = ("joints", "hand", "body")

# The fields each declaration carries into embodiments.json (the entry digest
# covers them). A key may add a note for its reader; it is not copied.
JOINT_FIELDS = ("count", "names", "unit")
HAND_FIELDS = ("count", "order", "unit", "polarity_note")
NOTE = "note"

# How the table says a body part records nothing.
NONE = ("none", None)


def refuse(key: str, said: str, expected: str) -> ValueError:
    return ValueError(
        f"config/embodiments.yaml key {key} {said}\n"
        f"expected {expected}\n"
        f"fix the key in config/embodiments.yaml ({EMBODIMENT_KEYS_SCHEMA})"
    )


def load(path: Path) -> dict[str, Any]:
    """The table, refused unless it is the v2 schema and every key states its declarations."""
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    schema = document.get("schema") if isinstance(document, dict) else None
    if schema in OLD_EMBODIMENT_KEYS_SCHEMAS:
        raise ValueError(
            f"{path} states schema {schema!r}, the v6 key table (no joints, hand or body)\n"
            f"expected {EMBODIMENT_KEYS_SCHEMA!r}\n"
            "state joints, hand and body for every key and the v2 schema"
        )
    if schema != EMBODIMENT_KEYS_SCHEMA or not isinstance(document.get("keys"), dict):
        raise ValueError(
            f"{path} states schema {schema!r} and keys of type "
            f"{type(document.get('keys')).__name__ if isinstance(document, dict) else None}\n"
            f"expected {EMBODIMENT_KEYS_SCHEMA!r} and a keys table\n"
            "write the key table"
        )
    for key, spec in document["keys"].items():
        missing = [name for name in DECLARATIONS if name not in (spec or {})]
        if missing:
            raise refuse(str(key), f"states no {missing}",
                         "joints, hand and body on every key (null or absent where none)")
        joints_of(str(key), spec)
        hand_of(str(key), spec)
        body_of(str(key), spec)
        action_groups_of(str(key), spec)
    return document


def _named(key: str, what: str, raw: Any, fields: tuple[str, ...], names: str,
           widest: int, units: tuple[str, ...]) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict) or set(raw) - set(fields) - {NOTE}:
        raise refuse(key, f"declares {what} {raw!r}",
                     f"null or an object of {list(fields)} (and an optional {NOTE})")
    count, listed, unit = raw.get("count"), raw.get(names), raw.get("unit")
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= widest \
            or not isinstance(listed, list) or len(listed) != count \
            or not all(isinstance(one, str) and one for one in listed) or unit not in units:
        raise refuse(key, f"declares {what} count {count!r}, {names} {listed!r}, unit {unit!r}",
                     f"a count 1 to {widest}, that many {names}, a unit in {list(units)}")
    return {field: raw[field] for field in fields if field in raw}


def action_groups_of(key: str, spec: dict[str, Any]) -> tuple[str, ...] | None:
    """The action groups a key copies from action_from (V6 4.7 D6(i)), or None for a key
    that measures its whole action block.

    action_from_groups lists names of neo_vla.data.embodiment.ACTION_GROUPS (joints, ee,
    hand); a key with action_from that states none copies every group, as before the field.
    """
    source, raw = spec.get("action_from"), spec.get("action_from_groups")
    if source is None:
        if raw is not None:
            raise refuse(key, f"states action_from_groups {raw!r} and no action_from",
                         "action_from_groups only beside action_from")
        return None
    try:
        return inherited_groups(raw, f"config/embodiments.yaml key {key}")
    except ValueError as error:
        raise refuse(key, f"states action_from_groups {raw!r}",
                     "a non-empty list of distinct names of joints, ee and hand") from error


def joints_of(key: str, spec: dict[str, Any]) -> dict[str, Any] | None:
    """The key's joints as embodiments.json states them, or None for a key with none."""
    return _named(key, "joints", spec.get("joints"), JOINT_FIELDS, "names", ARM_JOINTS,
                  JOINT_UNITS)


def hand_of(key: str, spec: dict[str, Any]) -> dict[str, Any] | None:
    """The key's hand as embodiments.json states it, or None for a gripper key."""
    return _named(key, "hand", spec.get("hand"), HAND_FIELDS, "order", HAND_JOINTS, HAND_UNITS)


def _part(key: str, raw: Any, part: str, widest: int) -> list[str] | None:
    if raw in NONE:
        return None
    if not isinstance(raw, list) or not 1 <= len(raw) <= widest \
            or not all(isinstance(one, str) and one for one in raw):
        raise refuse(key, f"declares body {part} {raw!r}", f"none or 1 to {widest} names")
    return list(raw)


def body_of(key: str, spec: dict[str, Any]) -> dict[str, Any] | None:
    """The key's body as embodiments.json states it ({base, torso, head}), or None if absent."""
    raw = spec.get("body")
    if raw == "absent":
        return None
    if not isinstance(raw, dict) or set(raw) != {"base", "torso", "head"} \
            or raw["base"] not in BASE_SOURCES:
        raise refuse(key, f"declares body {raw!r}",
                     f"absent, or {{base: one of {list(BASE_SOURCES)}, torso, head}}")
    return {"base": str(raw["base"]), "torso": _part(key, raw["torso"], "torso", TORSO_JOINTS),
            "head": _part(key, raw["head"], "head", HEAD_JOINTS)}


def body_counts(body: dict[str, Any] | None) -> dict[str, Any] | None:
    """A key's body declaration as the manifest's body block states it: base and counts."""
    if body is None:
        return None
    return {"base": body["base"], "torso": len(body["torso"] or ()),
            "head": len(body["head"] or ())}


def declares_stored(spec: dict[str, Any], key: str) -> bool:
    """Whether a key's directories must carry stamped v7 columns: it declares a part or a body."""
    return (joints_of(key, spec) is not None or hand_of(key, spec) is not None
            or body_of(key, spec) is not None)
