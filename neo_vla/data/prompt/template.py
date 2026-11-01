"""The prompt template artifact: load it, check it, say what it owns.

The template names the sections, the sentence forms and the slots. Section 1
holds anchor ids and projected slots and belongs to the encoder's scaffold,
which is fixed width; sections 2 to 5 are text the renderer produces per
sample. Loading a template that moves a section across that line is refused.

Only major 7 loads. Versions 4 and 5 laid section 2 down as the <scale>
column, a projected step size that no longer exists, and version 6 framed the
v6 layout (four <state> columns, the 10-number record, the 19-wide action
row), so their files describe a prefix this tree cannot build; they are
refused by name, and a checkpoint of that lineage is served by the tree that
trained it. Version 7 renders the words of 6.0.0 (the control frequency as a
line of text), and frequency_text below is the one formatter of that line.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import yaml

from ...constants import FREQUENCY_OPEN, INSTRUCTION_OPEN, TEMPLATE_MAJOR

# The layout this repository renders, and the only major that loads.
TEMPLATE_MAJORS = (TEMPLATE_MAJOR,)
TEMPLATE_FILE = f"neo_v{TEMPLATE_MAJOR}.yaml"

# Majors that loaded before and are refused by name now, with what replaced
# the section that made them unreadable here.
RETIRED_MAJORS = {
    4: "section 2 was the <scale> column and the extrinsics were a per-arm suffix token",
    5: "section 2 was the <scale> column (id 151690, retired and never reused)",
    6: "it framed the v6 layout: four <state> columns (Lp 906), the 10-number state "
       "record and the 19-wide action row; v7 adds the body column per role (Lp 908), "
       "the 29-wide record and the 26-wide row",
}
# The last tree that served each retired major's checkpoints.
SERVED_BY = {4: "dcd90a5", 5: "dcd90a5", 6: "0ca67f7"}

# Sections the encoder's scaffold lays down, by number. The renderer never
# emits these, because anything it emits varies in length per sample and the
# conditioning columns are addressed by fixed index.
SCAFFOLD_SECTIONS = (1,)

# Sections the renderer produces, in order. Section 4 renders its header and an
# empty body until in-context demonstrations exist, so the numbering does not
# shift when they arrive.
RENDERER_SECTIONS = (2, 3, 4, 5)

# How section 1 lays a view's calibration down: tokens, the one layout, puts the
# calibration and the rotation slots beside each camera anchor. The four removed
# layouts are refused by name.
GEOMETRY_LAYOUTS = ("tokens",)
REMOVED_GEOMETRY_LAYOUTS = ("none", "patch", "text", "tower")

# The whole artifact. A key outside this set is refused rather than ignored: a
# field nothing reads is a statement the file makes and the code does not keep,
# which is how the previous tree carried two spellings of one layout.
DOCUMENT_KEYS = ("template", "scaffold_sections", "sentences", "order", "section_joiner",
                 "slots")

# What the header block holds. prefix_layout is checked against the version
# rather than stored: the run's own layout goes into the stamp.
TEMPLATE_KEYS = ("name", "version", "prefix_layout")

# The scaffold block: section 1 only. The section header strings, the history
# and now headers and the state line live in the encoder, so this file cannot
# disagree with those and its digest does not move when they do.
SCAFFOLD_KEYS = {"axis": ("number", "per_view_slots", "geometry")}
# Of those, the ones every artifact states: all of them.
SCAFFOLD_REQUIRED = SCAFFOLD_KEYS

# The per-view slots the one layout lays down, in column order.
PER_VIEW_SLOTS = ("cam", "intrinsics", "extrinsics")

# The slots the renderer has a branch for. A name outside this is refused at
# load, so the order in the file and the dispatch in render.py cannot drift.
KNOWN_SLOTS = ("frequency_section", "frequency", "embodiment_section", "embodiment",
               "embodiment_frame_clause", "demonstration_section", "tasks_section",
               "instruction")

# Slots whose text is a sentence form in the file, and the section number each
# header opens. Every other slot's text comes from the sample's context.
SECTION_HEADERS = {"frequency_section": 2, "embodiment_section": 3,
                   "demonstration_section": 4, "tasks_section": 5}
SENTENCE_SLOTS = ("frequency", "instruction", *SECTION_HEADERS)

# The field each sentence form interpolates. A form missing it renders every
# sample identically, which is a prompt corpus with one row in it.
SENTENCE_FIELDS = {"frequency": "{frequency}", "instruction": "{task}"}

# The order opens with the frequency line: FREQUENCY_OPEN is how the encoder
# recognises the prompt, so the line cannot move.
ORDER_OPENING = ("frequency_section", "frequency")

# Names of a second control-rate channel, refused wherever they appear. The
# rate is the frequency slot and nothing else.
RATE_NAMES = ("control_rate", "rate", "hz")

# The frequency slot's block, whole. Every key is read: frequency_text writes
# the unit after the number, the unknown word for None, and rounds to the
# stated decimals, so the file states exactly what the formatter does.
FREQUENCY_SLOT = {"text": "per_dataset", "unit": "Hz", "unknown": "NA", "decimals": 2}


@dataclass(frozen=True)
class Template:
    """A loaded template artifact.

    version         the full version string, whose major is in TEMPLATE_MAJORS
    prefix_layout   the input layout it was written for, "v7"
    geometry        how section 1 lays a view's calibration down
    sentences       the sentence forms, by slot name
    order           the renderer's slots, in the order they are joined
    section_joiner  what joins them
    digest          the file's hash, which the stamp carries into the cache
    """

    name: str
    version: str
    prefix_layout: str
    geometry: str
    sentences: dict[str, str]
    order: tuple[str, ...]
    section_joiner: str
    digest: str


def frequency_text(hz) -> str:
    """A directory's control rate -> the text of its frequency line.

    None renders the unknown word. An int or a float, not a bool, that is
    finite, above zero and does not round to 0.00 renders with two decimals and
    the trailing zeros dropped: 30 -> "30 Hz", 250 / 15 -> "16.67 Hz". Anything
    else raises: a rate the text cannot state is not rendered as some other one.
    """
    if hz is None:
        return FREQUENCY_SLOT["unknown"]
    if isinstance(hz, bool) or not isinstance(hz, (int, float)):
        raise TypeError(
            f"the control rate is {hz!r} ({type(hz).__name__}).\n"
            "Expected an int or a float in Hz, or None when the rate is not available.\n"
            "Read control_hz from the manifest entry; do not pass it as text or a flag."
        )
    decimals = FREQUENCY_SLOT["decimals"]
    if not math.isfinite(hz) or hz <= 0 or round(hz, decimals) == 0:
        raise ValueError(
            f"the control rate is {hz!r}.\n"
            f"Expected a finite rate above zero that is not 0.00 at {decimals} decimals.\n"
            "Declare the directory's control_hz null (the text reads NA) if it is unknown."
        )
    number = f"{round(hz, decimals):.{decimals}f}".rstrip("0").rstrip(".")
    return f"{number} {FREQUENCY_SLOT['unit']}"


def _refuse_unread_keys(where: str, mapping: dict, allowed: tuple[str, ...]) -> None:
    """Raise for a key the loader does not read, naming it and the set."""
    unread = sorted(set(mapping) - set(allowed))
    if unread:
        raise ValueError(
            f"{where} declares {unread}, which this loader does not read.\n"
            f"Expected only {list(allowed)}.\n"
            "Remove them: a field nothing reads is a layout the code does not keep."
        )


def _require_keys(where: str, mapping: dict, required: tuple[str, ...]) -> None:
    """Raise for the keys the loader reads that the block does not state, all of them."""
    missing = [key for key in required if key not in mapping]
    if missing:
        raise ValueError(
            f"{where} has no {missing}.\n"
            f"Expected every one of {list(required)}.\n"
            "Add them: the loader reads every block."
        )


def _refuse_rate_names(where: str, names) -> None:
    """Raise when a control-rate channel is declared under another name."""
    found = sorted(name for name in names if str(name).lower() in RATE_NAMES)
    if found:
        raise ValueError(
            f"{where} declares {found}.\n"
            "Expected no control-rate channel besides the frequency slot.\n"
            "The rate is one line of section 2; a second statement of it is refused."
        )


def _mapping(where: str, value) -> dict:
    """Raise unless value is a mapping, naming what was found instead."""
    if not isinstance(value, dict):
        raise TypeError(
            f"{where} is {type(value).__name__}, expected a mapping.\n"
            "Expected the template's block.\n"
            "Fix the artifact."
        )
    return value


def _check_version(path: Path, header: dict) -> str:
    """The full version string, once its major is the one that loads."""
    version = str(header["version"])
    head = version.split(".")[0]
    major = int(head) if head.isdigit() else None
    if major in RETIRED_MAJORS:
        raise ValueError(
            f"{path.name} is template version {version}, major {major}, which this tree "
            f"no longer reads: {RETIRED_MAJORS[major]}.\n"
            f"Expected major {TEMPLATE_MAJOR} (config/prompt_template/{TEMPLATE_FILE}).\n"
            f"Re-render the corpus against {TEMPLATE_FILE}; a checkpoint of the old lineage is "
            f"served by the tree that trained it ({SERVED_BY[major]})."
        )
    if major not in TEMPLATE_MAJORS:
        raise ValueError(
            f"{path.name} is template version {version}.\n"
            f"Expected major version {TEMPLATE_MAJOR}.\n"
            "A checkpoint trained against another prefix layout cannot read this one."
        )
    _require_keys(f"{path.name}: template", header, TEMPLATE_KEYS)
    layout = str(header["prefix_layout"])
    if layout != f"v{major}":
        raise ValueError(
            f"{path.name} declares prefix layout {layout!r} at version {version}.\n"
            f"Expected 'v{major}'.\n"
            "The layout and the version name the same thing; make them agree."
        )
    return version


def _check_scaffold(path: Path, scaffold: dict) -> str:
    """Check section 1, and return how the calibration is laid down."""
    if "scale" in scaffold:
        raise ValueError(
            f"{path.name}: scaffold_sections declares a 'scale' block.\n"
            "Expected section 1 only: the <scale> column is retired (id 151690) and "
            "section 2 is the frequency line of the rendered text.\n"
            "Remove the block; the control rate is the frequency slot."
        )
    _refuse_unread_keys(f"{path.name}: scaffold_sections", scaffold, tuple(SCAFFOLD_KEYS))
    numbers = []
    for name, allowed in SCAFFOLD_KEYS.items():
        if name not in scaffold:
            raise ValueError(
                f"{path.name}: scaffold_sections has no {name!r} block.\n"
                f"Expected {sorted(SCAFFOLD_KEYS)}.\n"
                "The encoder lays section 1 down at fixed columns."
            )
        block = _mapping(f"{path.name}: scaffold_sections.{name}", scaffold[name])
        _refuse_unread_keys(f"{path.name}: scaffold_sections.{name}", block, allowed)
        _require_keys(f"{path.name}: scaffold_sections.{name}", block, SCAFFOLD_REQUIRED[name])
        numbers.append(int(block["number"]))
    if tuple(sorted(numbers)) != SCAFFOLD_SECTIONS:
        raise ValueError(
            f"{path.name}: the scaffold owns sections {sorted(numbers)}.\n"
            f"Expected {list(SCAFFOLD_SECTIONS)}.\n"
            "A section that moves across that line changes the prefix width."
        )
    axis = scaffold["axis"]
    geometry = str(axis.get("geometry", ""))
    if geometry in REMOVED_GEOMETRY_LAYOUTS:
        raise ValueError(
            f"{path.name}: section 1 lays the calibration down as {geometry!r}, a layout "
            "this tree no longer builds.\n"
            f"Expected {GEOMETRY_LAYOUTS[0]!r}: {list(REMOVED_GEOMETRY_LAYOUTS)} were "
            "removed.\n"
            f"Use config/prompt_template/{TEMPLATE_FILE}."
        )
    if geometry not in GEOMETRY_LAYOUTS:
        raise ValueError(
            f"{path.name}: section 1 lays the calibration down as {geometry!r}.\n"
            f"Expected one of {list(GEOMETRY_LAYOUTS)}.\n"
            "It names the same thing as model.prefix.intrinsics_mode and the two are "
            "checked against each other."
        )
    expected = PER_VIEW_SLOTS
    declared = tuple(str(slot) for slot in axis["per_view_slots"])
    if declared != expected:
        raise ValueError(
            f"{path.name}: section 1 lays down per-view slots {list(declared)}.\n"
            f"Expected {list(expected)} at geometry {geometry!r}.\n"
            "The order is the column order, so a slot that moves or goes missing moves "
            "every image column after it."
        )
    return geometry


def _check_order(path: Path, order: tuple[str, ...]) -> None:
    """Raise unless the renderer's slots are ones it has a branch for, in place."""
    unknown = [slot for slot in order if slot not in KNOWN_SLOTS]
    if unknown:
        raise ValueError(
            f"{path.name}: order lists {unknown}.\n"
            f"Expected slots from {list(KNOWN_SLOTS)}.\n"
            "A slot the renderer has no branch for would render nothing and say nothing."
        )
    if len(set(order)) != len(order):
        raise ValueError(
            f"{path.name}: order repeats a slot.\n"
            "Expected each slot once.\n"
            "A slot rendered twice states its clause twice."
        )
    required = (*SECTION_HEADERS, "frequency", "instruction", "embodiment_frame_clause")
    for slot in required:
        if slot not in order:
            raise ValueError(
                f"{path.name}: order has no {slot!r}.\n"
                f"Expected every one of {list(required)}.\n"
                "The sections are numbered; one missing renumbers the rest."
            )
    if order[: len(ORDER_OPENING)] != ORDER_OPENING:
        raise ValueError(
            f"{path.name}: order opens with {list(order[:2])}.\n"
            f"Expected {list(ORDER_OPENING)}: the encoder recognises the prompt by its "
            "opening, the frequency line.\n"
            "Put frequency_section and frequency first."
        )
    if order[-1] != "instruction":
        raise ValueError(
            f"{path.name}: order ends with {order[-1]!r}.\n"
            "Expected 'instruction'.\n"
            "Section 5 is the last section of the rendered text; the state line follows it."
        )
    if order.index("embodiment_frame_clause") + 1 != order.index("demonstration_section"):
        raise ValueError(
            f"{path.name}: the frame clause is not the last slot of section 3.\n"
            "Expected it immediately before 'demonstration_section'.\n"
            "A camera-frame clause replaces the frame clause by position."
        )


def _check_sentences(path: Path, sentences: dict, order: tuple[str, ...], joiner: str) -> None:
    """Raise unless every slot that needs a form has one, numbered and opened as held."""
    _refuse_rate_names(f"{path.name}: sentences", sentences)
    _refuse_unread_keys(f"{path.name}: sentences", sentences, SENTENCE_SLOTS)
    missing = [slot for slot in order if slot in SENTENCE_SLOTS and slot not in sentences]
    if missing:
        raise ValueError(
            f"{path.name}: no sentence form for {missing}.\n"
            f"Expected one for each of {list(SENTENCE_SLOTS)} that order lists.\n"
            "The renderer takes these words from the file, not from the code."
        )
    for slot, field in SENTENCE_FIELDS.items():
        if str(sentences[slot]).count(field) != 1:
            raise ValueError(
                f"{path.name}: the {slot!r} form is {sentences[slot]!r}.\n"
                f"Expected {field} in it exactly once.\n"
                "A form with nothing to interpolate renders every sample the same."
            )
    numbers = []
    for slot, number in SECTION_HEADERS.items():
        header = str(sentences[slot])
        if not header.startswith(f"{number}."):
            raise ValueError(
                f"{path.name}: the {slot!r} header is {header!r}.\n"
                f"Expected it to open section {number}.\n"
                "A numbering that shifts between runs is a different prompt."
            )
        numbers.append(number)
    if tuple(sorted(numbers)) != RENDERER_SECTIONS:
        raise ValueError(
            f"{path.name}: the renderer owns sections {sorted(numbers)}.\n"
            f"Expected {list(RENDERER_SECTIONS)}.\n"
            "Section 1 is the encoder's and never passes through the renderer."
        )
    opening = (str(sentences["frequency_section"]) + joiner
               + str(sentences["frequency"]).split("{frequency}")[0])
    if opening != FREQUENCY_OPEN:
        raise ValueError(
            f"{path.name}: the text opens {opening!r}.\n"
            f"Expected FREQUENCY_OPEN, {FREQUENCY_OPEN!r}: the encoder refuses a prompt "
            "that does not open with it.\n"
            "Change neo_vla/constants.py through the contract first, then this file."
        )
    if str(sentences["instruction"]) != INSTRUCTION_OPEN + "{task}":
        raise ValueError(
            f"{path.name}: the instruction form is {sentences['instruction']!r}.\n"
            f"Expected INSTRUCTION_OPEN followed by the task and nothing else, "
            f"{INSTRUCTION_OPEN + '{task}'!r}.\n"
            "Every prompt head ends with it and the tail after the task is empty."
        )


def _check_slots(path: Path, slots: dict, order: tuple[str, ...]) -> None:
    """Raise unless every slot block is one the renderer reads, the frequency's exactly.

    Every slot but the frequency is free text, so a block declaring a closed
    vocabulary is refused rather than ignored: nothing reads one, and a file
    stating a set the renderer does not hold the words to is a set that is not
    closed. The frequency block states the formatter's settings, all of them.
    """
    _refuse_rate_names(f"{path.name}: slots", slots)
    if "frequency" not in slots:
        raise ValueError(
            f"{path.name}: slots has no 'frequency' block.\n"
            f"Expected {FREQUENCY_SLOT}.\n"
            "The block states how the frequency line is written; frequency_text holds it."
        )
    outside = sorted(set(slots) - set(order))
    if outside:
        raise ValueError(
            f"{path.name}: slots declares {outside}, which order does not list.\n"
            f"Expected blocks only for {list(order)}.\n"
            "Remove them: a slot nothing renders is a statement the prompt does not make."
        )
    for name, block in slots.items():
        if "vocab" in _mapping(f"{path.name}: slots.{name}", block):
            raise ValueError(
                f"{path.name}: slots.{name} declares a vocabulary.\n"
                "Expected none: every slot in this layout renders free text.\n"
                "Remove it; nothing reads it, so the file would close a set the code does not."
            )
    frequency = slots["frequency"]
    _refuse_unread_keys(f"{path.name}: slots.frequency", frequency, tuple(FREQUENCY_SLOT))
    for key, held in FREQUENCY_SLOT.items():
        if key not in frequency:
            raise ValueError(
                f"{path.name}: slots.frequency has no {key!r}.\n"
                f"Expected every one of {list(FREQUENCY_SLOT)}.\n"
                "The block states the formatter's settings whole."
            )
        stated = frequency[key]
        if type(stated) is not type(held) or stated != held:
            raise ValueError(
                f"{path.name}: slots.frequency.{key} is {stated!r} and frequency_text "
                f"holds {held!r}.\n"
                "Expected the two to agree: the file states the format so the digest "
                "covers it, and frequency_text writes it.\n"
                "Change both, and re-render every cache stamped with the old digest."
            )


def load_template(path: str | Path) -> Template:
    """Read a template artifact and refuse anything this layout cannot render.

    Raises when the major version is not 7 (naming a retired 4, 5 or 6), when the
    per-view slots are not the ones the geometry lays down, when a second
    control-rate channel is declared, when the frequency block or the opening
    and instruction forms disagree with what the code holds, and when the frame
    clause is not the last renderer slot before the demonstration header.
    """
    path = Path(path)
    raw = path.read_bytes()
    document = _mapping(path.name, yaml.safe_load(raw.decode("utf-8")))
    _refuse_rate_names(path.name, document)
    _refuse_unread_keys(path.name, document, DOCUMENT_KEYS)
    # The header first, and within it the version: a retired major is refused
    # by name before anything its layout lacked is reported as missing.
    _require_keys(path.name, document, ("template",))
    header = _mapping(f"{path.name}: template", document["template"])
    _refuse_unread_keys(f"{path.name}: template", header, TEMPLATE_KEYS)
    _require_keys(f"{path.name}: template", header, ("version",))
    version = _check_version(path, header)
    _require_keys(path.name, document, DOCUMENT_KEYS)
    geometry = _check_scaffold(
        path, _mapping(f"{path.name}: scaffold_sections", document["scaffold_sections"]),
    )
    order = tuple(str(slot) for slot in document["order"])
    _check_order(path, order)
    joiner = str(document["section_joiner"])
    if not joiner:
        raise ValueError(
            f"{path.name}: section_joiner is empty.\n"
            "Expected the string the numbered sections are joined by.\n"
            "Without it the section headers run into the clauses before them."
        )
    sentences = _mapping(f"{path.name}: sentences", document["sentences"])
    _check_sentences(path, sentences, order, joiner)
    _check_slots(path, _mapping(f"{path.name}: slots", document["slots"]), order)
    return Template(
        name=str(header["name"]),
        version=version,
        prefix_layout=str(header["prefix_layout"]),
        geometry=geometry,
        sentences={str(slot): str(form) for slot, form in sentences.items()},
        order=order,
        section_joiner=joiner,
        digest=f"sha256:{hashlib.sha256(raw).hexdigest()}",
    )


def check_mode_agrees(template: Template, intrinsics_mode: str) -> None:
    """Raise when the artifact and model.prefix.intrinsics_mode name two layouts.

    They name the same thing: how many columns sit between <cam_k> and that
    view's picture, and what is in them; a pair that disagreed would put every
    column after the first camera somewhere else than the checkpoint was
    trained to find it. Both are tokens in this tree; a removed mode is refused
    where it is read.
    """
    if template.geometry != str(intrinsics_mode):
        raise ValueError(
            f"the template lays the calibration down as {template.geometry!r} and "
            f"model.prefix.intrinsics_mode is {str(intrinsics_mode)!r}.\n"
            "Expected the two to name the same layout.\n"
            f"config/prompt_template/{TEMPLATE_FILE} is tokens, the one mode; set "
            "model.prefix.intrinsics_mode=tokens."
        )
