"""One sample's text: sections 2 to 5 of the template, and nothing else.

The encoder lays section 1 down, then the text this module returns, then the
state line and the close, so the string produced here is the whole of the
sample's variable text and it has to fit the instruction budget. Nothing is
ever dropped to make it fit: a record over the budget raises, naming its
directory and episode, because a clause that is there in one sample and gone
in the next is a prompt the model has to learn to read two ways.

The text states one number about the robot, the control frequency, and states
it once. Geometry reaches the model through section 1's projected columns, and
the state reaches it through the state columns after the text; a clause here that named a pose
would be a second statement of it that no mask can retract.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .template import SECTION_HEADERS, Template, frequency_text

# Clauses dropped when the rendered text does not fit: none. Version 5 dropped
# the frame clause, then the embodiment; under version 6 the frequency, the
# embodiment and the frame all name what the actions mean, so an over-budget
# record is refused at the cache build instead.
DROP_ORDER: tuple[str, ...] = ()

# The clause separator, applied inside a section only. A clause that already
# ends in punctuation keeps it; one that does not gets a full stop, so the
# sentences of section 3 read as sentences however a source worded them.
TERMINAL_CHARS = (".", "!", "?", ":", ";", ",")
CLAUSE_JOINER = " "


@dataclass(frozen=True)
class RenderContext:
    """Everything one sample contributes to the text, already resolved.

    task                 the instruction, as recorded or as augmented
    embodiment_clauses   the robot description of the sample's embodiment key
    frame_clause         the sentence naming the frame the actions are in
    frequency            the directory's control_hz in Hz, or None for "NA";
                         required, so a caller cannot forget the line

    Every field is a value the caller has already chosen. The renderer makes
    no decisions about content; it joins, and it formats the frequency.
    """

    task: str
    embodiment_clauses: tuple[str, ...]
    frame_clause: str
    frequency: float | None


@dataclass(frozen=True)
class Rendered:
    """One sample's text and its two halves around the task.

    text  head + task + tail, what the encoder tokenises
    head  everything before the task; it ends with INSTRUCTION_OPEN and is the
          same for every episode of a directory and frame, which is what the
          checkpoint's trained prompt heads list
    task  the instruction as rendered
    tail  everything after the task; empty under version 6
    """

    text: str
    head: str
    task: str
    tail: str


def _joined(current: str, clause: str) -> str:
    """The clause separator rule, in the one place it is applied."""
    base = current.rstrip()
    if not base:
        return clause
    stop = "" if base.endswith(TERMINAL_CHARS) else "."
    return f"{base}{stop}{CLAUSE_JOINER}{clause}"


def _check_clauses(context: RenderContext) -> None:
    """Raise unless section 3 has its embodiment clause and its frame clause.

    Version 6 drops no clause, so an empty one is never a shorter prompt that
    is still right: it is a head without the sentence that names the sample's
    normaliser or its frame, which can then equal another key's or another
    frame's head.
    """
    clauses = tuple(context.embodiment_clauses)
    if not clauses or any(not str(clause).strip() for clause in clauses):
        raise ValueError(
            f"the sample's embodiment clauses are {list(clauses)!r}.\n"
            "Expected at least one embodiment clause, none of them empty: version 6 drops "
            "no clause, and the clause names the embodiment key.\n"
            "Give the key's clause from config/embodiments.yaml."
        )
    if not str(context.frame_clause).strip():
        raise ValueError(
            f"the sample's frame clause is {context.frame_clause!r}.\n"
            "Expected the sentence naming the frame the actions are written in.\n"
            "Give the layout's frame_clause for the sample's frame (robot or camera)."
        )


def render(template: Template, context: RenderContext) -> Rendered:
    """A context -> the text, unfitted, split around its task.

    The task is rendered without trailing whitespace: the prompt ends with
    the task and the encoder counts the <state> columns from
    the text's length, so a raw instruction ending in a stray space or
    newline gives the same text through every producer (the cache build, the
    eval frames, the server's clients), and the encoder refuses a prompt that
    ends in whitespace. Leading whitespace is kept: real records open with a
    newline (the gd5 cache's longest, utaustin_mutex position 57), and v6 is
    the frequency line plus the v5 text byte for byte.

    Raises for an empty task, an empty embodiment or frame clause, and a
    frequency frequency_text refuses. The result still has to pass fit before
    it reaches the encoder, which refuses a prompt over the budget rather than
    truncating it.
    """
    task = context.task.rstrip()
    if not task.strip():
        raise ValueError(
            "the sample's task is empty.\n"
            "Expected the instruction the episode was recorded under.\n"
            "Fix the directory's instruction in the manifest; a prompt with no task is a defect."
        )
    _check_clauses(context)
    sections: list[str] = []
    clauses = ""
    head = tail = None
    for slot in template.order:
        if slot in SECTION_HEADERS:
            # A numbered header opens a section, so whatever was accumulating
            # under the previous one is complete. Section 4's body is empty
            # until in-context demonstrations exist and renders nothing at all;
            # a placeholder there is a string the model would have to unlearn.
            if clauses:
                sections.append(clauses)
                clauses = ""
            sections.append(template.sentences[slot])
        elif slot == "frequency":
            if clauses:
                sections.append(clauses)
                clauses = ""
            sections.append(
                template.sentences[slot].format(frequency=frequency_text(context.frequency))
            )
        elif slot == "embodiment":
            for clause in context.embodiment_clauses:
                clauses = _joined(clauses, clause)
        elif slot == "embodiment_frame_clause":
            clauses = _joined(clauses, context.frame_clause)
        elif slot == "instruction":
            if clauses:
                sections.append(clauses)
                clauses = ""
            before, after = template.sentences[slot].split("{task}")
            head = template.section_joiner.join([*sections, ""]) + before
            tail = after
            sections.append(before + task + after)
        else:
            raise RuntimeError(
                f"the template admitted slot {slot!r} and this renderer has no branch.\n"
                "Expected every slot in KNOWN_SLOTS to have one.\n"
                "Add the branch here, or remove the slot from template.KNOWN_SLOTS."
            )
    if clauses or head is None:
        raise RuntimeError(
            "the template's order does not end with the instruction.\n"
            "Expected load_template to have refused it.\n"
            "Load templates through load_template only."
        )
    text = template.section_joiner.join(sections)
    if text != head + task + tail:
        raise RuntimeError(
            "the rendered text is not its head, task and tail.\n"
            "Expected the instruction to be the last section.\n"
            "Load templates through load_template only."
        )
    return Rendered(text=text, head=head, task=task, tail=tail)


def fit(
    template: Template,
    context: RenderContext,
    *,
    max_text_tokens: int,
    count: Callable[[str], int],
    directory: str | None = None,
    episode: int | None = None,
) -> Rendered:
    """Render, and raise when the text does not fit the budget; never drop.

    count is the caller's tokeniser (NEOpp-2B-SFT's, add_special_tokens=False,
    on the text alone), taken as an argument so this module does not load one
    of its own and measure the budget with a different vocabulary from the
    encoder. directory and episode name the record in the refusal.
    """
    rendered = render(template, context)
    length = int(count(rendered.text))
    if length > max_text_tokens:
        record = f"{directory or 'a directory'} episode {episode if episode is not None else '?'}"
        raise ValueError(
            f"the rendered text of {record} is {length} tokens against a budget of "
            f"{max_text_tokens}.\n"
            "Expected every record to fit whole: version 6 drops no clause.\n"
            "Shorten the instruction in the manifest, or raise model.prefix.max_text_tokens "
            "(which widens the prefix and needs a new pretrain)."
        )
    return rendered
