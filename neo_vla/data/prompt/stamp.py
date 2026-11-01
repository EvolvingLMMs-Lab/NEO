"""The identity of a set of rendered prompts, and the refusal to mix two.

Rendering every instruction ahead of a run is worth doing once and reading many
times, and a cache of rendered text is only usable by a run that would have
rendered the same text. A stamp is what that means, written next to the cache:
the template and its hash, the tokeniser, the instruction budget and the prefix
layout.

Every field is compared. The previous cache compared the budget alone, so a
cache rendered against a layout with two more slots per view loaded cleanly
into a run that had removed them.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .template import Template

STAMP_FILENAME = "stamp.json"


@dataclass(frozen=True)
class Stamp:
    """What a rendered-prompt cache was built against.

    template_name    the artifact's own name
    template_version its full version string
    template_digest  the hash of the artifact file
    tokenizer_id     the tokeniser directory or repository the ids came from
    max_text_tokens  the budget the text was fitted to
    prefix_layout    the input layout the sections were written for
    n_views          camera slots, which set how many blocks precede the text
    intrinsics_mode  how the calibration enters; tokens, the one mode
    """

    template_name: str
    template_version: str
    template_digest: str
    tokenizer_id: str
    max_text_tokens: int
    prefix_layout: str
    n_views: int
    intrinsics_mode: str


def stamp_of(
    template: Template,
    *,
    tokenizer_id: str,
    max_text_tokens: int,
    prefix_layout: str,
    n_views: int,
    intrinsics_mode: str,
) -> Stamp:
    """The stamp a run with these settings would write.

    Raises when prefix_layout is not the template's own: the sections were
    written for the template's layout, so a stamp naming another would let a
    cache claim a layout its text was never rendered for. The dataset refuses
    such a cache too, but only at the run that reads it.
    """
    if str(prefix_layout) != template.prefix_layout:
        raise ValueError(
            f"the stamp was asked for prefix_layout {str(prefix_layout)!r} and the template "
            f"{template.name} {template.version} is written for {template.prefix_layout!r}.\n"
            "Expected the template's own layout: the rendered sections are that layout's.\n"
            "Pass template.prefix_layout, or render against the template of the layout you mean."
        )
    return Stamp(
        template_name=template.name,
        template_version=template.version,
        template_digest=template.digest,
        tokenizer_id=str(tokenizer_id),
        max_text_tokens=int(max_text_tokens),
        prefix_layout=str(prefix_layout),
        n_views=int(n_views),
        intrinsics_mode=str(intrinsics_mode),
    )


def write(stamp: Stamp, directory: str | Path) -> Path:
    """Write the stamp beside a cache and return the file it wrote."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    written = target / STAMP_FILENAME
    written.write_text(
        json.dumps(asdict(stamp), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return written


def read(directory: str | Path) -> Stamp:
    """Read the stamp beside a cache.

    Raises when there is none. An unstamped cache is one whose provenance was
    never recorded, and it cannot be checked into agreement afterwards.
    """
    found = Path(directory) / STAMP_FILENAME
    if not found.is_file():
        raise FileNotFoundError(
            f"{directory} holds no {STAMP_FILENAME}.\n"
            "Expected the stamp written beside the rendered prompts.\n"
            "Render them again; a cache with no stamp has no provenance to check."
        )
    document = json.loads(found.read_text(encoding="utf-8"))
    missing = [field.name for field in fields(Stamp) if field.name not in document]
    if missing:
        raise ValueError(
            f"{found} has no {missing}.\n"
            f"Expected all of {[field.name for field in fields(Stamp)]}.\n"
            "Render the prompts again with this version of the stamp."
        )
    return Stamp(
        template_name=str(document["template_name"]),
        template_version=str(document["template_version"]),
        template_digest=str(document["template_digest"]),
        tokenizer_id=str(document["tokenizer_id"]),
        max_text_tokens=int(document["max_text_tokens"]),
        prefix_layout=str(document["prefix_layout"]),
        n_views=int(document["n_views"]),
        intrinsics_mode=str(document["intrinsics_mode"]),
    )


def check(found: Stamp, expected: Stamp) -> None:
    """Raise unless every field matches, naming the first that does not.

    All of them, not a chosen few. A field worth writing into the stamp is one
    whose change makes the cached text wrong, so there is no field here that a
    run may differ on and still read the cache.
    """
    for field in fields(Stamp):
        mine = getattr(expected, field.name)
        theirs = getattr(found, field.name)
        if mine != theirs:
            raise ValueError(
                f"the cache's {field.name} is {theirs!r}, this run has {mine!r}.\n"
                "Expected every field of the stamp to match.\n"
                "Render the prompts again against this template, tokeniser and budget."
            )
