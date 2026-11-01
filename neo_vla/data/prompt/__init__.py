"""The prompt: the template artifact, the renderer, and the cache's identity.

Sections 2 to 5 are this package's text -- the frequency line, the embodiment
and frame clauses, the empty demonstration header and the instruction -- and
the stamp says which template, tokeniser and budget a set of rendered prompts
was built against.

Section 1 belongs to the encoder's scaffold and does not pass through here.
"""
from .render import DROP_ORDER, RenderContext, Rendered, fit, render
from .stamp import Stamp, stamp_of
from .template import (
    FREQUENCY_OPEN,
    INSTRUCTION_OPEN,
    TEMPLATE_FILE,
    Template,
    check_mode_agrees,
    frequency_text,
    load_template,
)

__all__ = [
    "DROP_ORDER",
    "FREQUENCY_OPEN",
    "INSTRUCTION_OPEN",
    "TEMPLATE_FILE",
    "RenderContext",
    "Rendered",
    "Stamp",
    "Template",
    "check_mode_agrees",
    "fit",
    "frequency_text",
    "load_template",
    "render",
    "stamp_of",
]
