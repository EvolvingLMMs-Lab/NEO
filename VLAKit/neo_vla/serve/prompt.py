"""The policy prompt for one robot, rendered the way training rendered it.

Training renders sections 2 to 5 of config/prompt_template/policy.yaml from the
embodiment's clause and frame clause in config/embodiments.yaml and the
directory's control frequency. Rendering the same three inputs here gives the
same head byte for byte, which is what a step's trained prompt heads list.
"""
from __future__ import annotations

from pathlib import Path

from ..data.prompt.embodiments import _table
from ..data.prompt.render import RenderContext, render
from ..data.prompt.template import load_template

TEMPLATE = Path(__file__).resolve().parents[2] / "config" / "prompt_template" / "policy.yaml"


def frame_clause(key: str, frame_ref: str, *, source: str | None = None) -> str:
    """The sentence naming the action frame, for a camera or the robot base."""
    table = _table()
    if key not in table["keys"]:
        raise ValueError(f"embodiment key {key!r} is not in config/embodiments.yaml.")
    clauses = dict(table["frame_clause"])
    clauses.update(table.get("frame_clause_by_key", {}).get(key, {}))
    clauses.update(table["keys"][key].get("frame_clause_by_source", {}).get(source, {}))
    kind = "robot" if str(frame_ref).strip() == "robot" else "camera"
    if not clauses.get(kind):
        raise ValueError(f"{key!r} has no {kind} frame clause; it was not trained in that frame.")
    return clauses[kind]


def render_prompt(key: str, frame_ref: str, control_hz: float, task: str, *,
                  source: str | None = None, template: str | Path = TEMPLATE) -> str:
    """The full prompt text for one task, ready for the request's 'prompt'.

    source picks a source-specific frame clause where config/embodiments.yaml
    has one (DROID's camera sentence); leave it None otherwise.
    """
    clause = frame_clause(key, frame_ref, source=source)
    context = RenderContext(
        task=task,
        embodiment_clauses=(_table()["keys"][key]["clause"],),
        frame_clause=clause,
        frequency=float(control_hz),
    )
    return render(load_template(template), context).text
