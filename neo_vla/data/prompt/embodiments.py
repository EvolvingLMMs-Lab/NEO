"""Canonical prompt clauses and the refusal of divergent manifest copies.

The YAML is the authority; manifests and exported frames are frozen copies.
Checking them never rewrites a manifest, prompt cache, normaliser or stamp.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import yaml

TABLE = Path(__file__).resolve().parents[3] / "config" / "embodiments.yaml"
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _table() -> dict:
    table = yaml.safe_load(TABLE.read_text(encoding="utf-8"))
    keys, fields = table["keys"], set(table["frame_clause"])
    overrides = table.get("frame_clause_by_key", {})
    unknown = set(overrides) - set(keys)
    if unknown:
        raise ValueError(f"frame_clause_by_key names unknown embodiment keys: {sorted(unknown)}")
    for key, spec in keys.items():
        clauses = {"frame_clause_by_key": overrides.get(key, {}),
                   **spec.get("frame_clause_by_source", {})}
        for source, values in clauses.items():
            if not isinstance(values, dict) or set(values) - fields:
                raise ValueError(f"{key}/{source}: frame clause fields must be {sorted(fields)}")
    return table


def canonical_prompt(entry: dict) -> dict:
    """The entry's clauses from YAML, including its source-specific frame spelling.

    DROID's historic camera sentence differs from other franka_panda sources.
    Its source selects that spelling, never the manifest's proposed text.
    """
    table = _table()
    block = entry["embodiment"]
    keys = [block["key"]] if block["kind"] == "constant" else block["keys"]
    clauses, frames = {}, None
    for key in keys:
        if key not in table["keys"]:
            raise ValueError(
                f"{entry.get('name', '<directory>')}: embodiment key {key!r} is absent from "
                "config/embodiments.yaml; add its clause, frame clause and normaliser "
                "declarations there before using it in an official corpus.")
        spec = table["keys"][key]
        clauses[key] = spec["clause"]
        own = dict(table["frame_clause"])
        own.update(table.get("frame_clause_by_key", {}).get(key, {}))
        own.update(spec.get("frame_clause_by_source", {}).get(entry.get("source"), {}))
        if frames is not None and frames != own:
            raise ValueError(f"{entry.get('name')}: its embodiment keys disagree on frame "
                             "clauses in config/embodiments.yaml")
        frames = own
    if not clauses:
        raise ValueError(f"{entry.get('name')}: no embodiment keys for the official prompt")
    return {"embodiment_clauses": clauses, "frame_clause": frames}


def validate_manifest_prompts(document: dict) -> int:
    """Refuse any supplied clause that differs byte for byte; count redundant copies.

    A manifest need not repeat the canonical clauses. If it does, each supplied
    field is an assertion, including unused frame clauses and per-episode keys.
    The check is independent of the manifest's claimed template or digest.
    """
    redundant = 0
    for entry in document.get("datasets", []):
        if "prompt" not in entry:
            continue
        supplied = entry["prompt"]
        expected = canonical_prompt(entry)
        if not isinstance(supplied, dict) or set(supplied) - set(expected):
            raise ValueError(f"{entry.get('name')}: invalid manifest prompt override; "
                             "expected embodiment_clauses and frame_clause mappings")
        for field, values in supplied.items():
            if not isinstance(values, dict):
                raise ValueError(f"{entry.get('name')}: prompt.{field} must be a mapping")
            for key, value in values.items():
                if key not in expected[field] or value != expected[field][key]:
                    raise ValueError(
                        f"{entry.get('name')}: manifest prompt.{field}[{key!r}] override "
                        f"{value!r} differs from config/embodiments.yaml "
                        f"({expected[field].get(key)!r}). Official prompt clauses must be "
                        "byte-identical to the canonical table; add or correct the key "
                        "there, never override its text in a manifest.")
                redundant += 1
    if redundant:
        logger.info("%d redundant manifest prompt clause overrides match "
                    "config/embodiments.yaml byte for byte", redundant)
    return redundant
