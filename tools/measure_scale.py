"""Stage 5: the scale channel, one number per dataset, with its method beside it.

The 99th percentile of the per-step translation magnitude in centimetres, over
the kept episodes, pooled over arms so that one rule covers a one-armed and a
two-armed source alike. The method is recorded next to every value, and a
dataset with no computable scale gets null, which reaches the model as the
tokenizer's unknown row.

The median is refused: about half of all rows are still, so it comes out near
zero. So is any per-chunk statistic, because a scale computed from the chunk
being predicted states the target's magnitude and does not exist at serving
time. Episodes are the sampling unit and the error bar is a bootstrap over them.

Usage:
    python tools/measure_scale.py --root /corpus --inventory prep/inventory.jsonl \
        --kept prep/kept.jsonl --layouts config/layouts/gd7.json --out prep/scale.json
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import layouts as layout_table
import numpy as np
import stage
import unify

logger = logging.getLogger("measure_scale")

REPO = Path(__file__).resolve().parent.parent

METHOD = "q99 of the per-step translation magnitude in cm, arms pooled, kept episodes"

# Under this magnitude a step is still. Reported, never subtracted: the still
# rows are half the corpus and dropping them would change the statistic.
STILL_CM = 1e-3


def bootstrap(per_episode: list[np.ndarray], quantile: float, draws: int, seed: str) -> dict:
    """The relative half-width of the value's 95 percent interval over episodes."""
    if len(per_episode) < 2 or draws <= 0:
        return {}
    generator = np.random.default_rng(unify.stable_seed(seed))
    values = np.empty(draws)
    for index in range(draws):
        chosen = generator.integers(0, len(per_episode), len(per_episode))
        values[index] = np.percentile(
            np.concatenate([per_episode[position] for position in chosen]), quantile
        )
    low, high = float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))
    centre = float(np.percentile(np.concatenate(per_episode), quantile))
    return {
        "interval_95": [round(low, 6), round(high, 6)],
        "relative_half_width": round((high - low) / 2.0 / max(centre, 1e-12), 6),
        "draws": draws,
    }


def measure(
    root: Path,
    episodes: list[dict[str, Any]],
    layout: layout_table.Layout,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """One directory's kept episodes -> its scale, or null with the reason."""
    if not unify.factors_stated(layout.action):
        return {"scale_cm_per_step": None, "reason": "unit_factor_unestablished"}
    drawn = unify.sample(episodes, args.max_episodes, f"{args.seed}:{layout.name}")
    per_episode: list[np.ndarray] = []
    rows = 0
    for _, arms in unify.iter_blocks(root, drawn, layout, args.max_rows):
        magnitudes = np.concatenate([np.linalg.norm(arm[:, 0:3], axis=1) for arm in arms])
        per_episode.append(magnitudes)
        rows += int(magnitudes.shape[0])
    if not per_episode:
        return {"scale_cm_per_step": None, "reason": "no_kept_episodes"}
    pooled = np.concatenate(per_episode)
    return {
        "scale_cm_per_step": round(float(np.percentile(pooled, args.quantile)), 6),
        "method": METHOD,
        "quantile": args.quantile,
        "arms": len(layout.action["arms"]),
        "episodes": len(drawn),
        "episodes_available": len(episodes),
        "rows": rows,
        "still_fraction": round(float((pooled < STILL_CM).mean()), 6),
        "max_cm": round(float(pooled.max()), 6),
        "bootstrap": bootstrap(per_episode, args.quantile, args.bootstrap,
                               f"{args.seed}:{layout.name}"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--kept", required=True)
    parser.add_argument("--layouts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--quantile", type=float, default=99.0)
    parser.add_argument("--max-episodes", type=int, default=400)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", default="scale")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    root = stage.resolve_root(args.root)
    inventory_path = stage.under_root(root, args.inventory)
    kept_path = stage.under_root(root, args.kept)
    layouts_path = stage.under_root(root, args.layouts)
    out = stage.under_root(root, args.out)

    _, inventory = stage.read_jsonl(inventory_path)
    _, kept = stage.read_jsonl(kept_path)
    document = layout_table.load(layouts_path)
    grouped = unify.kept_by_dataset(kept, inventory)
    stage.refuse_write_into_input(out, [root / name for name in grouped])

    datasets: dict[str, Any] = {}
    for directory in sorted(grouped):
        layout = layout_table.resolve(document, directory)
        datasets[layout.name] = {"directory": directory, **measure(
            root, grouped[directory], layout, args)}

    payload = {
        "provenance": stage.header(
            "measure_scale", root=root,
            inputs={"inventory": inventory_path, "kept": kept_path,
                    "layouts": layouts_path},
            tree=REPO,
            extra={"quantile": args.quantile, "max_episodes": args.max_episodes,
                   "bootstrap": args.bootstrap, "seed": args.seed,
                   "refused": ["median", "per_chunk"]},
        ),
        "datasets": datasets,
    }
    written = stage.write_json(out, payload)
    known = sum(1 for entry in datasets.values() if entry["scale_cm_per_step"] is not None)
    logger.info("%d datasets, %d with a scale, %d unknown",
                len(datasets), known, len(datasets) - known)
    logger.info("%s %s", "wrote" if written else "unchanged", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
