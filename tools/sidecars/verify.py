"""Re-read every written sidecar against its directory and report coverage per family.

Per entry: the stamp and rows check of common.read_sidecar; the directory's
episodes and rows recounted from its own parquet; no arm whose joint block is
all exactly 0 on any row; no joint reading beyond common.JOINT_CEILING_RAD,
and the in-episode joint steps past pi counted (the steps build_stats' wrap
gate masks a joint for); for OXE group A, every parquet
row whose joint slots are all 0 is NaN in the sidecar. --sample-episodes N
re-reads N random AgiBot and DROID episodes from their raw files and compares
the sidecar values with them, and an AgiBot episode without odometry with an
all-NaN base. The report is a json under <out>/_reports with per-family totals.
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing
import random
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path[:0] = [str(Path(__file__).resolve().parents[2]),
                    str(Path(__file__).resolve().parents[1])]

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import stage  # noqa: E402

from sidecars import agibot, common, droid, oxe  # noqa: E402
from sidecars.run import family_of, read_entries, check_tree  # noqa: E402

logger = logging.getLogger("sidecars.verify")


def column(table, name: str) -> np.ndarray:
    """A fixed-width list column as [rows, width], slices included."""
    values = table.column(name).combine_chunks()
    return values.flatten().to_numpy(zero_copy_only=False).reshape(len(values), -1)


def zero_joint_rows(joints: np.ndarray, arms: int) -> int:
    """Rows on which some arm's joint values are all exactly 0 (its NaN set aside)."""
    blocks = joints.reshape(joints.shape[0], arms, -1)
    present = np.isfinite(blocks)
    zero = (np.where(present, blocks, 0.0) == 0.0).all(axis=2) & present.any(axis=2)
    return int(zero.any(axis=1).sum())


def joint_steps_past_pi(joints: np.ndarray, episodes: np.ndarray) -> int:
    """Consecutive rows of one episode whose joint reading steps by more than pi."""
    if joints.shape[0] < 2:
        return 0
    step = np.abs(joints[1:].astype(np.float64) - joints[:-1].astype(np.float64))
    inside = (episodes[1:] == episodes[:-1])[:, None]
    with np.errstate(invalid="ignore"):
        return int((np.isfinite(step) & inside & (step > np.pi)).sum())


def check_entry(entry: dict[str, Any], root: Path, out: Path) -> dict[str, Any]:
    """One entry's sidecar against its directory, with its coverage counts."""
    name = entry["name"]
    path = common.sidecar_path(out, name)
    if not path.exists():
        return {"name": name, "family": family_of(entry), "status": "missing"}
    meta, table = common.read_sidecar(path)
    directory = root / entry["path"]
    episodes = common.directory_episodes(directory)
    problems = []
    aligned = ([e.index for e in episodes] == meta["episodes"]
               and [e.rows for e in episodes] == meta["episode_rows"])
    if not aligned:
        problems.append("episodes or rows differ from the directory")
    arms = int(meta["arms"])
    joints = column(table, common.COLUMN_JOINTS)
    body = column(table, common.COLUMN_BODY)
    zero_rows = zero_joint_rows(joints, arms)
    if zero_rows:
        problems.append(f"{zero_rows} rows hold an all-zero joint block")
    with np.errstate(invalid="ignore"):
        beyond = int((np.abs(joints) > common.JOINT_CEILING_RAD).sum())
    if beyond:
        problems.append(f"{beyond} joint readings lie beyond {common.JOINT_CEILING_RAD:.4f} rad")
    result = {
        "name": name, "family": meta["family"], "rows": meta["rows"],
        "episodes": len(meta["episodes"]), "arms": arms,
        "joint_rows": [int(np.isfinite(joints.reshape(-1, arms, 7)[:, arm]).any(axis=1).sum())
                       for arm in range(arms)],
        "joint_values": int(np.isfinite(joints).sum()),
        "joint_values_beyond_ceiling": beyond,
        "joint_steps_past_pi": joint_steps_past_pi(
            joints, table.column(common.EPISODE_COLUMN).to_numpy()),
        "body_rows": {"base": int(np.isfinite(body[:, 0:3]).all(axis=1).sum()),
                      "torso": int(np.isfinite(body[:, 3:6]).any(axis=1).sum()),
                      "head": int(np.isfinite(body[:, 6:8]).any(axis=1).sum())},
        "hand_values": int(np.isfinite(column(table, common.COLUMN_HAND)).sum()),
        "checks": meta["checks"],
    }
    if common.COLUMN_JOINTS_COMMAND in meta["columns"]:
        result["command_rows"] = int(np.isfinite(
            column(table, common.COLUMN_JOINTS_COMMAND)).any(axis=1).sum())
    if name in oxe.JOINT_STATE and aligned:
        count = oxe.JOINT_STATE[name]
        state = np.concatenate([np.stack(pq.read_table(e.data_file, columns=[
            "observation.state"]).column(0).to_numpy(zero_copy_only=False)) for e in episodes])
        zero = (state[:, :count] == 0.0).all(axis=1)
        result["parquet_zero_rows"] = int(zero.sum())
        if not np.isnan(joints[zero]).all():
            problems.append("a parquet row with all joint slots 0 is not NaN")
        if not np.array_equal(joints[~zero, :count], state[~zero, :count].astype(np.float32)):
            problems.append("joint slots differ from observation.state")
    result["status"] = "problem" if problems else "ok"
    result["problems"] = problems
    return result


def verify_one(job: tuple[dict[str, Any], str, str]) -> dict[str, Any]:
    """check_entry for a pool, with an unreadable sidecar reported, not raised."""
    entry, root, out = job
    try:
        return check_entry(entry, Path(root), Path(out))
    except (ValueError, OSError, KeyError) as error:
        return {"name": entry["name"], "family": family_of(entry), "status": "problem",
                "problems": [str(error)], "rows": 0, "joint_rows": [0], "joint_values": 0,
                "hand_values": 0, "body_rows": {"base": 0, "torso": 0, "head": 0}}


def sample_agibot(entries: list[dict[str, Any]], root: Path, out: Path, raw: Path,
                  count: int, rng: random.Random) -> dict[str, Any]:
    """Raw h5 against the sidecar for random episodes, and every no-odometry one drawn."""
    compared = no_odometry = 0
    problems = []
    for entry in rng.sample(entries, min(len(entries), count)):
        meta, table = common.read_sidecar(common.sidecar_path(out, entry["name"]))
        position = rng.randrange(len(meta["episodes"]))
        first = sum(meta["episode_rows"][:position])
        rows = meta["episode_rows"][position]
        record = common.directory_episodes(root / entry["path"])[position].record
        with h5py.File(raw / record["raw_file_name"] / "proprio_stats.h5", "r") as handle:
            joints = handle[agibot.JOINTS][()].astype(np.float32)
            waist = handle[agibot.WAIST][()].astype(np.float32)
            head = handle[agibot.HEAD][()].astype(np.float32)
            quaternion = (handle[agibot.ODOMETRY_ORIENTATION][()]
                          if handle[agibot.ODOMETRY_ORIENTATION].size else np.zeros((rows, 4)))
        piece = table.slice(first, rows)
        side_joints = column(piece, common.COLUMN_JOINTS)
        side_body = column(piece, common.COLUMN_BODY)
        where = f"{entry['name']} episode {meta['episodes'][position]}"
        if not np.array_equal(side_joints, joints):
            problems.append(f"{where}: joints differ from the h5")
        if not (np.array_equal(side_body[:, 3:5], waist)
                and np.array_equal(side_body[:, 6:8], head)):
            problems.append(f"{where}: waist or head differ from the h5")
        unread = np.linalg.norm(quaternion, axis=1) <= agibot.QUATERNION_FLOOR
        if unread.all():
            no_odometry += 1
            if not np.isnan(side_body[:, 0:3]).all():
                problems.append(f"{where}: no odometry but a base value is not NaN")
        if not np.isnan(side_body[0, 0:3]).all():
            problems.append(f"{where}: row 0 has a base value")
        compared += 1
    return {"episodes": compared, "without_odometry": no_odometry, "problems": problems}


def sample_droid(entries: list[dict[str, Any]], root: Path, out: Path, raw: Path | None,
                 count: int, rng: random.Random) -> dict[str, Any]:
    """Raw trajectories against the sidecar for random episodes."""
    compared = 0
    problems = []
    for entry in rng.sample(entries, min(len(entries), count)):
        directory = root / entry["path"]
        base = raw if raw is not None else droid.raw_root_of(directory)
        meta, table = common.read_sidecar(common.sidecar_path(out, entry["name"]))
        position = rng.randrange(len(meta["episodes"]))
        first = sum(meta["episode_rows"][:position])
        rows = meta["episode_rows"][position]
        record = common.directory_episodes(directory)[position].record
        with h5py.File(base / record["droid_session"] / droid.TRAJECTORY_FILE, "r") as handle:
            state = handle[droid.JOINT_STATE][:rows].astype(np.float32)
            command = handle[droid.JOINT_COMMAND][:rows].astype(np.float32)
        piece = table.slice(first, rows)
        if not (np.array_equal(column(piece, common.COLUMN_JOINTS), state)
                and np.array_equal(column(piece, common.COLUMN_JOINTS_COMMAND), command)):
            problems.append(f"{entry['name']} episode {meta['episodes'][position]} differs")
        compared += 1
    return {"episodes": compared, "problems": problems}


def totals(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Coverage per family: directories, rows, and the rows each column holds."""
    families: dict[str, dict[str, Any]] = {}
    for result in results:
        family = families.setdefault(result["family"], {
            "directories": 0, "ok": 0, "missing": 0, "problem": 0, "rows": 0,
            "joint_rows_any_arm": 0, "joint_rows_per_arm": [], "base_rows": 0,
            "torso_rows": 0, "head_rows": 0, "command_rows": 0, "hand_values": 0,
            "joint_values_beyond_ceiling": 0, "joint_steps_past_pi": 0})
        family["directories"] += 1
        family[result["status"]] += 1
        if result["status"] == "missing":
            continue
        family["rows"] += result["rows"]
        per_arm = family["joint_rows_per_arm"]
        per_arm.extend([0] * (len(result["joint_rows"]) - len(per_arm)))
        for arm, rows in enumerate(result["joint_rows"]):
            per_arm[arm] += rows
        family["joint_rows_any_arm"] += max(result["joint_rows"])
        family["base_rows"] += result["body_rows"]["base"]
        family["torso_rows"] += result["body_rows"]["torso"]
        family["head_rows"] += result["body_rows"]["head"]
        family["command_rows"] += result.get("command_rows", 0)
        family["hand_values"] += result["hand_values"]
        family["joint_values_beyond_ceiling"] += result.get("joint_values_beyond_ceiling", 0)
        family["joint_steps_past_pi"] += result.get("joint_steps_past_pi", 0)
    return families


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out")
    parser.add_argument("--families", default="agibot,droid,rh20t,oxe")
    parser.add_argument("--sample-episodes", type=int, default=0)
    parser.add_argument("--agibot-raw")
    parser.add_argument("--droid-raw")
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    check_tree()
    root = stage.resolve_root(args.root)
    out = Path(args.out) if args.out else common.default_out(root)
    chosen = {name.strip() for name in args.families.split(",")}
    entries = [entry for entry in read_entries(root, Path(args.manifest))
               if family_of(entry) in chosen]
    jobs = [(entry, str(root), str(out)) for entry in
            sorted(entries, key=lambda item: -int(item.get("rows", 0)))]
    if args.workers > 1:
        with multiprocessing.get_context("fork").Pool(args.workers) as pool:
            results = pool.map(verify_one, jobs, chunksize=1)
    else:
        results = [verify_one(job) for job in jobs]
    samples = {}
    if args.sample_episodes:
        rng = random.Random(args.seed)
        if "agibot" in chosen and args.agibot_raw:
            samples["agibot"] = sample_agibot(
                [e for e in entries if family_of(e) == "agibot"], root, out,
                Path(args.agibot_raw), args.sample_episodes, rng)
        if "droid" in chosen:
            samples["droid"] = sample_droid(
                [e for e in entries if family_of(e) == "droid"], root, out,
                Path(args.droid_raw) if args.droid_raw else None, args.sample_episodes, rng)
    report = {"tool_version": common.TOOL_VERSION, "entries": len(entries),
              "families": totals(results), "samples": samples,
              "results": sorted(results, key=lambda item: item["name"])}
    path = out / "_reports" / "verify.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stage.dumps_json(report), encoding="utf-8")
    for name, family in sorted(report["families"].items()):
        logger.info("%s %s", name, {k: v for k, v in family.items()})
    for name, sample in samples.items():
        logger.info("sample %s: %d episodes, %d problems", name, sample["episodes"],
                    len(sample["problems"]))
    bad = [r for r in results if r["status"] != "ok"]
    bad_samples = sum(len(sample["problems"]) for sample in samples.values())
    logger.info("%d of %d entries ok; %d not; %d sample problems; report %s",
                len(results) - len(bad), len(results), len(bad), bad_samples, path)
    return 1 if bad or bad_samples else 0


if __name__ == "__main__":
    sys.exit(main())
