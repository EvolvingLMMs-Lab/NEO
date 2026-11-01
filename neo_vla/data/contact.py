"""Contact-focused window starts: an opt-in reweighting of where a window opens in its episode.

Off by default (data.train.contact_sampling null, or enabled false): nothing here is built, no
row is read for it, and the draw is the one neo_vla.data.mixture always made. On, the
directory, its quota, the episode, the wrap and the pass are the draw's own; only the start
inside the chosen episode moves, to the inverse CDF of a box weight: strength on every start
whose window holds a commanded gripper close or open, 1 elsewhere, normalised inside that
episode. The signal is the manifest's action gripper column read through its raw_open and
raw_closed polarity, so an event is a command, not a sensed contact. An episode's events are
found on its first draw in a process and kept in a store bounded by bytes.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .reader import ENTRY_BYTES, FileCache

logger = logging.getLogger(__name__)

WHERE = "data.train.contact_sampling"

# The two command events: close is the first closed row of a closed run (a grasp
# command), open the first open row after one (a release command).
EVENTS = ("close", "open")

# What a directory the signal cannot speak for does: refuse the run, or draw it as if off.
UNSUPPORTED = ("error", "skip")

KEYS = ("enabled", "strength", "windows", "close_at", "open_at", "directories", "overrides",
        "unsupported", "max_events", "cache_bytes")
OVERRIDE_KEYS = ("strength", "windows", "close_at", "open_at")

# The canonical gripper midpoint: one threshold, no hysteresis, unless the group states two.
THRESHOLD = 0.5

# More events than this in one episode, closes and opens summed over all its arms, is read as
# a chattering command, not as that many grasps: the episode keeps its uniform start, counted
# and said once per directory. A two-arm directory reaches it at about half the per-arm count.
MAX_EVENTS = 64

# The event store's budget per process. An episode is charged ENTRY_BYTES plus INTERVAL_BYTES
# a merged window (a tuple slot and two ints, rounded up), so a two-event episode is about
# 512 bytes and the default holds some 16,000 of them: at most 0.5 GiB over a node's 64 workers.
CACHE_BYTES = 8 * 1024 * 1024
INTERVAL_BYTES = 96
MIN_CACHE_BYTES = 4 * ENTRY_BYTES


@dataclass(frozen=True)
class EventRule:
    """How one directory's starts are weighted: the strength, the windows and the thresholds.

    windows holds (event, before, after) for each event kind used, in EVENTS order: a start t
    is in the window of an event at row e when e - before <= t <= e + after. A command enters
    closed at close_at and leaves it below open_at (canonical, 0 open and 1 closed).
    """

    strength: float
    windows: tuple[tuple[str, int, int], ...]
    close_at: float
    open_at: float


@dataclass(frozen=True)
class ContactConfig:
    """The data group's contact_sampling block, checked; None from from_config when absent."""

    enabled: bool
    rule: EventRule
    directories: tuple[str, ...] | None
    overrides: Mapping[str, EventRule]
    unsupported: str
    max_events: int
    cache_bytes: int

    @classmethod
    def from_config(cls, block, *, chunk: int) -> ContactConfig | None:
        """The block -> its config, refusing an unknown key or a value out of range.

        Keys, types and ranges are checked even with enabled false, so a bad value shows at the
        run that wrote it rather than at the run that switches it on. What needs the manifest
        (directory names, overrides outside the selection, unsupported directories, a block
        that moves no directory, the mode) is checked only when enabled, by ContactFocus and the
        dataset. strength is required when enabled.
        """
        if block is None:
            return None
        block = _mapping(block, WHERE)
        unknown = sorted(set(block) - set(KEYS))
        if unknown:
            raise ValueError(
                f"{WHERE} holds unknown {unknown}.\n"
                f"Expected only {list(KEYS)}.\n"
                "Fix the key."
            )
        enabled = block.get("enabled")
        if not isinstance(enabled, bool):
            raise ValueError(
                f"{WHERE}.enabled is {enabled!r}.\n"
                "Expected true or false, stated.\n"
                "Write enabled: true to switch contact sampling on; null the block leaves it off."
            )
        if enabled and block.get("strength") is None:
            raise ValueError(
                f"{WHERE} is enabled with no strength.\n"
                "Expected a strength at or above 1: the weight of a window start against 1.\n"
                "State it, e.g. strength: 2.0; there is no default strength."
            )
        stated = dict(block)
        if stated.get("strength") is None:
            stated["strength"] = 1.0                  # only reached with enabled false
        rule = _rule(stated, WHERE, int(chunk), None)
        directories = block.get("directories")
        if directories is not None:
            directories = _names(directories, f"{WHERE}.directories")
        overrides = {}
        for name, value in _mapping(block.get("overrides") or {}, f"{WHERE}.overrides").items():
            where = f"{WHERE}.overrides.{name}"
            one = _mapping(value, where)
            extra = sorted(set(one) - set(OVERRIDE_KEYS))
            if extra or not one:
                raise ValueError(
                    f"{where} holds {sorted(one)}.\n"
                    f"Expected one or more of {list(OVERRIDE_KEYS)}.\n"
                    "Override only what differs for that directory."
                )
            overrides[name] = _rule(one, where, int(chunk), rule)
        unsupported = block.get("unsupported")
        unsupported = "error" if unsupported is None else unsupported
        if unsupported not in UNSUPPORTED:
            raise ValueError(
                f"{WHERE}.unsupported is {unsupported!r}.\n"
                f"Expected one of {UNSUPPORTED}.\n"
                "error refuses a directory the gripper signal cannot speak for; skip draws it "
                "as if contact sampling were off."
            )
        max_events = block.get("max_events")
        max_events = _count(MAX_EVENTS if max_events is None else max_events,
                            f"{WHERE}.max_events", 1)
        cache_bytes = block.get("cache_bytes")
        cache_bytes = _count(CACHE_BYTES if cache_bytes is None else cache_bytes,
                             f"{WHERE}.cache_bytes", MIN_CACHE_BYTES)
        return cls(enabled, rule, directories, overrides, str(unsupported), max_events,
                   cache_bytes)


def _mapping(value, where: str) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError(
            f"{where} is {value!r}.\n"
            "Expected a mapping.\n"
            "State it as a block of keys."
        )
    return {str(key): item for key, item in value.items()}


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(float(value))


def _count(value, where: str, low: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < low:
        raise ValueError(
            f"{where} is {value!r}.\n"
            f"Expected a whole number at or above {low}.\n"
            "Fix the value."
        )
    return int(value)


def _names(value, where: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ValueError(
            f"{where} is {value!r}.\n"
            "Expected a non-empty list of manifest directory names, or null for every one.\n"
            "List the directories by the names the manifest gives them."
        )
    names = tuple(str(name) for name in value)
    if len(set(names)) != len(names):
        raise ValueError(
            f"{where} lists a directory twice: {list(names)}.\n"
            "Expected each name once.\n"
            "Drop the repeat."
        )
    return names


def _rule(block: dict, where: str, chunk: int, base: EventRule | None) -> EventRule:
    """One rule from a block's strength, windows and thresholds; base fills what it omits."""
    strength = base.strength if base is not None else None
    if block.get("strength") is not None:
        strength = block["strength"]
        if not _is_number(strength) or float(strength) < 1.0:
            raise ValueError(
                f"{where}.strength is {strength!r}.\n"
                "Expected a finite number at or above 1: the weight of a window start "
                "against 1 for every other start of the episode.\n"
                "State e.g. 2.0; 1 draws as if contact sampling were off."
            )
    windows = base.windows if base is not None else _windows(None, where, chunk)
    if block.get("windows") is not None:
        windows = _windows(block["windows"], f"{where}.windows", chunk)
    thresholds = {}
    for name in ("close_at", "open_at"):
        value = THRESHOLD if base is None else getattr(base, name)
        if block.get(name) is not None:
            value = block[name]
            if not _is_number(value) or not 0.0 < float(value) < 1.0:
                raise ValueError(
                    f"{where}.{name} is {value!r}.\n"
                    "Expected a canonical gripper value strictly between 0 (open) and 1 "
                    "(closed).\n"
                    "The directory's raw_open and raw_closed carry the polarity; state the "
                    "threshold on the canonical scale."
                )
        thresholds[name] = float(value)
    if thresholds["open_at"] > thresholds["close_at"]:
        raise ValueError(
            f"{where} opens at {thresholds['open_at']} and closes at {thresholds['close_at']}.\n"
            "Expected open_at at or below close_at: closed is entered at close_at and left "
            "below open_at.\n"
            "Swap them, or state one value for both."
        )
    return EventRule(float(strength), windows, thresholds["close_at"], thresholds["open_at"])


def _windows(value, where: str, chunk: int) -> tuple[tuple[str, int, int], ...]:
    """{close: [before, after], open: [...]} -> windows; None is both at [chunk - 1, 0]."""
    if value is None:
        return tuple((event, max(int(chunk) - 1, 0), 0) for event in EVENTS)
    block = _mapping(value, where)
    unknown = sorted(set(block) - set(EVENTS))
    if unknown or not block:
        raise ValueError(
            f"{where} names {sorted(block)}.\n"
            f"Expected one or both of {list(EVENTS)}.\n"
            "Leave an event out to ignore it."
        )
    windows = []
    for event in EVENTS:
        if event not in block:
            continue
        pair = block[event]
        if isinstance(pair, (str, bytes)) or not isinstance(pair, Sequence) or len(pair) != 2:
            raise ValueError(
                f"{where}.{event} is {pair!r}.\n"
                "Expected [before, after]: the rows before and after the event a start may "
                "sit.\n"
                f"State e.g. [{max(int(chunk) - 1, 0)}, 0], the starts whose chunk holds the event."
            )
        windows.append((event, _count(pair[0], f"{where}.{event}[0]", 0),
                        _count(pair[1], f"{where}.{event}[1]", 0)))
    return tuple(windows)


def canonical(raw, raw_open: float, raw_closed: float) -> np.ndarray:
    """Raw gripper commands -> 0 open and 1 closed by the directory's polarity, not clipped."""
    return (np.asarray(raw, dtype=np.float64) - float(raw_open)) / (
        float(raw_closed) - float(raw_open))


def gripper_events(g, valid, *, close_at: float, open_at: float) -> tuple[np.ndarray, np.ndarray]:
    """One arm's canonical commands of one episode -> (close rows, open rows), ascending.

    A row counts where valid is true and g is finite; any other row keeps the state before it
    and is never an event. The state starts at the first counted row's (closed when g >=
    close_at), which is not an event; after it, open becomes closed at g >= close_at and
    closed becomes open at g < open_at. An event row is the first row of the new state.
    """
    g = np.asarray(g, dtype=np.float64)
    counted = np.asarray(valid, dtype=bool) & np.isfinite(g)
    empty = np.zeros(0, dtype=np.int64)
    if not counted.any():
        return empty, empty
    g = np.where(counted, g, 0.0)            # compared below only where counted
    code = np.zeros(g.shape[0], dtype=np.int8)
    code[counted & (g >= close_at)] = 1
    code[counted & (g < open_at)] = -1
    first = int(np.argmax(counted))
    if code[first] == 0:
        code[first] = -1                     # between the thresholds at the first row: open
    code[0] = code[first]
    marks = np.where(code != 0, np.arange(g.shape[0]), 0)
    np.maximum.accumulate(marks, out=marks)
    state = code[marks]
    change = np.flatnonzero(state[1:] != state[:-1]) + 1
    return (change[state[change] == 1].astype(np.int64),
            change[state[change] == -1].astype(np.int64))


def event_windows(events: Mapping[str, Sequence[int]], windows, length: int) -> tuple[int, ...]:
    """Event rows by kind -> the union of their windows inside [0, length), as flat (a, b) pairs.

    Inclusive, ascending and merged (touching windows too), so overlapping windows weigh a
    start once, not once per event.
    """
    spans = []
    for event, before, after in windows:
        for row in events.get(event, ()):
            first, last = max(int(row) - int(before), 0), min(int(row) + int(after), length - 1)
            if first <= last:
                spans.append((first, last))
    spans.sort()
    merged: list[list[int]] = []
    for first, last in spans:
        if merged and first <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], last)
        else:
            merged.append([first, last])
    return tuple(value for pair in merged for value in pair)


def window_rows(intervals: tuple[int, ...]) -> int:
    """How many starts the windows cover."""
    return sum(intervals[i + 1] - intervals[i] + 1 for i in range(0, len(intervals), 2))


def contact_share(intervals: tuple[int, ...], length: int, strength: float) -> float:
    """The share of an episode's draws whose start is in a window: K C / (n + (K - 1) C)."""
    covered = window_rows(intervals)
    return float(strength) * covered / (int(length) + (float(strength) - 1.0) * covered)


def focused_start(intervals: tuple[int, ...], length: int, strength: float, u: float) -> int:
    """A uniform in [0, 1] -> the start the box weight's inverse CDF puts there.

    The weight is strength on the window starts and 1 on the rest of [0, length); the walk is
    over the merged windows, so a draw costs their count, not the episode's length.
    """
    total = int(length) + (float(strength) - 1.0) * window_rows(intervals)
    target = min(max(float(u), 0.0), 1.0) * total
    edge, mass = 0, 0.0
    for i in range(0, len(intervals), 2):
        first, last = intervals[i], intervals[i + 1]
        plain = first - edge
        if target < mass + plain:
            return min(edge + int(target - mass), first - 1)
        mass += plain
        heavy = (last - first + 1) * float(strength)
        if target < mass + heavy:
            return min(first + int((target - mass) / float(strength)), last)
        mass += heavy
        edge = last + 1
    return max(min(edge + int(max(target - mass, 0.0)), int(length) - 1), 0)


def gripper_columns(entry) -> tuple[int, ...]:
    """The flat action column of each arm's gripper the entry's layout names, in arm order."""
    return tuple(int(fields["gripper"][0]) for fields in entry.action.arms if "gripper" in fields)


def unsupported_reason(entry) -> str | None:
    """Why the gripper command of a directory cannot carry contact events, or None."""
    gripper = entry.gripper
    if gripper.method == "masked":
        return "its action gripper is masked (method masked), so its command is not trusted"
    if not gripper_columns(entry):
        return "no arm names an action gripper column (a joints-only or hand-only layout)"
    opened, closed = float(gripper.raw_open), float(gripper.raw_closed)
    if not (math.isfinite(opened) and math.isfinite(closed)) or opened == closed:
        return f"its gripper polarity raw_open {opened}, raw_closed {closed} is not two values"
    return None


@dataclass(frozen=True)
class _Target:
    rule: EventRule
    columns: tuple[int, ...]
    raw_open: float
    raw_closed: float


class ContactFocus:
    """The enabled mode for one dataset copy: the directories it moves and its event store.

    mixture.draw and mixture.draw_rows call it as focus(dataset_index, episode, length, u,
    start) once a window's episode is chosen, and use the start it returns. reader_of is the
    dataset's per-directory reader, the one the sample itself is read through. Every answer
    is a function of the directory, the episode's rows, the config and u, never of the store,
    so a dropped entry or another worker's store changes how long a draw takes and nothing else.
    """

    def __init__(self, config: ContactConfig, entries, reader_of: Callable):
        self.config = config
        self._reader_of = reader_of
        names = [entry.name for entry in entries]
        selected = set(names) if config.directories is None else set(config.directories)
        unknown = sorted((selected | set(config.overrides)) - set(names))
        if unknown:
            raise ValueError(
                f"{WHERE} names directories {unknown}, which the manifest does not hold.\n"
                f"Expected names among the manifest's {len(names)} directories.\n"
                "Use the names as the manifest spells them."
            )
        outside = sorted(set(config.overrides) - selected)
        if outside:
            raise ValueError(
                f"{WHERE}.overrides names {outside}, outside {WHERE}.directories.\n"
                "Expected overrides only for directories contact sampling moves.\n"
                "Add them to directories, or drop their overrides."
            )
        self.targets: dict[int, _Target] = {}
        refused: dict[str, str] = {}
        for index, entry in enumerate(entries):
            if entry.name not in selected:
                continue
            reason = unsupported_reason(entry)
            if reason is not None:
                refused[entry.name] = reason
                continue
            self.targets[index] = _Target(config.overrides.get(entry.name, config.rule),
                                          gripper_columns(entry), float(entry.gripper.raw_open),
                                          float(entry.gripper.raw_closed))
        if refused and config.unsupported == "error":
            raise ValueError(
                f"{WHERE} cannot read contact events in {len(refused)} directories: "
                + "; ".join(f"{name}: {reason}" for name, reason in sorted(refused.items()))
                + ".\nExpected a measured or declared action gripper column in every directory "
                "it moves.\n"
                "Restrict directories to those that have one, or state unsupported: skip to draw "
                "the others as if contact sampling were off."
            )
        self.skipped = dict(sorted(refused.items()))
        if not self.targets:
            raise ValueError(
                f"{WHERE} is enabled and moves no directory: every one it selects is skipped.\n"
                "Expected at least one directory with an action gripper column.\n"
                "Switch it off for this corpus, or select directories that have one."
            )
        self.names = tuple(names)
        self.store = FileCache(config.cache_bytes)
        self.counts = {"focused": 0, "unmoved": 0, "episodes_read": 0, "rows_read": 0,
                       "no_event_episodes": 0, "chatter_episodes": 0}
        self._warned: set[int] = set()

    def __call__(self, dataset_index: int, episode: int, length: int, u: float,
                 start: int) -> int:
        target = self.targets.get(int(dataset_index))
        if target is None:
            return start
        if target.rule.strength == 1.0:                # the off draw: no row is read for it
            self.counts["unmoved"] += 1
            return start
        intervals = self.intervals(int(dataset_index), int(episode), int(length))
        if not intervals:
            self.counts["unmoved"] += 1
            return start
        self.counts["focused"] += 1
        return focused_start(intervals, int(length), target.rule.strength, u)

    def intervals(self, dataset_index: int, episode: int, length: int) -> tuple[int, ...]:
        """One episode's merged window starts, from the store or from its rows."""
        key = f"{dataset_index}/{episode}"
        held = self.store.take(key)
        if held is not None:
            return held
        found = self._read(dataset_index, episode, length)
        self.store.put(key, found, ENTRY_BYTES + INTERVAL_BYTES * (len(found) // 2))
        return found

    def _read(self, dataset_index: int, episode: int, length: int) -> tuple[int, ...]:
        target = self.targets[dataset_index]
        rule = target.rule
        rows = self._reader_of(dataset_index).rows(episode, 0, length)
        action = np.asarray(rows["action"], dtype=np.float64)
        valid = np.asarray(rows["valid"], dtype=bool)
        events: dict[str, list[int]] = {event: [] for event in EVENTS}
        for column in target.columns:
            g = canonical(action[:, column], target.raw_open, target.raw_closed)
            closes, opens = gripper_events(g, valid, close_at=rule.close_at, open_at=rule.open_at)
            events["close"].extend(closes.tolist())
            events["open"].extend(opens.tolist())
        self.counts["episodes_read"] += 1
        self.counts["rows_read"] += int(length)
        found_events = len(events["close"]) + len(events["open"])
        if found_events > self.config.max_events:
            self.counts["chatter_episodes"] += 1
            if dataset_index not in self._warned:
                self._warned.add(dataset_index)
                logger.warning(
                    "contact sampling: directory %s episode %d holds %d gripper events, above "
                    "max_events %d; such episodes keep their uniform start (stats count them)",
                    self.names[dataset_index], episode, found_events, self.config.max_events)
            return ()
        found = event_windows(events, rule.windows, length)
        if not found:
            self.counts["no_event_episodes"] += 1
        return found

    def stats(self) -> dict:
        """This copy's counts: draws moved and not, episodes and rows read, the store's state."""
        return {**self.counts, "store_entries": len(self.store.entries),
                "store_bytes": self.store.held, "store_budget": self.store.budget,
                "store_hits": self.store.hits, "store_misses": self.store.misses,
                "store_evicted": self.store.evicted}

    def describe(self) -> list[str]:
        """The lines the dataset logs at construction."""
        def rule_text(rule: EventRule) -> str:
            windows = ", ".join(f"{event} [-{before}, +{after}]"
                                for event, before, after in rule.windows)
            return (f"strength {rule.strength:g}, windows {windows}, close_at {rule.close_at:g}, "
                    f"open_at {rule.open_at:g}")

        lines = [f"on: {len(self.targets)} of {len(self.names)} directories, start inside the "
                 f"episode only, episode-local box weight; {rule_text(self.config.rule)}; "
                 f"max_events {self.config.max_events}, event store {self.store.budget} bytes "
                 "per process"]
        for name, rule in sorted(self.config.overrides.items()):
            lines.append(f"override {name}: {rule_text(rule)}")
        for name, reason in self.skipped.items():
            lines.append(f"skipped {name} (drawn as if off): {reason}")
        return lines
