"""Which directory a sample comes from, and which window inside it.

Two samplings, chosen by the data group (data.train.sampling). draws (every
group but stage 1's): a directory by weight times rows, an episode by its
trajectory weight (uniform without one), a start uniformly inside it, each draw
independent. rows (stage 1's groups, protocol A; RowPlan): one pass of
epoch_samples draws laid out over the rows, each directory exactly its share
(largest remainder of weight times rows), every row a window start once per
wrap, a fresh keyed permutation a wrap. Either is a pure function of (index,
seed, manifest, epoch_samples), so a resume draws what it would have drawn.

Nothing after the first level can reject a draw: a window past the end of its
episode is padded and marked, never redrawn, so the realised mixture is the
declared one. A start focus (contact sampling, neo_vla.data.contact) may move the
start inside the chosen episode and nothing else; without one every draw is as before.
"""
from __future__ import annotations

import bisect
import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

SAMPLINGS = ("draws", "rows")

# The stage-1 source tempering: each source's share is
# its row count to this power, normalised over the sources (pi0's size
# tempering), and directories inside a source share it in proportion to rows.
SOURCE_TEMPER = 0.43

# The run seed's stride in stream_seed. Larger than any epoch_samples a run
# holds, so one run's index stream cannot reach into another's.
DRAW_SEED_STRIDE = 2_654_435_761

# How many draw streams one run seed has. torch's CPU generator seeds its
# Mersenne Twister with the low 32 bits of manual_seed's argument (measured on
# torch 2.6.0: seeds s and s + STREAM_SPAN give the same numbers, s and
# s + STREAM_SPAN / 2 do not), so two indices STREAM_SPAN apart would share a
# stream. _stream refuses an index outside [0, STREAM_SPAN) rather than alias
# it; stage 1's 928,972,800 draws (protocol A) are 21.6% of it.
STREAM_SPAN = 1 << 32

# torch.Generator seeds are signed; the product above is folded back under this.
_SEED_MASK = (1 << 62) - 1

# One uniform for the directory, one for the episode, one for the start frame.
_UNIFORMS_PER_DRAW = 3

# A start focus: (dataset_index, episode, episode length, a uniform in [0, 1], the start the
# draw would take) -> the start to take (neo_vla.data.contact.ContactFocus). It sees the draw
# after its directory and episode are fixed, so it cannot move mass between either.
StartFocus = Callable[[int, int, int, float, int], int]


@dataclass(frozen=True)
class Draw:
    """One sample's provenance: where it came from and which window it is.

    index is the draw's own number, which every stream this package spends
    folds on, so a later choice about the same sample cannot be seeded from
    anything else. dataset_index indexes the manifest's directory list, so it
    is the key for every per-directory table. start is a frame index inside the
    episode and may sit close enough to the end that most of the chunk is
    padding. pass_index counts the passes the sampler made over epoch_samples
    before handing this window out (0 on a run that never replays): the window
    does not read it, and every later choice about the sample does, through
    the draw's stream number pass_index * epoch_samples + index. wrap is, under
    rows sampling, how many times the directory's rows were used before this
    draw's turn (0 on its first permutation; always 0 under draws sampling).
    """

    index: int
    dataset_index: int
    name: str
    episode: int
    start: int
    pass_index: int = 0
    wrap: int = 0


@dataclass(frozen=True)
class MixturePlan:
    """The first-level table: one cumulative distribution over directories.

    names      the manifest's directory names, in manifest order
    sources    the source of each, as the manifest states it
    weights    the configured weight of each, before rows are counted
    rows       the row count of each, from the manifest
    cumulative the normalised cumulative mass, so a draw is one search

    Sampling mass is weight times rows, which is why a directory left out of
    the weight table cannot be given a default: a large corpus entering at the
    default takes most of the mixture and looks deliberate.
    """

    names: tuple[str, ...]
    sources: tuple[str, ...]
    weights: torch.Tensor
    rows: torch.Tensor
    cumulative: torch.Tensor

    def mass(self) -> dict[str, float]:
        """Each directory's declared share of the mixture, summing to one."""
        share = self.weights.double() * self.rows.double()
        share = share / share.sum()
        return {
            name: float(value)
            for name, value in zip(self.names, share, strict=True)
        }

    def by_source(self) -> dict[str, float]:
        """The same shares totalled by source, which is what a run is set by."""
        totals: dict[str, float] = {}
        for source, share in zip(self.sources, self.mass().values(), strict=True):
            totals[source] = totals.get(source, 0.0) + share
        return totals


@dataclass(frozen=True)
class TrajectoryWeights:
    """Per-episode weights inside one directory, or None for a uniform draw.

    cumulative is normalised inside the directory, so these weights cannot move
    mass between directories however they were fitted. The manifest holds the
    path relative to the corpus root: the previous table keyed its files by the
    absolute path they were built from, so moving the corpus left every lookup
    missing and every directory silently uniform.
    """

    dataset_index: int
    cumulative: torch.Tensor


def build_plan(
    names: tuple[str, ...],
    sources: tuple[str, ...],
    rows: tuple[int, ...],
    weights: tuple[float, ...],
) -> MixturePlan:
    """The manifest's directories and their weights -> the first-level plan.

    The weights are the manifest's own, one per entry, so there is no table to
    miss and no default to fall back on; a manifest entry without a weight is
    refused when it is read. Raises here for a weight that is negative or not
    finite. Zero is allowed and removes the directory from the draw, which is a
    statement a reader can see.
    """
    if not (len(names) == len(sources) == len(rows) == len(weights)):
        raise ValueError(
            f"{len(names)} names, {len(sources)} sources, {len(rows)} row counts "
            f"and {len(weights)} weights.\n"
            "Expected one source, one row count and one weight per directory.\n"
            "Rebuild the manifest; every entry carries all four."
        )
    if not names:
        raise ValueError(
            "the mixture has no directories.\n"
            "Expected at least one.\n"
            "Point data.manifest at a manifest whose datasets list is not empty."
        )
    weight = torch.tensor(weights, dtype=torch.float64)
    row = torch.tensor(rows, dtype=torch.int64)
    bad = ~torch.isfinite(weight) | (weight < 0.0)
    if bool(bad.any()):
        first = int(torch.nonzero(bad)[0])
        raise ValueError(
            f"directory {names[first]!r} has weight {weights[first]!r}.\n"
            "Expected a finite weight at or above 0.\n"
            "Fix the weight in the manifest; 0 removes the directory from the draw."
        )
    if bool((row < 0).any()):
        first = int(torch.nonzero(row < 0)[0])
        raise ValueError(
            f"directory {names[first]!r} reports {rows[first]} rows.\n"
            "Expected a row count at or above 0.\n"
            "Rebuild the manifest from the filter stage's kept.jsonl."
        )
    mass = weight * row.double()
    total = float(mass.sum())
    if total <= 0.0:
        raise ValueError(
            f"the mixture's total weight times rows is {total}.\n"
            "Expected a positive total; nothing can be drawn from this table.\n"
            "Give at least one directory a weight above 0."
        )
    cumulative = torch.cumsum(mass / total, dim=0)
    # The last element is 1 by construction and off it by rounding, which would
    # send a uniform of 0.9999... past the end of the table.
    cumulative[-1] = 1.0
    return MixturePlan(tuple(names), tuple(sources), weight, row, cumulative)


def draw(
    plan: MixturePlan,
    index: int,
    *,
    seed: int,
    episode_starts: list[torch.Tensor],
    trajectory: list[TrajectoryWeights | None],
    focus: StartFocus | None = None,
) -> Draw:
    """Sample index -> the directory, episode and window it stands for.

    A pure function of index and seed, so a run that resumes at a step draws
    the same windows it would have drawn without stopping, and a verify pass
    can reproduce any sample from its index alone. episode_starts holds each
    directory's episode boundaries, which is what turns an episode choice into
    a frame index. focus, when given, takes the start uniform the window spends
    and returns the start in its place; no other uniform is spent.
    """
    if index < 0:
        raise ValueError(
            f"draw index is {index}.\n"
            "Expected an index at or above 0.\n"
            "The sampler numbers draws from 0; pass the index it gave."
        )
    if len(episode_starts) != len(plan.names) or len(trajectory) != len(plan.names):
        raise ValueError(
            f"{len(plan.names)} directories, {len(episode_starts)} episode-start arrays "
            f"and {len(trajectory)} trajectory tables.\n"
            "Expected one of each per directory, in manifest order.\n"
            "Build all three from the same manifest."
        )
    uniform = _uniforms(index, seed)
    dataset_index = _dataset_index(plan, uniform[0])
    starts = episode_starts[dataset_index]
    name = plan.names[dataset_index]
    episodes = starts.numel() - 1
    if episodes < 1:
        raise ValueError(
            f"directory {name!r} has {episodes} episodes.\n"
            "Expected at least one, and an episode_starts array of episodes + 1 entries.\n"
            "Rebuild the manifest's episode_starts for this directory."
        )
    episode = _episode(trajectory[dataset_index], uniform[1], episodes, name)
    length = int(starts[episode + 1]) - int(starts[episode])
    if length < 1:
        raise ValueError(
            f"directory {name!r} episode {episode} holds {length} frames.\n"
            "Expected at least one frame per episode.\n"
            "Drop the empty episode in the filter stage and rebuild episode_starts."
        )
    # Uniform over the whole episode, including the last chunk-worth of frames.
    # Restricting the start to windows that fit would drop short episodes and
    # move their mass to whichever directory has the longest ones.
    start = min(int(float(uniform[2]) * length), length - 1)
    if focus is not None:
        start = int(focus(dataset_index, episode, length, float(uniform[2]), start))
    return Draw(int(index), dataset_index, name, episode, start)


def tempered_shares(rows_by_source: dict[str, float], *,
                    exponent: float = SOURCE_TEMPER) -> dict[str, float]:
    """Rows per source -> each source's share, n^exponent normalised over the sources.

    Recompute from the final row counts of the corpus being built: the shares
    move with every directory a build keeps or drops.
    """
    if not rows_by_source:
        raise ValueError(
            "no sources to temper.\n"
            "Expected at least one source with its row count.\n"
            "Pass the corpus's rows per source."
        )
    for source, rows in rows_by_source.items():
        if not math.isfinite(float(rows)) or float(rows) <= 0.0:
            raise ValueError(
                f"source {source!r} has {rows!r} rows, so no rows to draw from.\n"
                "Expected a positive row count per source.\n"
                "Leave a source with no rows out of the mixture."
            )
    if not math.isfinite(float(exponent)) or not 0.0 <= float(exponent) <= 1.0:
        raise ValueError(
            f"the tempering exponent is {exponent!r}, expected a number in [0, 1].\n"
            "0 draws every source equally, 1 in proportion to rows."
        )
    powered = {source: float(rows) ** float(exponent) for source, rows in rows_by_source.items()}
    total = sum(powered.values())
    return {source: value / total for source, value in powered.items()}


def family_shares(rows_by_family: dict[str, float], *, exponent: float = SOURCE_TEMPER,
                  fixed: dict[str, float] | None = None) -> dict[str, float]:
    """Rows per family -> each family's share, with some families' shares fixed.

    A family named in fixed takes exactly its stated share; the others keep
    their n^exponent ratios (tempered_shares), scaled to what the fixed ones
    leave, 1 - sum(fixed). With no fixed family this is tempered_shares. The
    stage-1 rule: the simulated InternData-A1 family fixed at 0.15, the other
    eight families n^0.43 over 0.85.
    """
    fixed = dict(fixed or {})
    if not fixed:
        return tempered_shares(rows_by_family, exponent=exponent)
    unknown = sorted(set(fixed) - set(rows_by_family))
    if unknown:
        raise ValueError(
            f"fixed share(s) for {unknown}, which hold no rows here.\n"
            f"Expected a fixed family among {sorted(rows_by_family)}.\n"
            "Fix the family name, or build the corpus with that family's directories."
        )
    for family, share in fixed.items():
        if not math.isfinite(float(share)) or not 0.0 < float(share) < 1.0:
            raise ValueError(
                f"family {family!r} is fixed at {share!r}.\n"
                "Expected a share strictly between 0 and 1.\n"
                "State the share the family must take of every draw."
            )
    free = {family: rows for family, rows in rows_by_family.items() if family not in fixed}
    left = 1.0 - sum(float(share) for share in fixed.values())
    if not free or left <= 0.0:
        raise ValueError(
            f"the fixed shares {fixed} leave {left!r} for {len(free)} other families.\n"
            "Expected at least one family outside the fixed ones and a positive share left "
            "for them.\n"
            "Fix fewer families, or lower their shares."
        )
    shares = {family: left * share
              for family, share in tempered_shares(free, exponent=exponent).items()}
    shares.update({family: float(share) for family, share in fixed.items()})
    return shares


def tempered_weights(sources: tuple[str, ...], rows: tuple[int, ...], *,
                     exponent: float = SOURCE_TEMPER) -> tuple[float, ...]:
    """Each directory's manifest weight for the tempered mixture, the largest 1.0.

    A directory's sampling mass is its weight times its rows (MixturePlan), so
    weight_d = share(source) / rows(source) gives each source its tempered
    share and keeps directories of one source in proportion to their rows.
    """
    if len(sources) != len(rows) or not sources:
        raise ValueError(
            f"{len(sources)} sources and {len(rows)} row counts.\n"
            "Expected one source and one row count per directory, at least one.\n"
            "Pass the manifest's directories in order."
        )
    totals: dict[str, float] = {}
    for source, count in zip(sources, rows, strict=True):
        totals[source] = totals.get(source, 0.0) + float(count)
    shares = tempered_shares(totals, exponent=exponent)
    weights = [shares[source] / totals[source] for source in sources]
    largest = max(weights)
    return tuple(weight / largest for weight in weights)


def realised_mass(
    plan: MixturePlan, count: int, *, seed: int
) -> dict[str, float]:
    """Draw count first-level choices and report the share each source took.

    The verify stage compares this with MixturePlan.by_source. They agree
    because nothing rejects a draw; if a rejection is ever added, this is the
    check that fails.
    """
    if count < 1:
        raise ValueError(
            f"realised_mass was asked for {count} draws.\n"
            "Expected at least one.\n"
            "Raise the draw count; the shares it reports are noisy below a few thousand."
        )
    taken: dict[str, float] = {source: 0.0 for source in plan.sources}
    for index in range(count):
        taken[plan.sources[_dataset_index(plan, _uniforms(index, seed)[0])]] += 1.0
    return {source: value / count for source, value in taken.items()}


def stream_seed(index: int, *, seed: int) -> int:
    """A run seed and a draw index -> the seed of that draw's own stream.

    The seed is multiplied before the index is added, so the two cannot trade
    places. Adding them first makes every pair with the same total one stream:
    the draws of seed s from index i would then be the draws of seed s + k from
    index i - k, and two runs differing only in seed would train on the same
    windows in the same order. Every draw in this package folds here.
    """
    return ((int(seed) * DRAW_SEED_STRIDE) + int(index)) & _SEED_MASK


def extra_uniforms(index: int, *, seed: int, count: int) -> torch.Tensor:
    """One draw's uniforms past the three its window spends, in order.

    A choice made about a sample after its window folds here rather than
    opening a stream of its own, so there is one seed per draw and one place
    it is derived. The window's own three come first and are untouched: torch
    fills a longer request with the same numbers in the same order, which
    tests/data/test_mixture.py holds against. The positions are fixed by
    neo_vla.constants: 0 to 2 the reference-camera draw, 3 the history step,
    4 the history dropout, 5 the noise switch, 6 to 109 the arm noise normals
    and 110 to 125 the body's (EXTRA_UNIFORMS 126), so asking for more never
    moves an earlier choice. index is the draw's stream number, which on a
    replayed window is not the window's index (Draw), so a replay redraws
    every one of these choices.
    """
    if count < 1:
        raise ValueError(
            f"extra_uniforms was asked for {count} uniforms.\n"
            "Expected at least one.\n"
            "Ask for the number the caller spends, whichever branch it takes."
        )
    return _stream(index, seed, _UNIFORMS_PER_DRAW + count)[_UNIFORMS_PER_DRAW:]


def _uniforms(index: int, seed: int) -> torch.Tensor:
    """The uniforms one draw index spends, in the order draw spends them."""
    return _stream(index, seed, _UNIFORMS_PER_DRAW)


def _stream(index: int, seed: int, count: int) -> torch.Tensor:
    """The first count uniforms of one draw index's own stream."""
    if not 0 <= int(index) < STREAM_SPAN:
        raise ValueError(
            f"draw stream {int(index)} is outside [0, {STREAM_SPAN}).\n"
            "Expected a stream torch's 32-bit generator seed can tell from every other "
            "of the run.\n"
            "Draw fewer samples per run, or move the streams to a 64-bit generator first."
        )
    generator = torch.Generator()
    generator.manual_seed(stream_seed(index, seed=seed))
    return torch.rand(count, generator=generator, dtype=torch.float64)


def _dataset_index(plan: MixturePlan, uniform: torch.Tensor) -> int:
    """One uniform -> a directory, by the first-level cumulative mass."""
    found = int(torch.searchsorted(plan.cumulative, uniform, right=True))
    return min(found, len(plan.names) - 1)


def _episode(
    weights: TrajectoryWeights | None, uniform: torch.Tensor, episodes: int, name: str
) -> int:
    """One uniform -> an episode inside a directory that is already chosen.

    Normalising against this directory's own total is what keeps the trajectory
    weights out of the first level: whatever they sum to, they redistribute
    inside the directory and the directory's share is unchanged.
    """
    if weights is None:
        return min(int(float(uniform) * episodes), episodes - 1)
    cumulative = weights.cumulative
    if cumulative.numel() != episodes:
        raise ValueError(
            f"directory {name!r} has {episodes} episodes and "
            f"{cumulative.numel()} trajectory weights.\n"
            "Expected one weight per episode.\n"
            "Rebuild the directory's trajectory weights, or drop the path from the manifest."
        )
    total = cumulative[-1]
    found = int(torch.searchsorted(cumulative, uniform * total, right=True))
    return min(found, episodes - 1)


# The stage-1 mixture rule follows.

# How far a family's manifest share (weight x rows) may sit from the rule's before the
# dataset refuses the manifest.
RULE_TOLERANCE = 5e-4


@dataclass(frozen=True)
class MixtureRule:
    """Families of manifest sources, the tempering exponent, and the families with a fixed share.

    A family's share is family_shares(rows per family, exponent, fixed); inside a family the
    directories share it in proportion to their rows, so every directory of a family carries
    one manifest weight. The data group states the rule (config/data/gd7.yaml mixture).
    """

    exponent: float
    families: dict[str, tuple[str, ...]]
    fixed: dict[str, float]

    @classmethod
    def from_config(cls, block) -> MixtureRule:
        """The data group's mixture block: exponent, families {name: [source, ...]}, fixed."""
        block = dict(block or {})
        missing = sorted({"exponent", "families"} - set(block))
        extra = sorted(set(block) - {"exponent", "families", "fixed"})
        if missing or extra:
            raise ValueError(
                f"data.train.mixture is missing {missing} and holds unknown {extra}.\n"
                "Expected exactly exponent, families and (optionally) fixed.\n"
                "State the rule as config/data/gd7.yaml does, or set mixture to null."
            )
        families = {str(name): tuple(str(source) for source in sources)
                    for name, sources in dict(block["families"]).items()}
        owner: dict[str, str] = {}
        for name, sources in families.items():
            if not sources:
                raise ValueError(
                    f"mixture family {name!r} names no source.\n"
                    "Expected at least one manifest source per family.\n"
                    "List the family's sources, or drop the family."
                )
            for source in sources:
                if source in owner:
                    raise ValueError(
                        f"source {source!r} is in two families, {owner[source]!r} and "
                        f"{name!r}.\nExpected each source in one family.\n"
                        "Remove it from one of them."
                    )
                owner[source] = name
        fixed = {str(name): float(share) for name, share in dict(block.get("fixed") or {}).items()}
        unknown = sorted(set(fixed) - set(families))
        if unknown:
            raise ValueError(
                f"mixture.fixed names {unknown}, which are not families.\n"
                f"Expected a fixed family among {sorted(families)}.\n"
                "Fix the name in mixture.fixed."
            )
        return cls(float(block["exponent"]), families, fixed)

    def owner(self) -> dict[str, str]:
        """source -> its family."""
        return {source: name for name, sources in self.families.items() for source in sources}


def family_table(plan: MixturePlan, rule: MixtureRule) -> list[dict]:
    """The manifest's family shares held to the rule; one row per family, largest share first.

    Raises when a manifest source is in no family, a family holds no directory, a family's
    weight x rows share is off the rule by more than RULE_TOLERANCE, or the directories of a
    family carry different weights (not row-proportional inside it).
    """
    owner = rule.owner()
    unowned = sorted({source for source in plan.sources if source not in owner})
    if unowned:
        raise ValueError(
            f"manifest source(s) {unowned} are in no mixture family.\n"
            f"Expected every source in one of {sorted(rule.families)}.\n"
            "Add the source to its family in the data group's mixture, or set mixture to null "
            "for a corpus the rule is not for."
        )
    rows: dict[str, int] = {}
    weights: dict[str, set] = {}
    for source, count, weight in zip(plan.sources, plan.rows.tolist(), plan.weights.tolist(),
                                     strict=True):
        rows[owner[source]] = rows.get(owner[source], 0) + int(count)
        weights.setdefault(owner[source], set()).add(float(weight))
    empty = sorted(set(rule.families) - {name for name, count in rows.items() if count > 0})
    if empty:
        raise ValueError(
            f"mixture families {empty} hold no rows in this manifest.\n"
            "Expected every family of the rule in the corpus.\n"
            "Build the corpus with those families' directories, or drop them from the rule."
        )
    want = family_shares(rows, exponent=rule.exponent, fixed=rule.fixed)
    got: dict[str, float] = {}
    for source, share in plan.by_source().items():
        got[owner[source]] = got.get(owner[source], 0.0) + share
    table = []
    for name in sorted(rows, key=lambda family: -want[family]):
        table.append({"family": name, "rows": rows[name], "rule": want[name], "manifest": got[name],
                      "fixed": name in rule.fixed})
    off = [row for row in table if abs(row["manifest"] - row["rule"]) > RULE_TOLERANCE]
    skewed = sorted(name for name, held in weights.items() if max(held) > min(held) * (1.0 + 1e-6))
    if off or skewed:
        raise ValueError(
            "the manifest's weights do not carry the data group's mixture rule: "
            + "; ".join(f"{row['family']} {row['manifest']:.4f} against {row['rule']:.4f}"
                        for row in off)
            + (f"; directories of {skewed} carry more than one weight" if skewed else "")
            + ".\nExpected each family's weight x rows share within "
            f"{RULE_TOLERANCE} of the rule, row-proportional inside it.\n"
            "Rebuild the manifest weights with tools/build_weights.py --share, one share "
            "per source as the rule gives it (family_shares)."
        )
    return table


# Rows sampling follows: one pass laid out over the rows.

_MASK64 = (1 << 64) - 1
_FEISTEL_ROUNDS = 4
_POSITION_TAG = 0x5052_4F57_5341_4D50     # keys the position permutation apart from the rows'
_FOCUS_TAG = 0x434F_4E54_4143_5446        # keys a focused start's jitter apart from both
_UNIT53 = 1.0 / (1 << 53)


def _mix64(value: int) -> int:
    """splitmix64's finaliser, a bijective 64-bit mix (deterministic on every platform)."""
    value = (int(value) + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return value ^ (value >> 31)


def permute(position: int, size: int, key: int) -> int:
    """A keyed bijection of [0, size): position -> its place in that key's permutation.

    A balanced Feistel network over the smallest even number of bits that covers size,
    cycle-walked back into [0, size). Each call is O(1) in memory, so a directory of 158M rows
    needs no permutation array in any of a node's 64 loader workers. Bijective for every key,
    because each Feistel round is and a cycle walk keeps a bijection's restriction one.
    """
    size, position = int(size), int(position)
    if not 0 <= position < size:
        raise ValueError(
            f"position {position} is outside [0, {size}).\n"
            "Expected a position inside the range the permutation is of.\n"
            "Reduce the position modulo the range first."
        )
    if size == 1:
        return 0
    bits = max((size - 1).bit_length(), 2)
    bits += bits % 2
    half = bits // 2
    mask = (1 << half) - 1
    rounds = [_mix64(int(key) ^ _mix64(round_ + 1)) for round_ in range(_FEISTEL_ROUNDS)]
    value = position
    while True:
        left, right = value >> half, value & mask
        for round_key in rounds:
            left, right = right, left ^ (_mix64(round_key ^ right) & mask)
        value = (left << half) | right
        if value < size:
            return value


def _name_key(name: str) -> int:
    """A directory's own 64-bit key, from its name (stable when other directories come and go)."""
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "big")


def quotas(mass: list[float], total: int) -> list[int]:
    """total draws shared out in proportion to mass by the largest remainder; they sum to total.

    A zero mass gets 0. Ties in the remainder go to the earlier entry, so the split is a pure
    function of the masses and the total.
    """
    whole = float(sum(mass))
    if not mass or whole <= 0.0 or int(total) < 0:
        raise ValueError(
            f"{len(mass)} masses summing to {whole} for {total} draws.\n"
            "Expected at least one positive mass and a total at or above 0.\n"
            "Give at least one directory a weight above 0."
        )
    exact = [float(value) / whole * int(total) for value in mass]
    base = [int(math.floor(value)) for value in exact]
    left = int(total) - sum(base)
    order = sorted((index for index, value in enumerate(mass) if value > 0.0),
                   key=lambda index: (-(exact[index] - base[index]), index))
    if not 0 <= left <= len(order):
        raise ValueError(
            f"{left} draws left over for {len(order)} directories with a mass.\n"
            "Expected the largest-remainder split to leave fewer than one draw per directory.\n"
            "This is a rounding fault; report the masses and the total."
        )
    for index in order[:left]:
        base[index] += 1
    return base


@dataclass(frozen=True)
class RowPlan:
    """rows sampling: one pass of epoch_samples draws laid out over the directories' rows.

    quotas    draws of each directory in one pass (largest remainder of weight x rows)
    offsets   the first position of each directory's block in the pass (a list, for bisect)
    rows      each directory's rows
    keys      each directory's permutation key, from the seed, epoch_samples and its name
    position  the key of the position permutation (index -> place in the pass)
    """

    epoch_samples: int
    quotas: tuple[int, ...]
    offsets: tuple[int, ...]
    rows: tuple[int, ...]
    keys: tuple[int, ...]
    position: int

    def wraps(self, dataset_index: int) -> tuple[int, int]:
        """(whole wraps, rows of the last partial wrap) of one directory in one pass."""
        rows = self.rows[dataset_index]
        return divmod(self.quotas[dataset_index], rows) if rows else (0, 0)


def build_row_plan(plan: MixturePlan, epoch_samples: int, *, seed: int) -> RowPlan:
    """The pass of epoch_samples draws over plan's directories, for rows sampling."""
    if int(epoch_samples) < 1:
        raise ValueError(
            f"epoch_samples is {epoch_samples}.\n"
            "Expected at least one draw in a pass.\n"
            "Set data.train.epoch_samples to the stage's draws."
        )
    mass = (plan.weights.double() * plan.rows.double()).tolist()
    split = quotas(mass, int(epoch_samples))
    offsets, running = [], 0
    for count in split:
        offsets.append(running)
        running += count
    # The pass length keys the permutations too: stage 2 reads stage 1's manifest at the same
    # seed with a shorter pass, and keyed by the seed alone its first wrap would be a prefix of
    # stage 1's, so a family drawn below its rows in stage 1 would see only those rows again.
    seed_key = _mix64(_mix64(_mix64(int(seed)) ^ 0x524F_5753) ^ int(epoch_samples))
    keys = tuple(_mix64(seed_key ^ _name_key(name)) for name in plan.names)
    return RowPlan(int(epoch_samples), tuple(split), tuple(offsets),
                   tuple(int(count) for count in plan.rows.tolist()), keys,
                   _mix64(seed_key ^ _POSITION_TAG))


def draw_rows(rows: RowPlan, plan: MixturePlan, index: int, *,
              episode_starts: list[torch.Tensor], focus: StartFocus | None = None) -> Draw:
    """Window index (in [0, epoch_samples)) -> the directory, episode and start it stands for.

    The index's place in the pass picks the directory block and the place k inside it; the
    directory's wrap is k // rows and its row the wrap's keyed permutation of k % rows, which
    episode_starts turns into an episode and a start frame. A pure function of its arguments.
    focus, when given, is handed u = (start + j) / length, j a jitter in [0, 1) keyed by the
    directory and k: a wrap takes every start of the episode once, so its u are one per
    stratum of [0, 1) and the starts focus returns are a jittered systematic sample.
    """
    if not 0 <= int(index) < rows.epoch_samples:
        raise ValueError(
            f"window index {int(index)} is outside the pass [0, {rows.epoch_samples}).\n"
            "Expected the index modulo epoch_samples.\n"
            "Reduce the index first; the pass number goes to Draw.pass_index."
        )
    place = permute(int(index), rows.epoch_samples, rows.position)
    dataset_index = bisect.bisect_right(rows.offsets, place) - 1
    k = place - rows.offsets[dataset_index]
    count = rows.rows[dataset_index]
    wrap, local = divmod(k, count)
    row = permute(local, count, _mix64(rows.keys[dataset_index] ^ _mix64(wrap)))
    starts = episode_starts[dataset_index]
    episode = int(torch.searchsorted(starts, torch.tensor(row, dtype=starts.dtype),
                                     right=True)) - 1
    start = row - int(starts[episode])
    if focus is not None:
        length = int(starts[episode + 1]) - int(starts[episode])
        jitter = (_mix64(rows.keys[dataset_index] ^ _mix64(k) ^ _FOCUS_TAG) >> 11) * _UNIT53
        start = int(focus(dataset_index, episode, length, (start + jitter) / length, start))
    return Draw(int(index), dataset_index, plan.names[dataset_index], episode,
                start, wrap=int(wrap))


def pass_table(rows: RowPlan, plan: MixturePlan, owner: dict[str, str] | None = None) -> list[dict]:
    """Per family (or per source without an owner map): rows, draws and passes in one pass.

    passes = draws / rows; a directory's whole wraps and partial rows are RowPlan.wraps.
    """
    grouped: dict[str, dict] = {}
    for dataset_index, source in enumerate(plan.sources):
        name = (owner or {}).get(source, source)
        group = grouped.setdefault(name, {"family": name, "rows": 0, "draws": 0, "wraps": set()})
        group["rows"] += rows.rows[dataset_index]
        group["draws"] += rows.quotas[dataset_index]
        if rows.rows[dataset_index]:
            group["wraps"].add(rows.wraps(dataset_index)[0])
    table = []
    for group in sorted(grouped.values(), key=lambda one: -one["draws"]):
        group["passes"] = group["draws"] / group["rows"] if group["rows"] else 0.0
        group["wraps"] = (min(group["wraps"]), max(group["wraps"])) if group["wraps"] else (0, 0)
        table.append(group)
    return table
