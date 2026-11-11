"""The flow decoder: the suffix layout and the flow-matching objective.

Flow matching in the paper's time convention, tau 0 pure noise and tau 1 clean
data: z = tau * a + (1 - tau) * eps, the target is a - eps, the loss is L2, and
tau is drawn with high-noise emphasis. The heads are zero-initialised, so the
predicted velocity is zero at step zero.

This module owns the suffix layout (template v7, SUFFIX_LAYOUT): one block per
arm and the body's block last, each its K action tokens and nothing else; the
block's frame and identity rows are added to every action token's input. Which
rows hold actions, which keys every row may attend, and the rotary positions
the encoder is handed. An arm row is ACTION_DIM wide (action_in/out_proj), the body's BODY_DIM
(body_in/out_proj). The loss per sample is a weighted mean of group means (EE,
joint, hand, body). The optional dct term (freq_loss, off
unless model.action.dct is enabled) reads the arm columns only.
"""
from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import constants
from ..engine.hostcopy import host_of, to_host_device
from ..engine.loop import GROUP_TRIPLES_SUFFIX
from .expert import ActionExpert
from .freq_loss import FreqLossConfig, Resolved, band_loss, dct_matrix, resolve
from .prefix import PrefixContext
from .tokenizers import SuffixInputs, SuffixTokenizers

logger = logging.getLogger(__name__)

# The sinusoidal time features span this range of periods, which is openpi's.
MIN_TIME_PERIOD = 4e-3
MAX_TIME_PERIOD = 4.0

# tau = TAU_SCALE * (1 - X) with X from Beta(TAU_ALPHA, 1).
TAU_SCALE = 0.999
TAU_ALPHA = 1.5

# The suffix layout a checkpoint states under SUFFIX_LAYOUT_KEY in meta.json
# (policy.checkpoint_meta), refused unless it is this one. v2: every block is
# its K action tokens, the block's frame row (<frame_ref>, <robot_mode>) and
# identity row (<arm_i>, <body>) added to every action token's input, and the
# suffix mask is key padding only (90 tokens at K 30). v1 opened each block with
# those two rows as pre tokens behind a block-local mask (96 tokens at K 30);
# its steps state no suffix layout at all, hold the same tensors under the same
# names, and are served and continued by the tree that wrote them.
SUFFIX_LAYOUT_KEY = "suffix_layout"
SUFFIX_LAYOUT = "neo_vla_suffix_v2"
OLD_SUFFIX_LAYOUTS = {"neo_vla_suffix_v1": "3d53a4b"}
UNSTATED_SUFFIX_LAYOUT = "neo_vla_suffix_v1"

# The loss groups, in the order their weights are held: the
# arm columns of each, by named slice, and the body's eight columns.
LOSS_GROUPS = ("ee", "joint", "hand", "body")
ARM_GROUP_COLUMNS = {
    "ee": constants.ACTION_EE,
    "joint": constants.ACTION_JOINTS,
    "hand": constants.ACTION_HAND,
}

# The diagnostics the step log reads (they replace non_gripper_loss and
# gripper_loss): each a pooled mean over the batch's loss triples of its
# columns, with its triple count beside it (engine.loop.GROUP_DIAGNOSTICS names
# them in the log's order). loss_ee is the EE group's motion columns (dpos and
# drot, v6's non-gripper loss) and loss_gripper its gripper column, so the six
# partition every trained column.
ARM_DIAGNOSTICS = {
    "loss_ee": slice(constants.ACTION_DPOS.start, constants.ACTION_DROT.stop),
    "loss_gripper": constants.ACTION_GRIPPER,
    "loss_joint": constants.ACTION_JOINTS,
    "loss_hand": constants.ACTION_HAND,
}
BODY_DIAGNOSTICS = {
    "loss_body_base": constants.BODY_BASE,
    "loss_body_posture": constants.BODY_POSTURE,
}


def time_embedding(tau: torch.Tensor, dim: int) -> torch.Tensor:
    """tau [B] -> [B, dim] sinusoidal features over a fixed period range."""
    if dim % 2:
        raise ValueError(f"time embedding dim is {dim}, expected an even width")
    fraction = torch.linspace(0.0, 1.0, dim // 2, device=tau.device, dtype=torch.float32)
    period = MIN_TIME_PERIOD * (MAX_TIME_PERIOD / MIN_TIME_PERIOD) ** fraction
    angle = tau.float()[:, None] * (2.0 * math.pi / period)[None, :]
    return torch.cat([angle.sin(), angle.cos()], dim=-1)


def sample_tau(batch_size: int, device: torch.device) -> torch.Tensor:
    """tau = 0.999 * (1 - X), X from Beta(1.5, 1), by the inverse distribution.

    Support is (0, 0.999], so the model never trains on a fully clean sample,
    and the mass sits at high noise where the field is hardest.
    """
    uniform = torch.rand(batch_size, device=device)
    return TAU_SCALE * (1.0 - uniform.pow(1.0 / TAU_ALPHA))


def pooled_mean(error: torch.Tensor, weight: torch.Tensor, columns: slice) -> torch.Tensor:
    """Weighted mean of error over the batch's loss triples of one column group.

    Pooled over the batch rather than averaged per sample, so a group most
    samples lack (the hand, the body) is read at its own scale rather than
    diluted by the samples without it. No triple gives zero rather than nan:
    this is a logged diagnostic, and a nan would spoil run averaging.
    """
    mass = weight[..., columns]
    return (error[..., columns] * mass).sum() / mass.sum().clamp_min(1.0)


def check_group_weights(weights) -> tuple[float, ...]:
    """model.action.group_weights -> the four weights in LOSS_GROUPS order, refused if not exact."""
    stated = dict(weights) if weights is not None else {}
    if set(stated) != set(LOSS_GROUPS):
        raise ValueError(
            f"model.action.group_weights names {sorted(stated)}.\n"
            f"Expected exactly {list(LOSS_GROUPS)} (D14 ships "
            "ee 2, joint 1, hand 1, body 1).\n"
            "Set model.action.group_weights."
        )
    values = tuple(float(stated[name]) for name in LOSS_GROUPS)
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError(
            f"model.action.group_weights is {dict(zip(LOSS_GROUPS, values))}.\n"
            "Expected a positive finite weight per group: a zero weight would keep a "
            "group's targets in the batch and train nothing on them.\n"
            "Set model.action.group_weights."
        )
    return values


class FlowDecoder(nn.Module):
    """The suffix, the expert, and the flow loss over a chunk of actions.

    freq_loss adds the band-weighted DCT-II term of freq_loss.py on top of the
    flow objective, over the arm columns. It is off unless a config names it,
    and with it off this class adds no buffer, no key and no draw for it.
    """

    expert: ActionExpert
    tokenizers: SuffixTokenizers

    def __init__(
        self,
        expert: ActionExpert,
        tokenizers: SuffixTokenizers,
        *,
        action_dim: int,
        body_dim: int,
        chunk: int,
        n_slots: int,
        group_weights,
        num_sample_steps: int = 10,
        gripper_dims: tuple[int, ...] = (constants.GRIPPER_COLUMN,),
        init_std: float = 0.02,
        freq_loss: FreqLossConfig | None = None,
    ):
        super().__init__()
        width = expert.width
        self.expert = expert
        self.tokenizers = tokenizers
        self.action_dim = int(action_dim)
        self.body_dim = int(body_dim)
        # The loss groups and the clamp are named slices of the v7 rows, so a
        # decoder over any other width would read the wrong columns (T6).
        if (self.action_dim, self.body_dim) != (constants.ACTION_DIM, constants.BODY_DIM):
            raise ValueError(
                f"the decoder is asked for {self.action_dim} action and {self.body_dim} "
                "body columns.\n"
                f"Expected {constants.ACTION_DIM} ({constants.ACTION_RECORD}) and "
                f"{constants.BODY_DIM} ({constants.BODY_RECORD}): the loss groups and the "
                "decode clamp are named slices of those rows.\n"
                "Set model.action.action_dim and model.action.body_dim."
            )
        gripper = tuple(sorted({int(column) for column in gripper_dims}))
        if gripper != (constants.GRIPPER_COLUMN,):
            raise ValueError(
                f"gripper_dims is {list(gripper)}, expected [{constants.GRIPPER_COLUMN}]: "
                "the gripper column of the v7 row.\nFix model.action.gripper_dims."
            )
        self.chunk = int(chunk)
        self.n_slots = int(n_slots)
        # Slot s reads arm row s and no row is reserved, so a decoder that
        # accepts more slots than the bank has rows renumbers the arms rather
        # than running out of them.
        if int(tokenizers.n_slots) != self.n_slots:
            raise ValueError(
                f"the decoder takes {self.n_slots} slots and its tokenizers hold "
                f"{int(tokenizers.n_slots)} arm rows; build both from "
                "model.action.n_slots"
            )
        self.group_weights = check_group_weights(group_weights)
        # Their tensor per dtype and device, made once (group_weight_table).
        self.group_weight_tables: dict[tuple, torch.Tensor] = {}
        self.num_sample_steps = int(num_sample_steps)
        self.slot_emb = nn.Parameter(torch.randn(self.chunk, width) * init_std)
        self.action_in_proj = nn.Linear(self.action_dim, width)
        self.time_mlp_in = nn.Linear(2 * width, width)
        self.time_mlp_out = nn.Linear(width, width)
        self.action_out_proj = nn.Linear(width, self.action_dim)
        nn.init.zeros_(self.action_out_proj.weight)
        nn.init.zeros_(self.action_out_proj.bias)
        # The body's own projections: its columns are a base motion, a torso
        # and a head, never an arm's joints, so nothing is shared but the time
        # MLP and the step embedding. Zero at the output, so a run with the
        # body block and one without it predict the same arms at step 0.
        self.body_in_proj = nn.Linear(self.body_dim, width)
        self.body_out_proj = nn.Linear(width, self.body_dim)
        nn.init.zeros_(self.body_out_proj.weight)
        nn.init.zeros_(self.body_out_proj.bias)
        # The auxiliary term, resolved once here so a band that would
        # contribute nothing raises at build time rather than after the first
        # step, and so what it reduces to is stated in a log a reader reads
        # anyway. None is the whole of "off": no buffer is registered, nothing
        # is added to the state dict, and loss() never reaches the branch.
        self.freq_loss: Resolved | None = None
        self._dct_matrices: dict[torch.device, torch.Tensor] = {}
        if freq_loss is not None and freq_loss.enabled:
            self.freq_loss = resolve(freq_loss, self.chunk)
            logger.info("%s", self.freq_loss.report)

    def dct(self, device: torch.device) -> torch.Tensor:
        """The [chunk, chunk] orthonormal DCT-II basis, cached per device.

        A plain attribute rather than a buffer on purpose: a buffer would be
        cast to the module's dtype by build_decoder's .to(dtype) and the basis
        has to stay float32, and a buffer would also show up in the module's
        parameter walk on a path that has nothing to learn.
        """
        cached = self._dct_matrices.get(device)
        if cached is None:
            cached = dct_matrix(self.chunk, device)
            self._dct_matrices[device] = cached
        return cached

    @staticmethod
    def block_len(chunk: int) -> int:
        """One block: its K action tokens, so 30 at chunk 30, arm and body alike.

        No block opens with a pre token (SUFFIX_LAYOUT v2): the frame and the
        identity are added to the action tokens' input instead.
        """
        return int(chunk)

    @staticmethod
    def n_blocks(n_slots: int) -> int:
        """Blocks in the suffix: one per arm slot and the body's, last."""
        return int(n_slots) + 1

    def suffix_positions(
        self, k: int, *, n_slots: int, device: torch.device
    ) -> torch.Tensor:
        """Rotary offsets from the suffix anchor: [n_blocks * block_len(K)].

        k is the arms' action-token count, n_slots * K; the body's block adds
        one block of the same length. The result is
        arange(block_len(K)).repeat(n_slots + 1), so with 2 arms and K 30 it is
        90 entries holding the 30 distinct positions 0..29 three times over.
        Every block sees the same positions and the blocks are told apart by
        the identity row (<arm_i> or <body>) added to each of their tokens.

        n_slots is keyword-only with no default, so omitting it raises. It is
        the batch's slot count, actions.shape[1], never a stored attribute.

        One-dimensional, because these are offsets from a per-sample anchor.
        There is no pos_offset: a scalar cannot express a repeated pattern.
        """
        slots, total = int(n_slots), int(k)
        self.check_slot_count(slots)
        if total % slots:
            raise ValueError(
                f"k is {total} action tokens, expected a multiple of "
                f"n_slots={slots}; k is the arms' total count, n_slots * K"
            )
        block = self.block_len(total // slots)
        # Built on the host and copied over unwaited, its host copy kept for
        # check_suffix_positions (engine.hostcopy).
        positions = torch.arange(block, dtype=torch.long).repeat(self.n_blocks(slots))
        return to_host_device(positions, device)

    def action_rows(
        self, chunk: int, *, n_slots: int, device: torch.device
    ) -> torch.Tensor:
        """Suffix row of every arm action token, slot-major: [n_slots * chunk].

        Row of action (s, j) is s * block_len(chunk) + j, ordered (s, j)
        row-major, which is the order the noise is flattened in and the order
        reshape folds back: with no pre tokens the arms' actions are the first
        n_slots * chunk rows, in order. An index vector and not a boolean mask:
        nonzero would force a host synchronisation on every sampling step.
        """
        slots = int(n_slots)
        self.check_slot_count(slots)
        block = self.block_len(chunk)
        start = torch.arange(slots, device=device, dtype=torch.long) * block
        offset = torch.arange(int(chunk), device=device, dtype=torch.long)
        return (start[:, None] + offset[None, :]).reshape(-1)

    def body_action_rows(
        self, chunk: int, *, n_slots: int, device: torch.device
    ) -> torch.Tensor:
        """Suffix row of every body action token: [chunk], in the last block.

        Row of body step j is n_slots * block_len(chunk) + j: the body's block
        follows every arm's.
        """
        slots = int(n_slots)
        self.check_slot_count(slots)
        start = slots * self.block_len(chunk)
        return start + torch.arange(int(chunk), device=device, dtype=torch.long)

    def check_slot_count(self, n_slots: int) -> None:
        """Raise unless the batch's slot count fits the arm rows that exist."""
        if n_slots < 1 or n_slots > self.n_slots:
            raise ValueError(
                f"batch carries {n_slots} slots, expected 1..{self.n_slots}; "
                "raise model.action.n_slots or send fewer arms"
            )

    def check_suffix_positions(
        self, prefix: PrefixContext, *, chunk: int, n_slots: int
    ) -> None:
        """Raise unless the prefix carries this batch's repeated block layout.

        Two checks. The length must be n_blocks * block_len(chunk), and every
        block, the body's included, must equal the first elementwise.

        The second one is the point. The wrong layout, arange(n_suffix), has
        exactly the same length as the right one, and so does a layout whose
        body block alone is continued (arange(30) twice, then arange(30, 60)),
        so every shape check and every broadcast passes and only the values
        differ. One tensor compare per block is what separates them.
        """
        slots, block = int(n_slots), self.block_len(chunk)
        blocks = self.n_blocks(slots)
        expected = blocks * block
        positions = host_of(prefix.suffix_pos)
        if positions.numel() != expected:
            raise ValueError(
                f"prefix.suffix_pos holds {positions.numel()} positions, expected "
                f"{expected} = {blocks} blocks of {block} ({slots} arms and the body); "
                "call suffix_positions with n_slots * K, not K"
            )
        first = positions[:block]
        for index in range(1, blocks):
            if not torch.equal(first, positions[index * block : (index + 1) * block]):
                name = "the body's block" if index == slots else f"block {index}"
                raise ValueError(
                    f"prefix.suffix_pos blocks 0 and {index} differ ({name}), expected "
                    f"arange({block}) repeated {blocks} times; the positions are "
                    "shared across blocks, not continued"
                )

    def key_valid(
        self, action_valid: torch.Tensor, body_action_valid: torch.Tensor
    ) -> torch.Tensor:
        """[B, S, K] and the body's [B, K] step validity -> [B, (S + 1) * block_len(K)] keys.

        Key (s, j) is action_valid[:, s, j], and the body's key j is
        body_action_valid[:, j]: a padded step, an absent arm (every step
        invalid) and an absent body are masked as keys. They carry real
        learned rows (the step embedding, the block's frame and identity), so
        leaving them attendable would let a present block read one that is not
        there, and gradient would flow back into it.
        """
        if action_valid.dim() != 3:
            raise ValueError(
                f"action_valid is {tuple(action_valid.shape)}, expected [B, S, K]"
            )
        batch, slots, chunk = (int(size) for size in action_valid.shape)
        if tuple(body_action_valid.shape) != (batch, chunk):
            raise ValueError(
                f"body_action_valid is {tuple(body_action_valid.shape)}, expected "
                f"{(batch, chunk)}: one flag per body step"
            )
        valid = torch.cat(
            [action_valid.bool(), body_action_valid.bool().to(action_valid.device)[:, None]],
            dim=1,
        )
        return valid.reshape(batch, self.n_blocks(slots) * self.block_len(chunk))

    def attention_bias(
        self,
        prefix: PrefixContext,
        action_valid: torch.Tensor,
        body_action_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Additive [B, 1, n_suffix, Lp + n_suffix] float32 for the suffix rows.

        The rule, stated once, and it is key padding only: every suffix row
        attends every real prefix token and every valid suffix token, the
        body's included, because a two-armed handover is not two independent
        arms and a reach on a moving base is not independent of the base. A
        block's identity needs no mask (SUFFIX_LAYOUT v2): it is added to every
        token of the block. Padded prefix columns, padded steps, absent arms
        and an absent body are masked as key columns only, never as rows, so
        no row is ever fully masked and no softmax row returns nan (the prefix
        always holds a real token): at every benchmark (body ABSENT) an arm row
        sees exactly what a fixed-base training sample showed it. Every row of
        a sample carries the same mask, so the tensor is one [B, 1, 1, L] row
        expanded over the queries, without a copy.
        """
        batch, slots, chunk = (int(size) for size in action_valid.shape)
        device = prefix.pad_mask.device
        total = self.n_blocks(slots) * self.block_len(chunk)
        keys = torch.cat(
            [prefix.pad_mask.bool(), self.key_valid(action_valid, body_action_valid).to(device)],
            dim=-1,
        )
        lowest = torch.finfo(torch.float32).min
        bias = torch.zeros(keys.shape, device=device, dtype=torch.float32).masked_fill(
            ~keys, lowest
        )
        return bias[:, None, None, :].expand(batch, 1, total, keys.shape[-1])

    def action_tokens(
        self,
        projected: torch.Tensor,
        features: torch.Tensor,
        conditioning: torch.Tensor,
        chunk: int,
    ) -> torch.Tensor:
        """A projected noisy row [B, R * K, width], tau's features, [B, R, width] rows -> tokens.

        Token j of block r is slot_emb[j] + time_mlp([projection, features]) +
        conditioning[:, r]: one rule for an arm's rows and the body's, so step
        j of every block carries the same step embedding, and every token of a
        block carries that block's frame and identity (SuffixTokenizers).

        The conditioning is added after the time MLP, beside the step
        embedding, rather than fed into it: the MLP keeps its input width and
        its weights, the rows enter the expert's residual stream as they are
        (as the pre tokens did, now as an offset of each token rather than as
        keys of their own), and a row's effect does not depend on tau or on
        the noisy action it sits beside.
        """
        batch, total, width = (int(size) for size in projected.shape)
        blocks = int(conditioning.shape[1])
        if tuple(conditioning.shape) != (batch, blocks, width) or total != blocks * chunk:
            raise ValueError(
                f"conditioning is {tuple(conditioning.shape)} for {total} tokens of "
                f"chunk {chunk}, expected [{batch}, {total // max(int(chunk), 1)}, {width}]: "
                "one row per block"
            )
        mixed = torch.cat(
            [projected, features[:, None, :].expand(batch, total, width)], dim=-1
        )
        steps = self.slot_emb[None, None, :chunk]
        rows = conditioning[:, :, None, :]
        added = (steps + rows).reshape(batch, total, width)
        return added + self.time_mlp_out(F.silu(self.time_mlp_in(mixed)))

    def build_suffix(
        self,
        inputs: SuffixInputs,
        z: torch.Tensor,
        z_body: torch.Tensor,
        tau: torch.Tensor,
        *,
        n_slots: int,
    ) -> torch.Tensor:
        """Conditioning and noisy actions -> [B, (n_slots + 1) * block_len(K), width].

        z is [B, n_slots * K, action_dim], z_body [B, K, body_dim] and tau
        [B]. Arm action token (s, j) is slot_emb[j] plus the time MLP of
        action_in_proj(z) and the time features, plus FRAME[s] + IDENT[s]: the
        frame_ref row this arm names in this sample and its <arm_s> row. Body
        action token j is slot_emb[j] plus the time MLP of body_in_proj(z_body),
        plus the <robot_mode> and <body> rows. The rows come from the
        tokenizers, one per block (SuffixTokenizers.forward). Blocks are
        concatenated arms first, the body last, so action_rows and
        body_action_rows index them back out.
        """
        slots = int(n_slots)
        self.check_slot_count(slots)
        batch, total, dims = (int(size) for size in z.shape)
        if dims != self.action_dim:
            raise ValueError(
                f"z carries {dims} action columns, expected {self.action_dim}"
            )
        if total % slots:
            raise ValueError(
                f"z carries {total} action tokens, expected a multiple of "
                f"n_slots={slots}"
            )
        chunk = total // slots
        if chunk > self.chunk:
            raise ValueError(
                f"z asks for {chunk} steps per arm, expected at most "
                f"{self.chunk}; raise model.action.chunk"
            )
        if tuple(z_body.shape) != (batch, chunk, self.body_dim):
            raise ValueError(
                f"z_body is {tuple(z_body.shape)}, expected {(batch, chunk, self.body_dim)}: "
                "one body row per step of the arms' chunk"
            )
        dtype = self.slot_emb.dtype
        width = self.slot_emb.shape[-1]
        features = time_embedding(tau, width).to(dtype)
        rows = self.tokenizers(inputs).to(dtype)
        blocks = self.n_blocks(slots)
        if tuple(rows.shape) != (batch, blocks, width):
            raise ValueError(
                f"the blocks' conditioning rows are {tuple(rows.shape)}, expected "
                f"{(batch, blocks, width)}: one frame + identity row per block"
            )
        arms = self.action_tokens(
            self.action_in_proj(z.to(dtype)), features, rows[:, :slots], chunk
        )
        body = self.action_tokens(
            self.body_in_proj(z_body.to(dtype)), features, rows[:, slots:], chunk
        )
        return torch.cat([arms, body], dim=1)

    def velocity(
        self,
        prefix: PrefixContext,
        attention_bias: torch.Tensor,
        inputs: SuffixInputs,
        z: torch.Tensor,
        z_body: torch.Tensor,
        tau: torch.Tensor,
        *,
        n_slots: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """z [B, n_slots * K, action_dim] and z_body [B, K, body_dim] -> their velocities."""
        suffix = self.build_suffix(inputs, z, z_body, tau, n_slots=n_slots)
        hidden = self.expert(suffix, prefix, attention_bias)
        chunk = z.shape[1] // int(n_slots)
        rows = self.action_rows(chunk, n_slots=int(n_slots), device=hidden.device)
        body_rows = self.body_action_rows(chunk, n_slots=int(n_slots), device=hidden.device)
        arms = self.action_out_proj(hidden.index_select(1, rows)).float()
        body = self.body_out_proj(hidden.index_select(1, body_rows)).float()
        return arms, body

    def group_loss(
        self,
        error: torch.Tensor,
        weight: torch.Tensor,
        body_error: torch.Tensor,
        body_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Per-sample loss [B]: sum_g w_g L_g / sum over present g of w_g.

        L_g is the mean squared error over the sample's loss triples (slot,
        step, dim) in group g, and a group is present when it has at least one.
        The weights are model.action.group_weights; with EE 2 and the others
        1, EE holds at least 0.4 of the weight wherever it is present (0.5 or
        more unless all four groups are; review v7spec corrected the spec's
        "at least half"). A sample with no group keeps a zero loss, as v6's
        denominator clamp gave it.
        """
        sums, counts = [], []
        for name in LOSS_GROUPS[:-1]:
            columns = ARM_GROUP_COLUMNS[name]
            mass = weight[..., columns]
            sums.append((error[..., columns] * mass).sum(dim=(1, 2, 3)))
            counts.append(mass.sum(dim=(1, 2, 3)))
        sums.append((body_error * body_weight).sum(dim=(1, 2)))
        counts.append(body_weight.sum(dim=(1, 2)))
        sums_t, counts_t = torch.stack(sums, dim=-1), torch.stack(counts, dim=-1)
        means = sums_t / counts_t.clamp_min(1.0)
        weights = self.group_weight_table(means.dtype, means.device)
        mass = (weights * (counts_t > 0).to(means.dtype)).sum(dim=-1)
        total = (weights * means).sum(dim=-1)
        return torch.where(mass > 0, total / torch.where(mass > 0, mass, torch.ones_like(mass)),
                           torch.zeros_like(total))

    def group_weight_table(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """self.group_weights as a dtype tensor on device, made once per dtype and device."""
        key = (dtype, device)
        if key not in self.group_weight_tables:
            self.group_weight_tables[key] = torch.tensor(
                self.group_weights, device=device, dtype=dtype
            )
        return self.group_weight_tables[key]

    def loss(
        self,
        prefix: PrefixContext,
        inputs: SuffixInputs,
        *,
        actions: torch.Tensor,
        action_valid: torch.Tensor,
        action_loss_dims: torch.Tensor,
        dim_valid: torch.Tensor,
        body_actions: torch.Tensor,
        body_action_valid: torch.Tensor,
        body_loss_dims: torch.Tensor,
        body_dim_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """actions [B, S, K, A] and body_actions [B, K, 8], normalised -> the loss, its diagnostics.

        Each mask has one job. action_valid [B, S, K] and body_action_valid
        [B, K] say a step exists: they decide which suffix tokens are keys, and
        a step the band flagged keeps its key, so no chunk has a hole in the
        middle. dim_valid [B, S, A] and body_dim_valid [B, 8] are the sample's
        columns: the directory's trained columns AND the dimensions read on
        every step the sample keeps, constant over the chunk as at serve. A
        column outside them enters the action token as exactly 0 (z is zeroed
        there before the projection), so no unread value ever reaches the
        input. action_loss_dims [B, S, K, A] and body_loss_dims [B, K, 8] weigh
        the flow loss and the DCT term per (step, dim): action_valid, dim_valid,
        the step's own read, the band per group and finiteness.

        actions must be finite everywhere: the loader writes 0 on every column
        it does not supply and clamps the rest to the band, so a NaN or an inf
        here is a producer defect and raises rather than being cleaned into a
        trained zero.

        Calls check_suffix_positions first, before anything reaches the expert.
        Draws tau, then the arms' noise, then the body's. Returns loss, the six
        group diagnostics (loss_ee, loss_gripper, loss_joint, loss_hand,
        loss_body_base, loss_body_posture) and tau_mean; under the auxiliary
        dct term also flow_mse_loss and dct_loss, and "loss" is then the sum.
        """
        batch, slots, chunk, dims = self.check_action_shapes(
            actions, action_valid, action_loss_dims, dim_valid,
            body_actions, body_action_valid, body_loss_dims, body_dim_valid,
        )
        self.check_suffix_positions(prefix, chunk=chunk, n_slots=slots)
        device = actions.device
        clean = actions.float()
        body_clean = body_actions.float()
        # The targets' host copies (engine.hostcopy): the same values, unwaited.
        for name, values in (("actions", host_of(actions).float()),
                             ("body_actions", host_of(body_actions).float())):
            if not bool(torch.isfinite(values).all()):
                raise ValueError(
                    f"{name} hold {int((~torch.isfinite(values)).sum())} values that are "
                    "not finite.\nExpected finite targets everywhere: the loader writes 0 "
                    "on columns it does not supply and flags a non-finite step's dims "
                    "loss-invalid.\n"
                    "Fix the producer; a NaN here is never cleaned into a trained zero."
                )
        tau = sample_tau(batch, device)
        noise = torch.randn_like(clean)
        body_noise = torch.randn_like(body_clean)
        column_weight = dim_valid.float()[:, :, None, :]
        body_column_weight = body_dim_valid.float()[:, None, :]
        mix = tau.view(batch, 1, 1, 1)
        # The projections are shared across columns, so noise left on an
        # invalid one would move the live ones, and an unread target left on
        # one would state a value nothing measured.
        z = (mix * clean + (1.0 - mix) * noise) * column_weight
        body_mix = tau.view(batch, 1, 1)
        z_body = (body_mix * body_clean + (1.0 - body_mix) * body_noise) * body_column_weight
        target = clean - noise
        body_target = body_clean - body_noise
        bias = self.attention_bias(prefix, action_valid, body_action_valid)
        predicted, body_predicted = self.velocity(
            prefix, bias, inputs, z.reshape(batch, slots * chunk, dims), z_body, tau,
            n_slots=slots,
        )
        predicted = predicted.reshape(batch, slots, chunk, dims)
        error = (predicted - target) ** 2
        body_error = (body_predicted - body_target) ** 2
        weight = action_loss_dims.float()
        body_weight = body_loss_dims.float()
        flow = self.group_loss(error, weight, body_error, body_weight).mean()
        result = {"loss": flow}
        with torch.no_grad():
            # The count beside each mean is what tells a group this batch does
            # not hold (0.0 over 0 triples) from one it holds at a loss of 0.
            for name, columns in ARM_DIAGNOSTICS.items():
                result[name] = pooled_mean(error, weight, columns).detach()
                result[name + GROUP_TRIPLES_SUFFIX] = weight[..., columns].sum()
            for name, columns in BODY_DIAGNOSTICS.items():
                result[name] = pooled_mean(body_error, body_weight, columns).detach()
                result[name + GROUP_TRIPLES_SUFFIX] = (
                    body_weight[..., columns].sum())
        result["tau_mean"] = tau.mean().detach()
        if self.freq_loss is None:
            return result
        # x0 from the velocity, in this module's time convention rather than
        # the reference's: z = tau * a + (1 - tau) * eps and v = a - eps give
        # a = z + (1 - tau) * v identically. The reference recovers the same
        # quantity from its own convention at VLANeXt.py:1628-1629. The
        # gradient through v therefore carries a factor (1 - tau), so this
        # term weights the high-noise end, which is where sample_tau already
        # puts its mass.
        if chunk != self.freq_loss.chunk:
            raise ValueError(
                f"the batch carries a chunk of {chunk} and the dct band was "
                f"resolved for {self.freq_loss.chunk}.\n"
                "The band edge is a count of coefficients, so it means a "
                "different frequency under a different chunk; build the "
                "decoder for the chunk the corpus emits."
            )
        predicted_start = z + (1.0 - tau).view(batch, 1, 1, 1) * predicted
        # weight is action_loss_dims as 0/1, so a padded step, a flagged
        # (step, dim) and an invalid column are exactly zero in the residual
        # and contribute to no coefficient. Masking before the transform is
        # what makes a chunk mean over a widened action vector mean anything.
        residual = (predicted_start - clean) * weight
        triples = weight.sum(dim=(1, 2, 3)).clamp_min(1.0)
        aux = band_loss(
            residual,
            self.dct(device),
            self.freq_loss.band_weights(device),
            triples,
        )
        result["loss"] = flow + self.freq_loss.weight * aux
        result["flow_mse_loss"] = flow.detach()
        result["dct_loss"] = aux.detach()
        return result

    def check_action_shapes(
        self,
        actions: torch.Tensor,
        action_valid: torch.Tensor,
        action_loss_dims: torch.Tensor,
        dim_valid: torch.Tensor,
        body_actions: torch.Tensor,
        body_action_valid: torch.Tensor,
        body_loss_dims: torch.Tensor,
        body_dim_valid: torch.Tensor,
    ) -> tuple[int, int, int, int]:
        """Raise unless the eight tensors agree; return (B, S, K, A).

        A loss mask may only narrow its keys and its columns: a (step, dim)
        the loss reads and the keys or the columns do not would be a target
        for a token nothing can attend, or for an input that was zeroed.
        """
        if actions.dim() != 4:
            raise ValueError(
                f"actions is {tuple(actions.shape)}, expected [B, S, K, A]"
            )
        batch, slots, chunk, dims = (int(size) for size in actions.shape)
        self.check_slot_count(slots)
        if dims != self.action_dim:
            raise ValueError(
                f"actions carries {dims} columns, expected "
                f"action_dim={self.action_dim}"
            )
        if tuple(body_actions.shape) != (batch, chunk, self.body_dim):
            raise ValueError(
                f"body_actions is {tuple(body_actions.shape)}, expected "
                f"{(batch, chunk, self.body_dim)}"
            )
        for name, tensor, shape in (
            ("action_valid", action_valid, (batch, slots, chunk)),
            ("action_loss_dims", action_loss_dims, (batch, slots, chunk, dims)),
            ("dim_valid", dim_valid, (batch, slots, dims)),
            ("body_action_valid", body_action_valid, (batch, chunk)),
            ("body_loss_dims", body_loss_dims, (batch, chunk, self.body_dim)),
            ("body_dim_valid", body_dim_valid, (batch, self.body_dim)),
        ):
            if tuple(tensor.shape) != shape:
                raise ValueError(
                    f"{name} is {tuple(tensor.shape)}, expected {shape}"
                )
            if tensor.dtype is not torch.bool:
                raise ValueError(
                    f"{name} is {tensor.dtype}, expected torch.bool; a "
                    "fractional mask is a loss weight, not a validity mask"
                )
        # The masks' host copies (engine.hostcopy): the same values, unwaited.
        action_valid, dim_valid = host_of(action_valid), host_of(dim_valid)
        body_action_valid, body_dim_valid = host_of(body_action_valid), host_of(body_dim_valid)
        allowed = action_valid[..., None] & dim_valid[:, :, None, :]
        if bool((host_of(action_loss_dims) & ~allowed).any()):
            raise ValueError(
                "action_loss_dims is True on a (step, dim) that action_valid or dim_valid "
                "calls absent.\n"
                "Expected action_loss_dims = action_valid & dim_valid & read & in_band & "
                "finite.\nFix pack_slots."
            )
        body_allowed = body_action_valid[..., None] & body_dim_valid[:, None, :]
        if bool((host_of(body_loss_dims) & ~body_allowed).any()):
            raise ValueError(
                "body_loss_dims is True on a (step, dim) that body_action_valid or "
                "body_dim_valid calls absent.\n"
                "Expected body_loss_dims = body_action_valid & body_dim_valid & read & "
                "in_band & finite.\nFix the loader's body targets."
            )
        return batch, slots, chunk, dims

    @torch.no_grad()
    def sample_actions(
        self,
        prefix: PrefixContext,
        inputs: SuffixInputs,
        *,
        n_slots: int,
        slot_valid: torch.Tensor,
        dim_valid: torch.Tensor,
        body_dim_valid: torch.Tensor,
        range_bound: float,
        noise: torch.Tensor | None = None,
        body_noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Euler integration from noise -> ([B, n_slots, K, A], [B, K, 8]), clamped per column.

        The clamp is the target space's own: a range column was normalised by
        the band to at most 1 + 2 alpha, which is range_bound and has no default
        (policy.embodiments.band_alpha states alpha), and a gripper column is
        2 g_c - 1, so it is clamped to [-1, 1]. Every body column is a range
        column. One bound for all would clip every kept step beyond q01/q99 or
        let the gripper leave its range.

        The prefix keys and values are computed once and reused across the
        steps; only the suffix is rebuilt. Calls check_suffix_positions before
        the first step. Masks the same columns training masks (dim_valid,
        body_dim_valid): the projections are shared across columns, so noise on
        an invalid one would move the live ones; and the returned chunks are
        exactly 0 on every column those mark invalid. A body step is a key
        when the body decodes any column, which is never for an ABSENT body.

        noise and body_noise, when given, ARE the draws: [B, n_slots, K, A] and
        [B, K, body_dim] on any device, moved here; both or neither. They exist
        because the default draw is one tensor for the whole batch, so which
        numbers a request receives depend on how many requests shared its
        window. Without them the arms' draw is taken first and the body's
        second, always, the body's included where the body is ABSENT: the arms'
        noise for a seed never depends on whether a body exists.
        """
        slots, chunk = int(n_slots), self.chunk
        self.check_slot_count(slots)
        bound = float(range_bound)
        if not bound >= 1.0:
            raise ValueError(
                f"range_bound is {range_bound}, expected 1 + 2 alpha, at least 1.\n"
                "Pass 1 + 2 * policy.embodiments.band_alpha."
            )
        batch = int(slot_valid.shape[0])
        if tuple(slot_valid.shape) != (batch, slots):
            raise ValueError(
                f"slot_valid is {tuple(slot_valid.shape)}, expected "
                f"{(batch, slots)}"
            )
        if tuple(dim_valid.shape) != (batch, slots, self.action_dim):
            raise ValueError(
                f"dim_valid is {tuple(dim_valid.shape)}, expected "
                f"{(batch, slots, self.action_dim)}"
            )
        if tuple(body_dim_valid.shape) != (batch, self.body_dim):
            raise ValueError(
                f"body_dim_valid is {tuple(body_dim_valid.shape)}, expected "
                f"{(batch, self.body_dim)}"
            )
        if (noise is None) != (body_noise is None):
            raise ValueError(
                "sample_actions was handed one draw of two.\n"
                "Expected noise and body_noise both, or neither: a caller that owns the "
                "arms' draw owns the body's, so a request reproduces whole.\n"
                "Pass both from the request's generator, arms first."
            )
        self.check_suffix_positions(prefix, chunk=chunk, n_slots=slots)
        device = prefix.pad_mask.device
        action_valid = slot_valid.bool()[:, :, None].expand(batch, slots, chunk)
        body_valid = body_dim_valid.bool().any(dim=-1)[:, None].expand(batch, chunk)
        bias = self.attention_bias(prefix, action_valid, body_valid.to(device))
        column_weight = dim_valid.float()[:, :, None, :].to(device)
        body_column_weight = body_dim_valid.float()[:, None, :].to(device)
        if noise is None:
            z = torch.randn(batch, slots, chunk, self.action_dim, device=device)
            z_body = torch.randn(batch, chunk, self.body_dim, device=device)
        else:
            for name, draw, want in (
                ("noise", noise, (batch, slots, chunk, self.action_dim)),
                ("body_noise", body_noise, (batch, chunk, self.body_dim)),
            ):
                if tuple(draw.shape) != want:
                    raise ValueError(
                        f"{name} is {tuple(draw.shape)}, expected {want}.\n"
                        "Expected one draw per sample, block row and dimension; a draw "
                        "shaped for another batch would silently pair the wrong rows."
                    )
            z = noise.to(device=device, dtype=torch.float32)
            z_body = body_noise.to(device=device, dtype=torch.float32)
        step = 1.0 / self.num_sample_steps
        for index in range(self.num_sample_steps):
            tau = torch.full((batch,), index * step, device=device)
            z = z * column_weight
            z_body = z_body * body_column_weight
            velocity, body_velocity = self.velocity(
                prefix,
                bias,
                inputs,
                z.reshape(batch, slots * chunk, self.action_dim),
                z_body,
                tau,
                n_slots=slots,
            )
            z = z + step * velocity.reshape(batch, slots, chunk, self.action_dim)
            z_body = z_body + step * body_velocity
        # Once more after the last step: the mask above zeroes an invalid column
        # before each velocity, so the last step's own velocity would otherwise
        # reach the reply on a column the masks say carry nothing.
        z = z * column_weight
        z_body = z_body * body_column_weight
        bounds = self.column_bounds(bound, device)
        return z.clamp(-bounds, bounds), z_body.clamp(-bound, bound)

    def column_bounds(self, range_bound: float, device: torch.device) -> torch.Tensor:
        """[A] per-column clamp: range_bound on the range columns, 1 on the gripper."""
        bounds = torch.full((self.action_dim,), float(range_bound), device=device)
        bounds[constants.ACTION_GRIPPER] = 1.0
        return bounds
