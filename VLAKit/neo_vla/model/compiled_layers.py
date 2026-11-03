"""The backbone's and the expert's layers through torch.compile (trainer.compile_layers).

The forty backbone layers and the expert's layers carry most of a stage-1
step's GPU time, and in eager mode each norm, rotation and gate is its own
kernel. Compiling each layer's forward with inductor at static shapes, and
leaving everything around the layers eager, measured 4,048 -> 2,402 ms a step
on one 8-GPU node at the stage-1 recipe, with the loss difference to eager
inside eager's own run-to-run spread.

One compiled function per class is bound to each layer instance, never
written over the class, so any other instance in the process -- a serving
policy, a test -- stays eager. The parameters, their names and the state dict
are untouched; only the kernels that run a layer change.

compile_robot (trainer.compile_robot, stage 2) compiles the robot path only,
under co-training and recomputation: see the section at its foot.
"""
from __future__ import annotations

import types

import torch
import torch._dynamo
import torch.utils.checkpoint
from torch import nn

from .backbone.neo_llm import DecoderLayer
from .expert import ExpertLayer

COMPILED_CLASSES = (DecoderLayer, ExpertLayer)
# Every layer of a class shares one compiled function and dynamo keys its cache on
# that function's code, so a guard that ever splits per layer (forty of them) would
# overrun the default of 8 entries and drop the rest back to eager without a word.
CACHE_SIZE_LIMIT = 128


def compile_layers(module: nn.Module, backend: str = "inductor") -> dict[str, int]:
    """Route every DecoderLayer and ExpertLayer under module through torch.compile.

    Returns how many layers of each class now run compiled. Raises when the
    module holds no layer of a class or a layer already has its own forward:
    either way the run would not be the configuration that was measured.
    backend is for the cpu tests; training uses inductor.
    """
    found: dict[str, list[tuple[str, nn.Module]]] = {kind.__name__: [] for kind in COMPILED_CLASSES}
    for name, layer in module.named_modules():
        if type(layer) in COMPILED_CLASSES:
            found[type(layer).__name__].append((name, layer))
    missing = [kind for kind, layers in found.items() if not layers]
    if missing:
        raise ValueError(
            f"compile_layers: {type(module).__name__} holds no {' and no '.join(missing)}.\n"
            f"Expected both {' and '.join(found)}: the speed runs compiled both stacks.\n"
            "Set trainer.compile_layers false for this model."
        )
    bound = [name for layers in found.values() for name, layer in layers if "forward" in vars(layer)]
    if bound:
        raise ValueError(
            f"compile_layers: {len(bound)} layer(s) already carry their own forward, "
            f"the first {bound[0]!r}.\n"
            "Expected each layer to run its class's forward before it is compiled.\n"
            "Compile a module once, right after it is built."
        )
    torch._dynamo.config.cache_size_limit = max(
        torch._dynamo.config.cache_size_limit, CACHE_SIZE_LIMIT
    )
    for kind in COMPILED_CLASSES:
        compiled = torch.compile(kind.forward, dynamic=False, backend=backend)
        for _, layer in found[kind.__name__]:
            layer.forward = types.MethodType(compiled, layer)
    return {kind: len(layers) for kind, layers in found.items()}


# The robot path compiled under co-training (trainer.compile_robot), stage 2.
#
# compile_layers above binds a compiled forward to every layer, so every caller
# of a layer runs it compiled; under co-training that is the chat pack too,
# whose flex attention has its own compiled kernel (neo_llm.CHAT_KERNEL), and
# under recomputation it fails outright: the eager checkpoint around a compiled
# layer recomputes it from the autograd engine's thread and saved 44 tensors
# against the forward's 53 (CheckpointError).
# compile_robot compiles the robot path only: the backbone's layers as the
# robot prefix calls them (encoder.encode: an additive bias, no final hidden
# state) through one compiled function that holds its own recomputation, and
# the expert's layers, which only the robot rows reach. The chat pack keeps its
# eager layers, its eager recomputation and its compiled flex kernel.

# How many graphs one prefix shape may build: the backbone layer with its
# recomputation, the backbone layer without (trainer.robot_recompute_layers),
# and the expert layer. Another is a guard that failed (a layer per graph, say)
# and is refused, as the chat kernel refuses a second graph (CHAT_KERNEL_FRAMES).
ROBOT_GRAPHS_PER_SHAPE = 3


# The robot stack's numerics (trainer.robot_eager_numerics, off by default; the
# stage-2 launch states it). With it on, measured on 8 ranks at the stage-2
# recipe: max allocated 71.82 GiB, step period 6.08 s against 6.25.
#
# ROBOT_INDUCTOR_PATCHES, for the robot stack's compile only: emulate_precision_casts,
# so a pointwise op that eager runs with a bf16 output is rounded to bf16 inside the
# fused kernel as eager rounds it (the feed-forward's silu output before the product
# with the up projection, and the backward's bf16 pointwise ops; inductor's default
# keeps those intermediates in float32 until a store). ROBOT_SAVED_OPS: each recomputed
# layer keeps its attention output and log-sum-exp for the backward instead of running
# the attention a second time; the partitioner then keeps the attention's query too,
# which MergedAttention.cast_query hands over in bf16 (the float32 query cost 0.36 GiB
# more over 12 layers).
#
# Why: some micro-batches are ill-conditioned in bf16 (one measured case: the backward
# of the backbone's first ~8 layers amplified a perturbation of the residual cotangent
# at a handful of tokens about 270-fold), so which rounding a variant draws decides
# that micro-batch's gradient; the layers themselves agree with float32 within 4% each,
# recomputed or not. Over 256 real micro-batches at three runs' weights (relL2 of the
# micro-batch gradient against float32): inductor defaults mean 0.172 / 0.118 / 0.109,
# 7 above 0.5, max 8.45; these settings 0.101 / 0.099 / 0.098, 3 above 0.5, max 1.46;
# eager 0.095 / 0.152 / 0.102, 2 above 0.5, max 4.15.
ROBOT_INDUCTOR_PATCHES = {"emulate_precision_casts": True}
ROBOT_SAVED_OPS = ("sdpa",)

# Ops a recomputed robot layer saves instead of recomputing, by name -> the aten
# overloads; read by RobotStack(save=...).
SAVE_OPS = {
    "sdpa": ("_scaled_dot_product_efficient_attention", "_scaled_dot_product_flash_attention",
             "_scaled_dot_product_cudnn_attention", "_scaled_dot_product_flash_attention_for_cpu"),
    "mm": ("mm", "addmm", "bmm"),
}


def save_policy(names):
    """A selective-checkpoint context_fn that saves the named ops' outputs, recomputes the rest."""
    import functools

    from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

    ops = set()
    rest = CheckpointPolicy.PREFER_RECOMPUTE
    for name in names:
        if name == "strict":
            rest = CheckpointPolicy.MUST_RECOMPUTE   # diagnostic: every other op must recompute
            continue
        for op in SAVE_OPS[name]:
            packet = getattr(torch.ops.aten, op, None)
            if packet is not None:
                ops.add(packet.default)

    def policy(ctx, op, *args, **kwargs):
        return CheckpointPolicy.MUST_SAVE if op in ops else rest

    return functools.partial(create_selective_checkpoint_contexts, policy)


def robot_layer(layer: DecoderLayer, x, attention_bias, rope, recompute: bool):
    """One backbone layer on the robot prefix, recomputed in the backward when asked.

    The recomputation is inside the compiled function, so torch.compile traces
    the checkpoint as its activation-checkpoint region and the backward graph
    recomputes the layer from the saved input itself (no eager checkpoint, no
    second call of a compiled function from the autograd thread). recompute is
    a Python bool: each value is its own graph, at most two.
    """
    if recompute:
        return torch.utils.checkpoint.checkpoint(
            layer, x, attention_bias, rope, use_reentrant=False)
    return layer(x, attention_bias, rope)


def resolve_backend(backend: str):
    """A backend name -> its compiler function (inductor in training, aot_eager on the cpu)."""
    from torch._dynamo.backends.registry import lookup_backend

    return lookup_backend(backend)


class RobotStack:
    """The robot path's compiled layers and the count of the graphs they built.

    Called by NeoBackbone.forward for every backbone layer of a robot prefix,
    and bound to every ExpertLayer of the policy. frames counts what the
    compiler is handed through a counting backend; check refuses more than
    ROBOT_GRAPHS_PER_SHAPE per prefix shape seen.
    """

    def __init__(self, backend: str = "inductor", config_patches: dict | None = None,
                 save: tuple[str, ...] = ()):
        self.backend = backend
        self.compiler = resolve_backend(backend)
        self.config_patches = dict(config_patches or {})
        self.save = tuple(save)
        self.frames = 0
        self.shapes: set[tuple[int, ...]] = set()
        self.layer = torch.compile(robot_layer if not save else self.saving_layer(save),
                                   dynamic=False, fullgraph=True, backend=self.count)
        self.expert = torch.compile(ExpertLayer.forward, dynamic=False, fullgraph=True,
                                    backend=self.count)

    @staticmethod
    def saving_layer(save):
        context_fn = save_policy(save)

        def robot_layer_saving(layer, x, attention_bias, rope, recompute: bool):
            if recompute:
                return torch.utils.checkpoint.checkpoint(
                    layer, x, attention_bias, rope, use_reentrant=False, context_fn=context_fn)
            return layer(x, attention_bias, rope)

        return robot_layer_saving

    def count(self, graph, example_inputs):
        """The compiler backend: the named one, counting every graph it is handed."""
        self.frames += 1
        if self.config_patches:
            if self.backend != "inductor":
                raise ValueError(f"config patches {self.config_patches} need inductor, "
                                 f"not {self.backend!r}")
            from torch._inductor.compile_fx import compile_fx

            # compile_fx patches the backward's compile too (it runs out of this scope).
            return compile_fx(graph, example_inputs, config_patches=dict(self.config_patches))
        return self.compiler(graph, example_inputs)

    def __call__(self, layer: DecoderLayer, x, attention_bias, rope, recompute: bool):
        self.shapes.add(tuple(x.shape))
        out = self.layer(layer, x, attention_bias, rope, bool(recompute))
        self.check()
        return out

    def check(self) -> None:
        """Raise once the compiler has built more graphs than the shapes seen allow."""
        allowed = ROBOT_GRAPHS_PER_SHAPE * max(1, len(self.shapes))
        if self.frames > allowed:
            raise RuntimeError(
                f"the compiled robot path has built {self.frames} graphs over "
                f"{len(self.shapes)} prefix shape(s).\n"
                f"Expected at most {ROBOT_GRAPHS_PER_SHAPE} a shape: the backbone layer with and "
                "without recomputation and the expert layer. Another is a guard that failed "
                "(one graph per layer, say), and each costs a compile on every rank.\n"
                "Find the failing guard with TORCH_LOGS=recompiles, or set "
                "trainer.compile_robot false."
            )


def compile_robot(module: nn.Module, backend: str = "inductor",
                  config_patches: dict | None = None, save: tuple[str, ...] | None = None,
                  eager_numerics: bool = False) -> RobotStack:
    """Route module's robot path through torch.compile; returns the RobotStack.

    module holds one NeoBackbone and an expert of ExpertLayers (the policy).
    The backbone gets the stack as robot_stack, which its forward calls for a
    robot prefix only; every ExpertLayer gets a forward bound to the stack's
    compiled expert. Raises when a layer already carries its own forward
    (compile_layers ran) or a stack is missing.

    eager_numerics (trainer.robot_eager_numerics, default off) compiles with
    ROBOT_INDUCTOR_PATCHES (inductor only; the cpu tests' aot_eager runs eager
    kernels) and saves ROBOT_SAVED_OPS in the recomputed layers; off is inductor's
    defaults with every op recomputed. config_patches and save, when given,
    replace both (diagnostics).
    """
    if config_patches is None:
        config_patches = ROBOT_INDUCTOR_PATCHES if eager_numerics and backend == "inductor" else {}
    if save is None:
        save = ROBOT_SAVED_OPS if eager_numerics else ()
    from .backbone.neo_llm import NeoBackbone

    backbones = [layer for layer in module.modules() if isinstance(layer, NeoBackbone)]
    experts = [layer for layer in module.modules() if type(layer) is ExpertLayer]
    if len(backbones) != 1 or not experts:
        raise ValueError(
            f"compile_robot: {type(module).__name__} holds {len(backbones)} backbone(s) and "
            f"{len(experts)} expert layer(s).\n"
            "Expected one NeoBackbone and the action expert's layers: the robot path.\n"
            "Set trainer.compile_robot false for this model."
        )
    bound = [name for name, layer in module.named_modules()
             if type(layer) in COMPILED_CLASSES and "forward" in vars(layer)]
    if bound or backbones[0].robot_stack is not None:
        raise ValueError(
            f"compile_robot: {len(bound)} layer(s) already carry their own forward"
            f"{', the first ' + repr(bound[0]) if bound else ''}, or the backbone has a robot "
            "stack already.\n"
            "Expected each layer to run its class's forward: trainer.compile_layers and "
            "trainer.compile_robot are two ways to compile the same layers.\n"
            "Set one of them, once, right after the model is built."
        )
    torch._dynamo.config.cache_size_limit = max(
        torch._dynamo.config.cache_size_limit, CACHE_SIZE_LIMIT
    )
    # The activation-checkpoint region is a higher-order op, which dynamo's DDP
    # graph splitter (optimize_ddp, on by default for a frame compiled inside a
    # DDP forward) refuses: "DDPOptimizer backend: Found a higher order op in the
    # graph" (torch/_dynamo/backends/distributed.py). Off, each compiled layer is
    # one graph whose gradients are ready together at the end of its backward:
    # a layer is 50M parameters, 8 of DDP's 25 MiB buckets, so the all-reduce
    # still overlaps the backward a layer at a time. The chat kernel, the other
    # graph compiled inside the forward, holds no parameter, so nothing splits it.
    torch._dynamo.config.optimize_ddp = False
    stack = RobotStack(backend, config_patches, tuple(save))
    # A recomputed layer that keeps its attention also keeps the attention's query; cast
    # before the join it is kept in bf16 (30 MB a layer at 8 x 908) rather than float32.
    from .backbone.neo_llm import MergedAttention

    for attention in backbones[0].modules():
        if isinstance(attention, MergedAttention):
            attention.cast_query = "sdpa" in stack.save
    backbones[0].robot_stack = stack

    def expert_forward(layer, *args):
        out = stack.expert(layer, *args)
        stack.check()
        return out

    for layer in experts:
        layer.forward = types.MethodType(expert_forward, layer)
    return stack
