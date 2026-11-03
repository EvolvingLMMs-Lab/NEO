"""The NEO vision tower: patch embedding, two-dimensional rope, and a merge.

The checkpoint's vision model has no transformer. A 16x16 patch convolution
lifts the image to 1024 channels, a gelu follows, a two-dimensional rope is
applied with the first half of the channels rotated by column and the second
half by row, and a 2x2 convolution merges four patches into one backbone
token. So one token covers 32x32 pixels.

Images keep their native aspect. Every view arrives on the same canvas, sized
by model.max_grid, with the real image in the top left and the rest zero;
image_shapes says how much is real and the encoder masks the rest. The patch
convolution has kernel equal to stride, so padding cannot leak into a real
patch and a view's tokens are what they would be on their own.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..deferred import require

# NEO's own normalisation, VLMTrainKit neo/data/constants.py IMAGENET_MEAN and
# IMAGENET_STD, applied after ToTensor by its training and eval transforms.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# What the tower does with the [-1, 1] pixels every producer hands it: NEO's
# own ImageNet mean and std, the only pixel convention this tree has. The
# legacy convention handed [-1, 1] to the patch embedding unchanged; it was
# spelled "none" here and is refused by name.
INPUT_NORMS = ("imagenet",)
RETIRED_INPUT_NORMS = {"none": "legacy_minus1_1, retired with checkpoint format v2"}

# How far outside [-1, 1] an input may sit before imagenet refuses it: float32
# rounding of uint8 / 255 * 2 - 1, not a second normalisation.
RANGE_SLACK = 1e-5


def apply_rope_2d(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Rotate x [..., d] with cos and sin [..., d // 2], even and odd interleaved.

    This is the interleaved convention, not the split-halves one the backbone
    uses. The two live in the same checkpoint and they differ.
    """
    even = x[..., 0::2]
    odd = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = even * cos - odd * sin
    out[..., 1::2] = even * sin + odd * cos
    return out


class PatchEmbeddings(nn.Module):
    """Named embeddings so the checkpoint keys map one to one."""

    def __init__(self, vision_hidden: int, backbone_hidden: int, patch: int, merge: int):
        super().__init__()
        self.patch_embedding = nn.Conv2d(
            3, vision_hidden, kernel_size=patch, stride=patch
        )
        self.dense_embedding = nn.Conv2d(
            vision_hidden, backbone_hidden, kernel_size=merge, stride=merge
        )


def refuse_pixel_range(low: float, high: float) -> None:
    """VisionTower.normalise's refusal: the pixels are not the producers' [-1, 1]."""
    raise ValueError(
        f"pixels span {low:.4f} to {high:.4f}, expected [-1, 1].\n"
        "Every producer hands the tower uint8 / 255 * 2 - 1; the tower "
        "applies ImageNet mean and std itself.\n"
        "Remove the normalisation upstream of the tower."
    )


class VisionTower(nn.Module):
    """pixels -> backbone tokens, one token per 32x32 pixels."""

    accepted_norms = INPUT_NORMS

    def __init__(
        self,
        *,
        vision_hidden: int,
        backbone_hidden: int,
        patch: int,
        downsample_ratio: float,
        rope_theta: float,
        input_norm: str,
    ):
        super().__init__()
        if input_norm in RETIRED_INPUT_NORMS:
            raise ValueError(
                f"input_norm is {input_norm!r}, the {RETIRED_INPUT_NORMS[input_norm]}.\n"
                f"Expected one of {list(self.accepted_norms)}: NEO reads its pixels "
                "through ImageNet mean and std.\n"
                "Build with model.vision.pixel_convention neo_native."
            )
        if input_norm not in self.accepted_norms:
            raise ValueError(
                f"input_norm is {input_norm!r}.\n"
                f"Expected one of {list(self.accepted_norms)}.\n"
                "Pass images.tower_norm(model.vision.pixel_convention)."
            )
        # A string and two constants rather than buffers, so the state dict,
        # the checkpoint keys and a dtype cast are all what they were. The
        # tensors are made once per dtype and device (imagenet_tables).
        self.input_norm = input_norm
        self.norm_tables: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        self.vision_hidden = vision_hidden
        self.patch = patch
        self.merge = int(round(1.0 / downsample_ratio))
        self.pixels_per_token = self.patch * self.merge
        self.rope_theta = rope_theta
        self.embeddings = PatchEmbeddings(
            vision_hidden, backbone_hidden, patch, self.merge
        )

    def grid_of(self, image_shapes: torch.Tensor) -> torch.Tensor:
        """[..., 2] pixel sizes -> [..., 2] token grids.

        Raises on a remainder. The loader snaps every image to a multiple of
        the token size, so a remainder means the loader and this tower disagree
        about the same image, and floor division would silently drop the bottom
        and right edges.
        """
        size = self.pixels_per_token
        remainder = image_shapes % size
        if int(remainder.abs().sum()) != 0:
            offender = image_shapes[remainder.any(-1)][0].tolist()
            raise ValueError(
                f"image shape {offender} is not a multiple of {size} pixels.\n"
                f"Expected both sides to be multiples of {size}, one token's worth.\n"
                "Snap the image in the loader; this tower will not crop it."
            )
        return image_shapes // size

    def grid_rope(
        self, grid_h: int, grid_w: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """A patch grid -> (cos, sin) by column and (cos, sin) by row, float32.

        Each axis rotates half the channels, so the frequencies run over
        arange(0, vision_hidden // 2, 2) and the tables are a quarter as wide
        as the feature. Computed here rather than held in a buffer, which a
        dtype cast would otherwise reach.
        """
        half = self.vision_hidden // 2
        steps = torch.arange(0, half, 2, dtype=torch.float32, device=device)
        inverse = 1.0 / (self.rope_theta ** (steps / half))
        rows, columns = torch.meshgrid(
            torch.arange(grid_h, dtype=torch.float32, device=device),
            torch.arange(grid_w, dtype=torch.float32, device=device),
            indexing="ij",
        )
        by_column = columns.reshape(-1)[:, None] * inverse
        by_row = rows.reshape(-1)[:, None] * inverse
        return by_column.cos(), by_column.sin(), by_row.cos(), by_row.sin()

    def normalise(self, pixels: torch.Tensor) -> torch.Tensor:
        """[-1, 1] pixels -> what the patch embedding reads.

        In float32 at least, (x + 1) / 2 recovers ToTensor's [0, 1] and NEO's
        mean and std follow. An input outside [-1, 1] raises: a producer that
        normalised already, or one that left uint8 unscaled. A producer that
        stopped at ToTensor's [0, 1] is inside the range and is not caught
        here; tests/data/test_images.py holds every producer to black at -1.
        """
        wide = pixels.to(torch.promote_types(pixels.dtype, torch.float32))
        if wide.numel() == 0:
            return wide
        low, high = torch.aminmax(wide)
        # In float64, as the Python floats this compared before: exact for every
        # float32, and a NaN passes as it did. Read where the scope reads it
        # (model.deferred): at once outside training, once a step inside it.
        outside = (low.double() < -1.0 - RANGE_SLACK) | (high.double() > 1.0 + RANGE_SLACK)
        require(~outside, lambda: refuse_pixel_range(float(low), float(high)))
        mean, std = self.imagenet_tables(wide.dtype, wide.device)
        return ((wide + 1.0) * 0.5 - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)

    def imagenet_tables(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """IMAGENET_MEAN and IMAGENET_STD as dtype tensors on device, made once each."""
        key = (dtype, device)
        if key not in self.norm_tables:
            self.norm_tables[key] = (
                torch.tensor(IMAGENET_MEAN, dtype=dtype, device=device),
                torch.tensor(IMAGENET_STD, dtype=dtype, device=device),
            )
        return self.norm_tables[key]

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """pixels [N, 3, H, W] in [-1, 1] -> [N, (H // 32) * (W // 32),
        backbone_hidden] in raster order.

        H and W are the canvas, shared by the whole call. Rows of the canvas
        that no image covers come back as ordinary tokens and are dropped by
        the caller's mask.
        """
        # Normalised in float32, then cast: the loader hands over float32 and
        # the tower is built at model.dtype, so the cast belongs here -- cuda
        # autocast covers training and nothing covers serving.
        pixels = self.normalise(pixels)
        pixels = pixels.to(self.embeddings.patch_embedding.weight.dtype)
        count = pixels.shape[0]
        canvas = torch.tensor(pixels.shape[-2:], device="cpu")
        tokens_h, tokens_w = self.grid_of(canvas).tolist()
        grid_h, grid_w = tokens_h * self.merge, tokens_w * self.merge

        features = F.gelu(self.embeddings.patch_embedding(pixels))
        features = features.permute(0, 2, 3, 1).reshape(count, grid_h * grid_w, -1)

        cos_x, sin_x, cos_y, sin_y = self.grid_rope(grid_h, grid_w, pixels.device)
        half = self.vision_hidden // 2
        wide = features.float()
        by_column = apply_rope_2d(wide[..., :half], cos_x[None], sin_x[None])
        by_row = apply_rope_2d(wide[..., half:], cos_y[None], sin_y[None])
        features = torch.cat([by_column, by_row], dim=-1).to(features.dtype)

        grid = features.view(count, grid_h, grid_w, self.vision_hidden)
        merged = self.embeddings.dense_embedding(grid.permute(0, 3, 1, 2))
        return merged.permute(0, 2, 3, 1).reshape(count, tokens_h * tokens_w, -1)
