"""
Image-space occlusion augmentation: random rectangular Cutout patches on the
input image, applied before the backbone.

Rationale: unlike dfrnet.corruption.OcclusionDiffusionCorruption (which
corrupts the *encoder's output feature*, producing out-of-distribution
vectors the backbone would never actually produce), Cutout corrupts the
pixels the backbone actually sees — real occlusion (dirt, glare, a finger)
is a pixel-level phenomenon, so this lets the backbone's own convolutions
propagate a realistic, in-distribution corruption pattern into the feature
space, training the encoder itself to be robust (no separate refinement
module needed).
"""

import random

import paddle


def random_cutout(
    images: paddle.Tensor,
    num_patches: int = 2,
    patch_size_frac: float = 0.15,
    p: float = 0.5,
) -> paddle.Tensor:
    """
    Args:
        images: (B, C, H, W), normalized to roughly [-1, 1]
        num_patches: max number of rectangular patches per image
        patch_size_frac: patch side length as a fraction of min(H, W)
        p: per-image probability of applying any cutout at all

    Returns:
        images with random patches zeroed (0.0 == mid-gray in this
        normalization), same shape.
    """
    B, C, H, W = images.shape
    side = max(1, int(patch_size_frac * min(H, W)))

    out = images.clone()
    for b in range(B):
        if random.random() > p:
            continue
        n = random.randint(1, num_patches)
        for _ in range(n):
            y = random.randint(0, max(0, H - side))
            x = random.randint(0, max(0, W - side))
            out[b, :, y : y + side, x : x + side] = 0.0
    return out


OCCLUSION_MODES = ("top", "bottom", "left", "right", "random_pixels")


def apply_occlusion_mask(
    images: paddle.Tensor, mode: str, frac: float = 0.5
) -> paddle.Tensor:
    """
    Deterministic single-mode image-space occlusion, applied to the whole
    batch. Used both as a training augmentation primitive and as a fixed
    eval-time corruption (same mode/frac for every sample -> reproducible
    per-mode accuracy).

    Args:
        images: (B, C, H, W)
        mode:   one of OCCLUSION_MODES
        frac:   fraction of the relevant dimension (H for top/bottom, W for
                left/right, total pixels for random_pixels) to zero out
    """
    out = images.clone()
    B, C, H, W = out.shape

    if mode == "top":
        out[:, :, : int(frac * H), :] = 0.0
    elif mode == "bottom":
        out[:, :, H - int(frac * H) :, :] = 0.0
    elif mode == "left":
        out[:, :, :, : int(frac * W)] = 0.0
    elif mode == "right":
        out[:, :, :, W - int(frac * W) :] = 0.0
    elif mode == "random_pixels":
        keep = (paddle.rand([B, 1, H, W]) >= frac).cast(out.dtype)
        out = out * keep
    else:
        raise ValueError(f"unknown occlusion mode: {mode}")
    return out


def random_occlusion_mix(
    images: paddle.Tensor,
    p: float = 0.5,
    frac_range: tuple[float, float] = (0.2, 0.5),
) -> paddle.Tensor:
    """
    Training-time augmentation: per-sample, with probability `p`, pick one of
    the 5 occlusion modes at a random severity in `frac_range` and apply it.
    Varying severity (rather than always exactly 50%) avoids every batch
    containing samples with total information loss in one half of the image.
    """
    out = images.clone()
    B = out.shape[0]
    for b in range(B):
        if random.random() > p:
            continue
        mode = random.choice(OCCLUSION_MODES)
        frac = random.uniform(*frac_range)
        out[b : b + 1] = apply_occlusion_mask(out[b : b + 1], mode, frac)
    return out
