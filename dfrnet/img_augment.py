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
        if float(paddle.rand([1])) > p:
            continue
        n = int(paddle.randint(1, num_patches + 1, shape=[1]))
        for _ in range(n):
            y = int(paddle.randint(0, max(1, H - side + 1), shape=[1]))
            x = int(paddle.randint(0, max(1, W - side + 1), shape=[1]))
            out[b, :, y : y + side, x : x + side] = 0.0
    return out
