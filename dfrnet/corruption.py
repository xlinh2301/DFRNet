"""
Occlusion-aware Diffusion Feature Corruption (Paddle implementation).

Forward process:
    F_t = M_t ⊙ (sqrt(ā_t) * F + sqrt(1 - ā_t) * ε)

M_t ∈ {0,1}^{B×L×d}: structured span mask simulating character-block occlusion.
The Gaussian term models low-level degradation (blur, noise).

Novelty over plain Gaussian diffusion (IPAD) and PerturbCTC:
  - M_t explicitly zeros contiguous token spans, directly simulating
    physical occlusion of character regions in the feature sequence.
"""

import math
import paddle
import paddle.nn as nn


def _cosine_alpha_bar(T: int) -> paddle.Tensor:
    """Cosine schedule ā_t (Nichol & Dhariwal, 2021)."""
    steps = paddle.arange(T + 1, dtype="float32")
    alphas_bar = paddle.cos(((steps / T) + 0.008) / 1.008 * math.pi / 2) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    return alphas_bar  # (T+1,)


class OcclusionDiffusionCorruption(nn.Layer):
    """
    Corrupts a clean feature F into F_t given timestep t.

    Args:
        T:              total diffusion steps
        mask_ratio_max: max fraction of tokens masked at t=T
        span_len:       length of each contiguous occlusion span
    """

    def __init__(self, T: int = 1000, mask_ratio_max: float = 0.5, span_len: int = 3):
        super().__init__()
        self.T = T
        self.mask_ratio_max = mask_ratio_max
        self.span_len = span_len

        alphas_bar = _cosine_alpha_bar(T)
        self.register_buffer("alphas_bar", alphas_bar)

    def _build_occlusion_mask(
        self, B: int, L: int, d: int, t: paddle.Tensor
    ) -> paddle.Tensor:
        """
        M_t ∈ {0,1}^{B×L×d}: mask ratio scales with t/T.
        Spans are contiguous blocks of `span_len` tokens.
        """
        ratios = (t.cast("float32") / self.T) * self.mask_ratio_max  # (B,)
        mask = paddle.ones([B, L], dtype="float32")

        for b in range(B):
            n_mask = int(ratios[b].item() * L)
            if n_mask == 0:
                continue
            n_spans = max(1, n_mask // self.span_len)
            perm = paddle.randperm(max(1, L - self.span_len))[:n_spans]
            for s in perm.numpy().tolist():
                end = min(s + self.span_len, L)
                mask[b, s:end] = 0.0

        return mask.unsqueeze(-1).expand([B, L, d])  # (B, L, d)

    def forward(
        self, F: paddle.Tensor, t: paddle.Tensor
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        """
        Args:
            F: clean feature  (B, L, d)
            t: timestep  (B,)  integers in [1, T]

        Returns:
            F_t:  corrupted feature  (B, L, d)
            M_t:  occlusion mask     (B, L, d)
        """
        B, L, d = F.shape

        alpha_bar_t = paddle.index_select(self.alphas_bar, t, axis=0)  # (B,)
        sqrt_alpha = alpha_bar_t.sqrt().unsqueeze([1, 2])               # (B,1,1)
        sqrt_one_minus = (1.0 - alpha_bar_t).sqrt().unsqueeze([1, 2])

        eps = paddle.randn_like(F)
        F_noisy = sqrt_alpha * F + sqrt_one_minus * eps

        M_t = self._build_occlusion_mask(B, L, d, t)
        F_t = M_t * F_noisy

        return F_t, M_t
