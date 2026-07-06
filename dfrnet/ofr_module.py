"""
Occlusion-aware Feature Refinement (OFR) Module (Paddle implementation).

R_θ(F_t, t) → F̂

Lightweight transformer stack where each LayerNorm is replaced by
AdaLayerNorm conditioned on the diffusion timestep t (following IPAD).

Design constraints:
  - No cross-attention to image: OFR refines from F_t alone
  - 2 layers by default — backbone dominates capacity, OFR acts as corrector
  - Output dim == input dim so the shared CTC head needs no adaptation
"""

import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class SinusoidalTimeEmb(nn.Layer):
    def __init__(self, T: int, dim: int):
        super().__init__()
        self.T = float(T)
        self.dim = dim

    def forward(self, t: paddle.Tensor) -> paddle.Tensor:
        half = self.dim // 2
        freq = math.log(10000) / (half - 1)
        freq = paddle.exp(paddle.arange(half, dtype="float32") * -freq)
        t_scaled = t.cast("float32") / self.T * 8000.0
        emb = t_scaled.unsqueeze(1) * freq.unsqueeze(0)  # (B, half)
        return paddle.concat([paddle.sin(emb), paddle.cos(emb)], axis=-1)  # (B, dim)


class AdaLayerNorm(nn.Layer):
    """Scale-shift LayerNorm conditioned on timestep embedding."""

    def __init__(self, dim: int, T: int):
        super().__init__()
        self.time_emb = SinusoidalTimeEmb(T, dim)
        self.proj = nn.Linear(dim, dim * 2)
        self.silu = nn.Silu()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: paddle.Tensor, t: paddle.Tensor) -> paddle.Tensor:
        emb = self.proj(self.silu(self.time_emb(t))).unsqueeze(1)  # (B, 1, 2d)
        scale, shift = paddle.chunk(emb, 2, axis=-1)
        return self.norm(x) * (1 + scale) + shift


class OFRLayer(nn.Layer):
    def __init__(self, dim: int, nhead: int, T: int, ffn_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiHeadAttention(dim, nhead, dropout=dropout)
        self.norm1 = AdaLayerNorm(dim, T)
        self.drop1 = nn.Dropout(dropout)

        ffn_dim = dim * ffn_ratio
        self.ff = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = AdaLayerNorm(dim, T)

    def forward(self, x: paddle.Tensor, t: paddle.Tensor) -> paddle.Tensor:
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.drop1(attn_out), t)
        x = self.norm2(x + self.ff(x), t)
        return x


class OFRModule(nn.Layer):
    """
    Args:
        dim:       feature dimension (must match PPOCRv5 neck out_channels)
        nhead:     number of self-attention heads
        depth:     number of OFR layers
        T:         diffusion timesteps
        ffn_ratio: FFN hidden / dim
        dropout:   dropout rate
    """

    def __init__(
        self,
        dim: int,
        nhead: int = 4,
        depth: int = 2,
        T: int = 1000,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.LayerList(
            [OFRLayer(dim, nhead, T, ffn_ratio, dropout) for _ in range(depth)]
        )
        self.out_proj = nn.Linear(dim, dim)
        self._init_weights()

    def _init_weights(self):
        eye = paddle.eye(self.out_proj.weight.shape[0])
        self.out_proj.weight.set_value(eye)
        paddle.nn.initializer.Constant(0.0)(self.out_proj.bias)

    def forward(self, F_t: paddle.Tensor, t: paddle.Tensor) -> paddle.Tensor:
        """
        Args:
            F_t: corrupted feature  (B, L, d)
            t:   timestep  (B,)

        Returns:
            F_hat: recovered feature  (B, L, d)
        """
        x = F_t
        for layer in self.layers:
            x = layer(x, t)
        return self.out_proj(x)
