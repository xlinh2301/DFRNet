"""
Lightweight bidirectional refinement head (BCN-style, ABINet).

Unlike OFR (which refines the encoder's 120-d *feature* and is discarded at
inference), this head refines the CTC head's *class-probability* output and
runs at BOTH train and inference time. Each position attends to every other
position (bidirectional) but is masked from attending to itself, forcing it
to predict y_i purely from the context of the other positions rather than
copying its own (already-visual) prediction — this is what lets it recover
digits that are fully occluded, using the surrounding digit sequence.
"""

import paddle
import paddle.nn as nn


class RefineLayer(nn.Layer):
    def __init__(self, dim: int, nhead: int, ffn_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiHeadAttention(dim, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.drop1 = nn.Dropout(dropout)

        ffn_dim = dim * ffn_ratio
        self.ff = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: paddle.Tensor, attn_mask: paddle.Tensor) -> paddle.Tensor:
        attn_out = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + self.drop1(attn_out))
        x = self.norm2(x + self.ff(x))
        return x


class BidirectionalRefineHead(nn.Layer):
    """
    Args:
        num_classes: CTC class count (input/output dim, e.g. 11 for digits+blank)
        d_model:     internal attention dim (probability vector is projected up)
        nhead:       attention heads
        depth:       number of refine layers
    """

    def __init__(self, num_classes: int, d_model: int = 64, nhead: int = 4, depth: int = 2):
        super().__init__()
        self.in_proj = nn.Linear(num_classes, d_model)
        self.layers = nn.LayerList(
            [RefineLayer(d_model, nhead) for _ in range(depth)]
        )
        self.out_proj = nn.Linear(d_model, num_classes)

    def forward(self, probs: paddle.Tensor) -> paddle.Tensor:
        """
        Args:
            probs: (B, T, C) CTC class probabilities (softmax output)

        Returns:
            refined_logits: (B, T, C) raw logits (caller applies log_softmax/softmax)
        """
        B, T, C = probs.shape

        # block each position from attending to itself
        diag_mask = paddle.eye(T, dtype="float32") * -1e9  # (T, T)
        attn_mask = diag_mask.unsqueeze([0, 1])  # (1, 1, T, T), broadcast over B/heads

        x = self.in_proj(probs)
        for layer in self.layers:
            x = layer(x, attn_mask)
        return self.out_proj(x)
