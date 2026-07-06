"""
DFRNet smoke test — pure PyTorch, no Paddle/PaddleOCR required.

Validates:
  1. OcclusionDiffusionCorruption forward pass
  2. OFRModule forward + backward
  3. DFRNet training step (shared CTC head)
  4. Loss decreases over 50 iterations on synthetic data

Mirrors the Paddle implementation exactly.
Run with:
    python tools/smoke_test_torch.py
"""

import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Corruption (mirrors dfrnet/corruption.py)
# ──────────────────────────────────────────────────────────────────────────────

def _cosine_alpha_bar(T: int) -> torch.Tensor:
    steps = torch.arange(T + 1, dtype=torch.float64)
    ab = torch.cos(((steps / T) + 0.008) / 1.008 * math.pi / 2) ** 2
    return (ab / ab[0]).float()


class OcclusionDiffusionCorruption(nn.Module):
    def __init__(self, T=1000, mask_ratio_max=0.5, span_len=3):
        super().__init__()
        self.T = T
        self.mask_ratio_max = mask_ratio_max
        self.span_len = span_len
        self.register_buffer("alphas_bar", _cosine_alpha_bar(T))

    def _build_mask(self, B, L, d, t, device):
        ratios = (t.float() / self.T) * self.mask_ratio_max
        mask = torch.ones(B, L, device=device)
        for b in range(B):
            n_mask = int(ratios[b].item() * L)
            if n_mask == 0:
                continue
            n_spans = max(1, n_mask // self.span_len)
            starts = torch.randperm(max(1, L - self.span_len), device=device)[:n_spans]
            for s in starts.tolist():
                mask[b, s : s + self.span_len] = 0.0
        return mask.unsqueeze(-1).expand(B, L, d)

    def forward(self, F, t):
        B, L, d = F.shape
        ab = self.alphas_bar[t].to(F.device)
        sqrt_a = ab.sqrt().view(B, 1, 1)
        sqrt_1ma = (1 - ab).sqrt().view(B, 1, 1)
        eps = torch.randn_like(F)
        F_noisy = sqrt_a * F + sqrt_1ma * eps
        M = self._build_mask(B, L, d, t, F.device)
        return M * F_noisy, M


# ──────────────────────────────────────────────────────────────────────────────
# OFR Module (mirrors dfrnet/ofr_module.py)
# ──────────────────────────────────────────────────────────────────────────────

class SinusoidalTimeEmb(nn.Module):
    def __init__(self, T, dim):
        super().__init__()
        self.T = float(T)
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freq = math.log(10000) / (half - 1)
        freq = torch.exp(torch.arange(half, device=t.device) * -freq)
        t_s = t.float() / self.T * 8000.0
        emb = t_s[:, None] * freq[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class AdaLayerNorm(nn.Module):
    def __init__(self, dim, T):
        super().__init__()
        self.emb = SinusoidalTimeEmb(T, dim)
        self.proj = nn.Linear(dim, dim * 2)
        self.silu = nn.SiLU()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, t):
        emb = self.proj(self.silu(self.emb(t))).unsqueeze(1)
        scale, shift = emb.chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class OFRLayer(nn.Module):
    def __init__(self, dim, nhead, T, ffn_ratio=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        self.norm1 = AdaLayerNorm(dim, T)
        self.drop = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ffn_ratio), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim * ffn_ratio, dim), nn.Dropout(dropout),
        )
        self.norm2 = AdaLayerNorm(dim, T)

    def forward(self, x, t):
        a, _ = self.attn(x, x, x)
        x = self.norm1(x + self.drop(a), t)
        x = self.norm2(x + self.ff(x), t)
        return x


class OFRModule(nn.Module):
    def __init__(self, dim, nhead=4, depth=2, T=1000, ffn_ratio=4, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [OFRLayer(dim, nhead, T, ffn_ratio, dropout) for _ in range(depth)]
        )
        self.out_proj = nn.Linear(dim, dim)
        nn.init.eye_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, F_t, t):
        x = F_t
        for layer in self.layers:
            x = layer(x, t)
        return self.out_proj(x)


# ──────────────────────────────────────────────────────────────────────────────
# Toy DFRNet (synthetic backbone — just a Linear for testing)
# ──────────────────────────────────────────────────────────────────────────────

class ToyDFRNet(nn.Module):
    """
    Replaces the real PPOCRv5 backbone with a random Linear for smoke testing.
    Everything else mirrors the real DFRNet exactly.
    """
    def __init__(self, feat_dim=120, seq_len=25, num_classes=11, T=1000):
        super().__init__()
        self.feat_dim = feat_dim
        self.seq_len = seq_len
        self.T = T

        # simulated encoder (PPLCNetV3 + SVTR → (B, L, d))
        self.encoder = nn.Linear(feat_dim, feat_dim)

        # shared CTC head — the key constraint
        self.ctc_fc = nn.Linear(feat_dim, num_classes)

        # OFR branch (training-only)
        self.corruption = OcclusionDiffusionCorruption(T=T, mask_ratio_max=0.5, span_len=3)
        self.ofr = OFRModule(dim=feat_dim, nhead=4, depth=2, T=T)

    def forward(self, F_raw):
        F_clean = self.encoder(F_raw)           # (B, L, d)
        B = F_clean.shape[0]
        t = torch.randint(1, self.T + 1, (B,), device=F_clean.device)

        F_t, _ = self.corruption(F_clean.detach(), t)  # detach for corruption input
        F_hat = self.ofr(F_t, t)

        logits_main = self.ctc_fc(F_clean)      # (B, L, C)
        logits_aux  = self.ctc_fc(F_hat)        # shared head

        return logits_main, logits_aux, F_clean, F_hat


# ──────────────────────────────────────────────────────────────────────────────
# Loss (mirrors dfrnet/loss.py)
# ──────────────────────────────────────────────────────────────────────────────

def ctc_loss(logits, labels, blank=0):
    """logits: (B, T, C), labels: (B, label_len) padded with blank."""
    log_probs = F.log_softmax(logits, dim=2).permute(1, 0, 2)  # (T, B, C)
    B, T, _ = logits.shape
    input_lengths = torch.full((B,), T, dtype=torch.long, device=logits.device)
    label_lengths = (labels != blank).sum(dim=1)
    return F.ctc_loss(log_probs, labels, input_lengths, label_lengths,
                      blank=blank, reduction="mean", zero_infinity=True)


def dfrnet_loss(logits_main, logits_aux, F_clean, F_hat, labels,
                lambda_aux=0.5, beta_rec=0.1):
    l_main = ctc_loss(logits_main, labels)
    l_aux  = ctc_loss(logits_aux,  labels)
    l_rec  = F.mse_loss(F_hat, F_clean.detach())
    loss = l_main + lambda_aux * l_aux + beta_rec * l_rec
    return loss, l_main, l_aux, l_rec


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic data
# ──────────────────────────────────────────────────────────────────────────────

def make_batch(B=8, seq_len=25, feat_dim=120, num_classes=11, label_len=8, device="cpu"):
    F_raw = torch.randn(B, seq_len, feat_dim, device=device)
    # labels: random digit sequences (1–10), padded with blank (0)
    labels = torch.zeros(B, label_len, dtype=torch.long, device=device)
    for b in range(B):
        length = torch.randint(3, label_len + 1, (1,)).item()
        labels[b, :length] = torch.randint(1, num_classes, (length,))
    return F_raw, labels


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    FEAT_DIM = 120
    SEQ_LEN = 25
    NUM_CLASSES = 11
    BATCH = 8
    STEPS = 200

    model = ToyDFRNet(feat_dim=FEAT_DIM, seq_len=SEQ_LEN, num_classes=NUM_CLASSES).to(device)

    # two param groups — OFR gets full lr, encoder gets 0.1x (mirrors real training)
    encoder_params = list(model.encoder.parameters()) + list(model.ctc_fc.parameters())
    ofr_params = list(model.ofr.parameters())
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": 5e-5},
        {"params": ofr_params,     "lr": 5e-4},
    ], weight_decay=3e-5)

    print(f"\n{'step':>5}  {'loss':>8}  {'l_main':>8}  {'l_aux':>8}  {'l_rec':>8}")
    print("-" * 50)

    history = []
    for step in range(1, STEPS + 1):
        model.train()
        F_raw, labels = make_batch(BATCH, SEQ_LEN, FEAT_DIM, NUM_CLASSES, device=device)

        logits_main, logits_aux, F_clean, F_hat = model(F_raw)
        loss, l_main, l_aux, l_rec = dfrnet_loss(
            logits_main, logits_aux, F_clean, F_hat, labels
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        history.append(loss.item())

        if step % 20 == 0 or step == 1:
            print(f"{step:>5}  {loss.item():>8.4f}  {l_main.item():>8.4f}"
                  f"  {l_aux.item():>8.4f}  {l_rec.item():>8.4f}")

    # ── Convergence check ──────────────────────────────────────────────
    first20  = sum(history[:20])  / 20
    last20   = sum(history[-20:]) / 20
    decreased = last20 < first20

    print()
    print("=" * 50)
    print(f"  avg loss (steps   1-20) : {first20:.4f}")
    print(f"  avg loss (steps 181-200): {last20:.4f}")
    print(f"  Loss decreased?        : {'YES' if decreased else 'NO'}")
    print()

    # ── Gradient flow check ───────────────────────────────────────────
    print("Gradient flow check (last backward):")
    for name, p in model.named_parameters():
        if p.grad is not None:
            gnorm = p.grad.norm().item()
            print(f"  {name:<55} grad_norm={gnorm:.4e}")
        else:
            print(f"  {name:<55} NO GRAD (frozen or unused)")

    print()
    if decreased:
        print("PASS - DFRNet training loop converges on synthetic data.")
        sys.exit(0)
    else:
        print("WARN - Loss did not decrease. Check gradient flow above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
