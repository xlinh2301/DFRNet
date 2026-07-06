"""
DFRNet Ablation — OFR Module Contribution Experiment
======================================================

Hypothesis:
    Training with the OFR branch forces the encoder to produce features
    that are robust to structured occlusion, because the OFR module must
    recover clean features from corrupted ones using the SAME CTC head.
    At inference (OFR off), this regularisation should manifest as better
    CTC accuracy on occluded inputs.

Experiment:
    Both models share the SAME architecture at inference:
        Encoder → CTC head → prediction

    The only difference is HOW they are trained:
        Baseline : Encoder + CTC head, standard CTC loss only
        DFRNet   : Encoder + CTC head + OFR branch, L = L_main + λ·L_aux + β·L_rec

    Both are evaluated on three test sets:
        clean     : unmodified encoder features
        occ-light : 20% of tokens zeroed (structured spans)
        occ-heavy : 50% of tokens zeroed (structured spans)

    If OFR contributes, DFRNet should outperform Baseline on occ-* sets
    while remaining comparable on the clean set.

Metrics:
    - CTC loss (lower = better)
    - Sequence accuracy (exact match after greedy CTC decode)
    - Relative degradation gap: how much accuracy drops from clean → occ
      (smaller gap = more robust encoder)
"""

import math
import sys
import random
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic task
# ──────────────────────────────────────────────────────────────────────────────
# We simulate a digit-recognition task:
#   - Encoder receives an image → produces F ∈ R^{L×d}
#   - CTC head decodes F → digit sequence
#
# For controlled experiments, the "encoder" is a 2-layer MLP that maps
# a deterministic input (a one-hot + noise) to features.  Labels are fixed
# digit sequences.  This gives a learnable mapping with a clear ground truth.

NUM_CLASSES = 12    # 0=blank, 1-10=digits, 11=<extra> (for length coverage)
FEAT_DIM    = 120
SEQ_LEN     = 25    # CTC input length
MAX_LABEL   = 8     # max digit sequence length
BLANK       = 0


def make_dataset(n_samples: int, device: str) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Returns a list of (input_signal, label) pairs.
    input_signal: (SEQ_LEN, FEAT_DIM) — structured noise tied to the label
    label:        (MAX_LABEL,)         — padded with BLANK
    """
    data = []
    for _ in range(n_samples):
        length  = random.randint(3, MAX_LABEL)
        digits  = [random.randint(1, NUM_CLASSES - 2) for _ in range(length)]
        label   = torch.zeros(MAX_LABEL, dtype=torch.long)
        label[:length] = torch.tensor(digits)

        # structured signal: for each sequence position, embed the digit as a
        # repeating pattern so the encoder can learn something real
        signal = torch.zeros(SEQ_LEN, FEAT_DIM)
        for i, d in enumerate(digits):
            start = i * (SEQ_LEN // MAX_LABEL)
            end   = start + (SEQ_LEN // MAX_LABEL)
            signal[start:end, d * (FEAT_DIM // NUM_CLASSES) :
                              (d + 1) * (FEAT_DIM // NUM_CLASSES)] = 1.0
        signal = signal + 0.2 * torch.randn_like(signal)

        data.append((signal.to(device), label.to(device)))
    return data


def apply_occlusion(F: torch.Tensor, ratio: float, span_len: int = 3,
                    corruption: "OcclusionDiffusionCorruption | None" = None) -> torch.Tensor:
    """
    Apply test-time occlusion using the same corruption type as training
    (Gaussian noise + structured span mask at high t), so train/test
    corruption distributions match.
    If corruption is None, falls back to pure zero-masking.
    """
    if corruption is not None:
        B, L, d = F.shape
        # fix t at a high value proportional to ratio so heavier ratio → heavier corruption
        T = corruption.T
        t_val = max(1, int(ratio * T))
        t = torch.full((B,), t_val, dtype=torch.long, device=F.device)
        F_t, _ = corruption(F, t)
        return F_t

    # fallback: pure zero-masking
    B, L, d = F.shape
    device  = F.device
    n_spans = max(1, int(ratio * L) // span_len)
    starts  = torch.rand(B, n_spans, device=device)
    starts  = (starts * (L - span_len)).long()
    pos     = torch.arange(L, device=device).view(1, 1, L)
    s       = starts.unsqueeze(-1)
    occ     = ((pos >= s) & (pos < s + span_len)).any(dim=1)
    return F * (~occ).float().unsqueeze(-1)


def collate(batch):
    signals = torch.stack([s for s, _ in batch])   # (B, L, d)
    labels  = torch.stack([l for _, l in batch])   # (B, MAX_LABEL)
    return signals, labels


# ──────────────────────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """Shared 2-layer MLP encoder (B, L, d) → (B, L, d)."""
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(dim * 2, dim), nn.LayerNorm(dim),
        )

    def forward(self, x):   # (B, L, d) → (B, L, d)
        return self.net(x)


class CTCHead(nn.Module):
    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):   # (B, L, d) → (B, L, C)
        return self.fc(x)


# ── OFR components ────────────────────────────────────────────────────────────

def _cosine_alpha_bar(T: int) -> torch.Tensor:
    steps = torch.arange(T + 1, dtype=torch.float64)
    ab = torch.cos(((steps / T) + 0.008) / 1.008 * math.pi / 2) ** 2
    return (ab / ab[0]).float()


class OcclusionDiffusionCorruption(nn.Module):
    def __init__(self, T=200, mask_ratio_max=0.5, span_len=3):
        super().__init__()
        self.T = T
        self.mask_ratio_max = mask_ratio_max
        self.span_len = span_len
        self.register_buffer("alphas_bar", _cosine_alpha_bar(T))

    def forward(self, F, t):
        B, L, d = F.shape
        device = F.device
        ab = self.alphas_bar[t]
        sqrt_a   = ab.sqrt().view(B, 1, 1)
        sqrt_1ma = (1 - ab).sqrt().view(B, 1, 1)
        eps = torch.randn_like(F)
        F_noisy = sqrt_a * F + sqrt_1ma * eps

        # vectorised mask: sample n_spans random start positions per batch
        ratios   = (t.float() / self.T) * self.mask_ratio_max   # (B,)
        n_spans  = (ratios * L / self.span_len).long().clamp(min=1)   # (B,)
        max_span = int(n_spans.max().item())

        # random starts: (B, max_span) in [0, L - span_len)
        starts = torch.rand(B, max_span, device=device)
        starts = (starts * (L - self.span_len)).long()           # (B, max_span)

        # build mask via scatter on (B, L) grid
        mask = torch.ones(B, L, device=device)
        pos  = torch.arange(L, device=device).view(1, 1, L)     # (1,1,L)
        s    = starts.unsqueeze(-1)                              # (B, max_span, 1)
        # span_mask[b, k, l] = 1 if l in [starts[b,k], starts[b,k]+span_len)
        span_mask = ((pos >= s) & (pos < s + self.span_len)).any(dim=1).float()  # (B,L)
        # zero only spans within the per-sample quota
        valid = (torch.arange(max_span, device=device).unsqueeze(0)
                 < n_spans.unsqueeze(1))                         # (B, max_span)
        quota_mask = ((pos >= s) & (pos < s + self.span_len) &
                      valid.unsqueeze(-1)).any(dim=1).float()    # (B, L)
        mask = mask - quota_mask                                 # 1=kept, 0=occluded
        m = mask.unsqueeze(-1)                                   # (B, L, 1)
        return m * F_noisy, m


class SinusoidalTimeEmb(nn.Module):
    def __init__(self, T, dim):
        super().__init__()
        self.T = float(T)
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freq = math.log(10000) / (half - 1)
        freq = torch.exp(torch.arange(half, device=t.device) * -freq)
        emb  = (t.float()[:, None] / self.T * 8000.0) * freq[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class AdaLN(nn.Module):
    def __init__(self, dim, T):
        super().__init__()
        self.emb  = SinusoidalTimeEmb(T, dim)
        self.proj = nn.Linear(dim, dim * 2)
        self.silu = nn.SiLU()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, t):
        e = self.proj(self.silu(self.emb(t))).unsqueeze(1)
        scale, shift = e.chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class OFRLayer(nn.Module):
    def __init__(self, dim, nhead, T):
        super().__init__()
        self.attn  = nn.MultiheadAttention(dim, nhead, batch_first=True)
        self.norm1 = AdaLN(dim, T)
        self.ff    = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.norm2 = AdaLN(dim, T)

    def forward(self, x, t):
        a, _ = self.attn(x, x, x)
        x = self.norm1(x + a, t)
        x = self.norm2(x + self.ff(x), t)
        return x


class OFRModule(nn.Module):
    def __init__(self, dim, nhead=4, depth=2, T=200):
        super().__init__()
        self.layers   = nn.ModuleList([OFRLayer(dim, nhead, T) for _ in range(depth)])
        self.out_proj = nn.Linear(dim, dim)
        nn.init.eye_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.T = T
        self.corruption = OcclusionDiffusionCorruption(T=T)

    def forward(self, F_clean, t):
        F_t, mask = self.corruption(F_clean, t)
        x = F_t
        for layer in self.layers:
            x = layer(x, t)
        return self.out_proj(x), mask


# ──────────────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────────────

def ctc_loss_fn(logits, labels):
    log_probs = F.log_softmax(logits, dim=2).permute(1, 0, 2)
    B, T, _   = logits.shape
    input_len = torch.full((B,), T, dtype=torch.long, device=logits.device)
    label_len = (labels != BLANK).sum(dim=1)
    return F.ctc_loss(log_probs, labels, input_len, label_len,
                      blank=BLANK, reduction="mean", zero_infinity=True)


# ──────────────────────────────────────────────────────────────────────────────
# Greedy CTC decode → sequence accuracy
# ──────────────────────────────────────────────────────────────────────────────

def greedy_decode(logits: torch.Tensor) -> list[list[int]]:
    """Greedy CTC decode, return list of sequences (no blanks/repeats)."""
    preds = logits.argmax(-1)   # (B, T)
    results = []
    for seq in preds:
        decoded, prev = [], BLANK
        for tok in seq.tolist():
            if tok != prev and tok != BLANK:
                decoded.append(tok)
            prev = tok
        results.append(decoded)
    return results


def seq_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = greedy_decode(logits)
    correct = 0
    for pred, lab in zip(preds, labels.tolist()):
        gt = [t for t in lab if t != BLANK]
        correct += (pred == gt)
    return correct / len(labels)


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_baseline(encoder, ctc_head, dataset, epochs, batch_size, lr, device):
    """Standard CTC training — no OFR."""
    params = list(encoder.parameters()) + list(ctc_head.parameters())
    opt    = torch.optim.AdamW(params, lr=lr, weight_decay=3e-5)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history = []
    for epoch in range(epochs):
        random.shuffle(dataset)
        epoch_loss = 0.0
        for i in range(0, len(dataset), batch_size):
            batch  = dataset[i : i + batch_size]
            if len(batch) < 2:
                continue
            sig, lab = collate(batch)
            F  = encoder(sig)
            lp = ctc_head(F)
            loss = ctc_loss_fn(lp, lab)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            epoch_loss += loss.item()
        sched.step()
        history.append(epoch_loss)
    return history


def train_dfrnet(encoder, ctc_head, ofr, dataset, epochs, batch_size, lr,
                 lambda_aux, beta_rec, device):
    """
    DFRNet training:

        L = L_main + λ·L_aux + β·L_rec_masked

    L_rec_masked : MSE only on occluded positions — focused signal
    """
    all_params = (list(encoder.parameters()) +
                  list(ctc_head.parameters()) +
                  list(ofr.parameters()))
    opt   = torch.optim.AdamW(all_params, lr=lr, weight_decay=3e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history = []
    T = ofr.T
    for epoch in range(epochs):
        random.shuffle(dataset)
        epoch_loss = 0.0
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            if len(batch) < 2:
                continue
            sig, lab = collate(batch)
            B = sig.shape[0]

            F_clean = encoder(sig)
            t       = torch.randint(1, T + 1, (B,), device=device)

            F_t, mask = ofr.corruption(F_clean, t)
            x = F_t
            for layer in ofr.layers:
                x = layer(x, t)
            F_hat = ofr.out_proj(x)

            lp_main = ctc_head(F_clean)
            lp_aux  = ctc_head(F_hat)

            l_main = ctc_loss_fn(lp_main, lab)
            l_aux  = ctc_loss_fn(lp_aux,  lab)

            # MSE only on occluded positions
            occ_mask = (mask == 0).float()
            n_masked = occ_mask.sum().clamp(min=1)
            l_rec    = ((F_hat - F_clean.detach()) ** 2 * occ_mask).sum() / n_masked

            loss = l_main + lambda_aux * l_aux + beta_rec * l_rec

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            opt.step()
            epoch_loss += loss.item()
        sched.step()
        history.append(epoch_loss)
    return history


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    loss: float
    acc:  float


def evaluate(encoder, ctc_head, dataset, batch_size,
             occlusion_ratio=0.0, corruption=None) -> EvalResult:
    encoder.eval()
    ctc_head.eval()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            if not batch:
                continue
            sig, lab = collate(batch)
            F = encoder(sig)
            if occlusion_ratio > 0:
                F = apply_occlusion(F, occlusion_ratio, corruption=corruption)
            lp = ctc_head(F)
            total_loss += ctc_loss_fn(lp, lab).item()
            total_acc  += seq_accuracy(lp, lab)
            n_batches  += 1
    encoder.train()
    ctc_head.train()
    return EvalResult(total_loss / max(n_batches, 1), total_acc / max(n_batches, 1))


# ──────────────────────────────────────────────────────────────────────────────
# Main experiment
# ──────────────────────────────────────────────────────────────────────────────

def run_trial(trial_id: int, device: str, epochs: int, batch_size: int):
    torch.manual_seed(SEED + trial_id)

    # shared dataset
    train_data = make_dataset(2000, device)
    test_data  = make_dataset(500, device)

    # ── Baseline ──────────────────────────────────────────────────────
    enc_base = Encoder(FEAT_DIM).to(device)
    ctc_base = CTCHead(FEAT_DIM, NUM_CLASSES).to(device)
    train_baseline(enc_base, ctc_base, train_data, epochs, batch_size, lr=5e-4, device=device)

    # ── DFRNet ────────────────────────────────────────────────────────
    enc_dfr = Encoder(FEAT_DIM).to(device)
    ctc_dfr = CTCHead(FEAT_DIM, NUM_CLASSES).to(device)
    ofr     = OFRModule(FEAT_DIM, nhead=4, depth=2, T=200).to(device)
    train_dfrnet(enc_dfr, ctc_dfr, ofr, train_data, epochs, batch_size,
                 lr=5e-4, lambda_aux=0.5, beta_rec=0.5, device=device)

    # shared corruption for consistent test-time occlusion
    shared_corruption = OcclusionDiffusionCorruption(T=200).to(device)

    # ── Evaluate both on three test conditions ─────────────────────────
    results = {}
    for name, enc, ctc in [("Baseline", enc_base, ctc_base), ("DFRNet", enc_dfr, ctc_dfr)]:
        results[name] = {
            "clean":     evaluate(enc, ctc, test_data, batch_size, occlusion_ratio=0.0),
            "occ-light": evaluate(enc, ctc, test_data, batch_size, occlusion_ratio=0.2,
                                  corruption=shared_corruption),
            "occ-heavy": evaluate(enc, ctc, test_data, batch_size, occlusion_ratio=0.5,
                                  corruption=shared_corruption),
        }
    return results


def print_results(all_trials: list[dict]):
    n = len(all_trials)
    models    = ["Baseline", "DFRNet"]
    test_sets = ["clean", "occ-light", "occ-heavy"]

    # aggregate
    agg: dict = {m: {t: {"loss": [], "acc": []} for t in test_sets} for m in models}
    for trial in all_trials:
        for m in models:
            for t in test_sets:
                r = trial[m][t]
                agg[m][t]["loss"].append(r.loss)
                agg[m][t]["acc"].append(r.acc)

    def mean(xs):  return sum(xs) / len(xs)
    def std(xs):
        mu = mean(xs)
        return (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5

    print("\n" + "=" * 72)
    print("  ABLATION RESULTS  (mean +/- std over {} trials)".format(n))
    print("=" * 72)
    print(f"  {'':12}  {'clean acc':>12}  {'occ-light acc':>14}  {'occ-heavy acc':>14}")
    print("-" * 72)
    for m in models:
        row = f"  {m:<12}"
        for t in test_sets:
            accs  = agg[m][t]["acc"]
            mu, sd = mean(accs), std(accs)
            row  += f"  {mu*100:>9.1f}%{'':2}"
        print(row)

    print()
    print(f"  {'':12}  {'clean loss':>12}  {'occ-light loss':>14}  {'occ-heavy loss':>14}")
    print("-" * 72)
    for m in models:
        row = f"  {m:<12}"
        for t in test_sets:
            losses  = agg[m][t]["loss"]
            mu, sd  = mean(losses), std(losses)
            row    += f"  {mu:>10.4f}{'':4}"
        print(row)

    print()
    print("  ROBUSTNESS GAP  (clean acc - occ acc):  lower = more robust")
    print("-" * 72)
    for m in models:
        clean_acc  = mean(agg[m]["clean"]["acc"])
        light_gap  = clean_acc - mean(agg[m]["occ-light"]["acc"])
        heavy_gap  = clean_acc - mean(agg[m]["occ-heavy"]["acc"])
        print(f"  {m:<12}  occ-light gap: {light_gap*100:>6.1f}%    occ-heavy gap: {heavy_gap*100:>6.1f}%")

    print()
    print("  DELTA  (DFRNet - Baseline, positive = DFRNet better)")
    print("-" * 72)
    for t in test_sets:
        dfr_acc  = mean(agg["DFRNet"][t]["acc"])
        base_acc = mean(agg["Baseline"][t]["acc"])
        delta    = (dfr_acc - base_acc) * 100
        sign     = "+" if delta >= 0 else ""
        print(f"  {t:<14}  acc delta: {sign}{delta:.1f}%")

    print("=" * 72)

    # verdict
    dfr_heavy  = mean(agg["DFRNet"]["occ-heavy"]["acc"])
    base_heavy = mean(agg["Baseline"]["occ-heavy"]["acc"])
    if dfr_heavy > base_heavy:
        print("\n  VERDICT: OFR module CONTRIBUTES -- DFRNet is more robust to occlusion.")
    else:
        print("\n  VERDICT: No clear contribution detected on this synthetic task.")
    print()


def main():
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS  = 120
    BATCH   = 64
    TRIALS  = 5

    print(f"Device : {device}")
    print(f"Epochs : {EPOCHS} per model per trial")
    print(f"Trials : {TRIALS}")
    print(f"Task   : {NUM_CLASSES-1} class CTC, seq_len={SEQ_LEN}, feat_dim={FEAT_DIM}")
    print()

    all_trials = []
    for trial in range(TRIALS):
        print(f"--- Trial {trial + 1}/{TRIALS} ---")
        results = run_trial(trial, device, EPOCHS, BATCH)
        all_trials.append(results)

        # quick per-trial preview
        for m in ["Baseline", "DFRNet"]:
            r_c = results[m]["clean"]
            r_h = results[m]["occ-heavy"]
            print(f"  {m:<12}  clean acc={r_c.acc*100:.1f}%  occ-heavy acc={r_h.acc*100:.1f}%")
        print()

    print_results(all_trials)


if __name__ == "__main__":
    main()
