"""
DFRNet: Diffusion Feature Refinement Network for Occluded Scene Text Recognition.

Checkpoint key mapping (PPOCRv5 best_accuracy.pdparams):
  backbone.*           → self.backbone.*
  head.ctc_encoder.encoder.* → self.ctc_encoder.*   (EncoderWithSVTR, dim=120)
  head.ctc_head.fc.*   → self.ctc_fc.*        (Linear 120 → 11)
  head.gtc_head.*      → (skipped, NRTR not used)
  head.before_gtc.*    → (skipped)

Training pipeline:

    Image ──► backbone ──► ctc_encoder ──► F (B, L, 120)
                                            │
                    ┌───────────────────────┴──────────────────────────┐
                    │                                                   │
                 ctc_fc                          OcclusionDiffusionCorruption
                    │                                                   │
                L_main                                         F_t (corrupted)
                                                                       │
                                                             OFR Module R_θ(F_t, t)
                                                                       │
                                                                  F̂ (recovered)
                                                                       │
                                                    ┌──────────────────┴────────────┐
                                                 ctc_fc (shared)              L2 Rec Loss
                                                    │                               │
                                                 L_aux                           L_rec

Inference: Image → backbone → ctc_encoder → F → ctc_fc → logits
           OFR branch not executed — zero inference latency.
"""

import os
import sys
import pickle

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

_PADDLE_OCR_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../PaddleOCR")
)
if _PADDLE_OCR_ROOT not in sys.path:
    sys.path.insert(0, _PADDLE_OCR_ROOT)

from ppocr.modeling.backbones import build_backbone
from ppocr.modeling.necks.rnn import EncoderWithSVTR

from .corruption import OcclusionDiffusionCorruption
from .ofr_module import OFRModule


class DFRNet(nn.Layer):
    """
    Args:
        backbone_cfg:    config for PPLCNetV3 (name, scale)
        svtr_cfg:        config for EncoderWithSVTR (dims, depth, hidden_dims, ...)
        num_classes:     output classes including blank (11 for digit charset)
        ofr_nhead:       attention heads in OFR (must divide svtr hidden_dims)
        ofr_depth:       number of OFR transformer layers
        T:               diffusion timesteps
        mask_ratio_max:  max occlusion fraction at t=T
        span_len:        tokens per occlusion span
        pretrained:      path to PPOCRv5 .pdparams checkpoint
    """

    def __init__(
        self,
        backbone_cfg: dict,
        svtr_cfg: dict,
        num_classes: int = 11,
        ofr_nhead: int = 4,
        ofr_depth: int = 2,
        T: int = 1000,
        mask_ratio_max: float = 0.5,
        span_len: int = 3,
        pretrained: str | None = None,
    ):
        super().__init__()

        # ── PPOCRv5 backbone ───────────────────────────────────────────
        cfg = dict(backbone_cfg)
        cfg.setdefault("in_channels", 3)
        self.backbone = build_backbone(cfg, model_type="rec")

        # ── SVTR encoder (was head.ctc_encoder in PPOCRv5) ────────────
        svtr = dict(svtr_cfg)
        svtr["in_channels"] = self.backbone.out_channels
        self.ctc_encoder = EncoderWithSVTR(**svtr)

        feat_dim = svtr.get("hidden_dims", svtr.get("dims", 120))

        # ── Shared CTC linear head ─────────────────────────────────────
        self.ctc_fc = nn.Linear(feat_dim, num_classes)

        # ── Training-only OFR branch ───────────────────────────────────
        self.corruption = OcclusionDiffusionCorruption(
            T=T, mask_ratio_max=mask_ratio_max, span_len=span_len
        )
        self.ofr = OFRModule(dim=feat_dim, nhead=ofr_nhead, depth=ofr_depth, T=T)
        self.T = T

        if pretrained is not None:
            self._load_pretrained(pretrained)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def _load_pretrained(self, path: str):
        """
        Load backbone + SVTR encoder + CTC head from a PPOCRv5 .pdparams.

        Key remapping:
            backbone.*          → backbone.*
            head.ctc_encoder.*  → ctc_encoder.*
            head.ctc_head.fc.*  → ctc_fc.*

        OFR weights are randomly initialised (new module, no pretrained source).
        """
        with open(path, "rb") as f:
            raw = pickle.load(f)

        model_state = self.state_dict()
        matched, skipped = {}, []

        remap_prefix = {
            "head.ctc_encoder.encoder.": "ctc_encoder.",
            "head.ctc_head.fc.": "ctc_fc.",
        }

        for ck, cv in raw.items():
            if ck == "StructuredToParameterName@@":
                continue

            # apply prefix remapping
            mk = ck
            for src, dst in remap_prefix.items():
                if ck.startswith(src):
                    mk = dst + ck[len(src):]
                    break

            if mk not in model_state:
                skipped.append((ck, "key not in model"))
                continue

            # convert raw numpy/paddle tensor to paddle Tensor
            import numpy as np
            if hasattr(cv, "numpy"):
                arr = cv.numpy()
            elif isinstance(cv, np.ndarray):
                arr = cv
            else:
                arr = np.array(cv)

            expected = model_state[mk].shape
            if list(arr.shape) != list(expected):
                skipped.append((ck, f"shape mismatch: ckpt {arr.shape} vs model {expected}"))
                continue

            matched[mk] = paddle.to_tensor(arr)

        self.set_state_dict(matched)
        print(
            f"[DFRNet] Loaded {len(matched)}/{len(model_state)} params from checkpoint."
        )
        if skipped:
            print(f"[DFRNet] Skipped {len(skipped)} entries:")
            for ck, reason in skipped[:10]:
                print(f"  {ck} — {reason}")
            if len(skipped) > 10:
                print(f"  ... and {len(skipped) - 10} more (likely gtc_head, before_gtc)")

    # ------------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------------

    def encode(self, images: paddle.Tensor) -> paddle.Tensor:
        """
        backbone → ctc_encoder → F (B, L, d)

        EncoderWithSVTR returns (B, L, d) directly.
        """
        x = self.backbone(images)
        x = self.ctc_encoder(x)
        if x.ndim == 4:
            # fallback: (B, d, 1, W) → (B, W, d)
            x = x.squeeze(2).transpose([0, 2, 1])
        return x  # (B, L, d)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        images: paddle.Tensor,
        labels: paddle.Tensor | None = None,
    ) -> dict | paddle.Tensor:
        """
        Training  → dict(logits_main, logits_aux, F_clean, F_hat)
        Inference → logits (B, T, C)  [softmax applied, matches PPOCRv5 output]
        """
        F_clean = self.encode(images)  # (B, L, d)

        if not self.training:
            logits = self.ctc_fc(F_clean)
            return F.softmax(logits, axis=2)

        B = F_clean.shape[0]
        t = paddle.randint(1, self.T + 1, shape=[B])

        F_t, _ = self.corruption(F_clean, t)
        F_hat = self.ofr(F_t, t)

        logits_main = self.ctc_fc(F_clean)
        logits_aux = self.ctc_fc(F_hat)   # shared head — no new params

        return {
            "logits_main": logits_main,
            "logits_aux": logits_aux,
            "F_clean": F_clean,
            "F_hat": F_hat,
        }
