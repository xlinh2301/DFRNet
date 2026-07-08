"""
DFRNet training objective:

    L = L_main + λ * L_aux + β * L_rec

L_main  — CTC loss on clean feature F through shared head
L_aux   — CTC loss on recovered feature F̂ through the SAME shared head
L_rec   — MSE reconstruction: ||F̂ - sg(F)||_2^2   (sg = stop gradient)

Sharing the head forces OFR to produce features the base recogniser
can already use — it cannot learn a shortcut classifier.
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class DFRNetLoss(nn.Layer):
    """
    Args:
        lambda_aux: weight for auxiliary CTC loss on recovered features
        beta_rec:   weight for L2 feature reconstruction loss
        blank_idx:  CTC blank token index (0 in PaddleOCR by convention)
    """

    def __init__(
        self,
        lambda_aux: float = 0.5,
        beta_rec: float = 0.1,
        blank_idx: int = 0,
        lambda_refine: float = 0.5,
    ):
        super().__init__()
        self.lambda_aux = lambda_aux
        self.beta_rec = beta_rec
        self.blank_idx = blank_idx
        self.lambda_refine = lambda_refine

    def _ctc_loss(self, logits: paddle.Tensor, labels: paddle.Tensor) -> paddle.Tensor:
        """
        Args:
            logits: (B, T, C) raw logits
            labels: (B, label_len) token ids, blank_idx used as padding
        """
        log_probs = F.log_softmax(logits, axis=2)
        log_probs = log_probs.transpose([1, 0, 2])  # (T, B, C)

        B, T, _ = logits.shape
        input_lengths = paddle.full([B], T, dtype="int64")
        label_lengths = (labels != self.blank_idx).cast("int64").sum(axis=1)

        return F.ctc_loss(
            log_probs,
            labels.cast("int32"),
            input_lengths.cast("int64"),
            label_lengths.cast("int64"),
            blank=self.blank_idx,
            reduction="mean",
        )

    def forward(
        self,
        logits_main: paddle.Tensor,
        logits_aux: paddle.Tensor,
        F_clean: paddle.Tensor,
        F_hat: paddle.Tensor,
        labels: paddle.Tensor,
        logits_refined: paddle.Tensor | None = None,
    ) -> dict:
        l_main = self._ctc_loss(logits_main, labels)
        l_aux = self._ctc_loss(logits_aux, labels)
        # stop gradient on target: OFR learns to approach F, not the other way around
        l_rec = F.mse_loss(F_hat, F_clean.detach())

        loss = l_main + self.lambda_aux * l_aux + self.beta_rec * l_rec

        result = {
            "loss": loss,
            "loss_main": l_main,
            "loss_aux": l_aux,
            "loss_rec": l_rec,
        }
        if logits_refined is not None:
            l_refine = self._ctc_loss(logits_refined, labels)
            result["loss"] = result["loss"] + self.lambda_refine * l_refine
            result["loss_refine"] = l_refine
        return result
