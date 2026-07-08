"""
Confidence-guided occlusion: probe the model's own CTC frame confidence,
then occlude the image regions corresponding to low-confidence frames —
training the model to become robust exactly where it is currently weakest,
instead of uniformly random image regions (dfrnet.img_augment).

Risk (worth testing empirically, not assuming): occluding the very region
the model is already unsure about removes the visual evidence for that
position. Prior image-space-occlusion ablation (img_left/img_right) showed
context alone cannot recover fully-missing characters for this dataset, so
this technique could teach the model to give up at hard positions rather
than get better at them.
"""

import paddle
import paddle.nn.functional as F


def confidence_guided_occlude(model, images: paddle.Tensor, conf_threshold: float = 0.8):
    """
    Args:
        model:          a DFRNet instance (uses model.encode + model.ctc_fc
                         as a read-only probe — does not affect gradients)
        images:         (B, C, H, W)
        conf_threshold: frames with max class-probability below this are
                         considered "weak" and occluded

    Returns:
        images with the image column-range under each weak CTC frame zeroed.
    """
    was_training = model.training
    model.eval()
    with paddle.no_grad():
        feat = model.encode(images)
        probs = F.softmax(model.ctc_fc(feat), axis=2)  # (B, T, C)
    if was_training:
        model.train()

    B, T, _ = probs.shape
    _, _, _, W = images.shape
    max_conf = probs.max(axis=2).numpy()  # (B, T)

    out = images.clone()
    frame_w = max(1, W // T)
    for b in range(B):
        for t in range(T):
            if max_conf[b, t] >= conf_threshold:
                continue
            x0 = t * frame_w
            x1 = min(W, x0 + frame_w)
            out[b, :, :, x0:x1] = 0.0
    return out
