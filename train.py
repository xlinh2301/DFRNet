"""
DFRNet training entry point.

Usage:
    python train.py --config configs/dfrnet.yaml
"""

import argparse
import logging
import os
import sys

import yaml
import paddle
import paddle.optimizer as optim
from paddle.io import DataLoader

# Make PaddleOCR importable
_ROOT = os.path.dirname(os.path.abspath(__file__))
_PADDLE_OCR_ROOT = os.path.abspath(os.path.join(_ROOT, "../PaddleOCR"))
if _PADDLE_OCR_ROOT not in sys.path:
    sys.path.insert(0, _PADDLE_OCR_ROOT)

from ppocr.data import build_dataloader
from ppocr.postprocess import build_post_process
from ppocr.metrics import build_metric

from dfrnet import DFRNet, DFRNetLoss
from dfrnet.img_augment import random_cutout, random_occlusion_mix


def build_optimizer(model: DFRNet, cfg: dict) -> optim.Optimizer:
    """
    Two parameter groups: backbone+neck at lr * backbone_lr_ratio,
    OFR + CTC head at full lr.
    """
    base_lr = cfg["learning_rate"]
    backbone_lr_ratio = cfg.get("backbone_lr_ratio", 0.1)
    wd = cfg.get("weight_decay", 3e-5)

    backbone_neck_params = (
        list(model.backbone.parameters()) + list(model.ctc_encoder.parameters())
    )
    ofr_head_params = (
        list(model.ofr.parameters()) + list(model.ctc_fc.parameters())
    )
    if getattr(model, "use_refine_head", False):
        ofr_head_params += list(model.refine_head.parameters())

    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
        learning_rate=base_lr,
        T_max=cfg["epochs"],
    )

    # NOTE: Paddle's per-param-group "learning_rate" is a *scale* of the
    # global `learning_rate=scheduler` value, not an absolute rate — pass
    # ratios here, not absolute lrs (passing base_lr/backbone_lr directly
    # multiplies them by the scheduler again, shrinking effective lr ~1000x).
    optimizer = optim.AdamW(
        parameters=[
            {"params": backbone_neck_params, "learning_rate": backbone_lr_ratio},
            {"params": ofr_head_params, "learning_rate": 1.0},
        ],
        learning_rate=scheduler,
        weight_decay=wd,
    )
    return optimizer, scheduler


def evaluate(model: DFRNet, val_loader, post_process, metric):
    model.eval()
    metric.reset()
    for batch in val_loader:
        images, labels = batch[0], batch[1]
        with paddle.no_grad():
            logits = model(images)
        preds, targets = post_process(logits, label=labels.numpy())
        metric((preds, targets))
    result = metric.get_metric()
    model.train()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dfrnet.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["Model"]
    loss_cfg = cfg["Loss"]
    train_cfg = cfg["Train"]
    eval_cfg = cfg["Eval"]
    save_cfg = cfg["Save"]

    os.makedirs(save_cfg["save_dir"], exist_ok=True)

    # ── Model ──────────────────────────────────────────────────────────
    model = DFRNet(
        backbone_cfg=model_cfg["backbone"],
        svtr_cfg=model_cfg["svtr"],
        num_classes=model_cfg.get("num_classes", 11),
        ofr_nhead=model_cfg.get("ofr_nhead", 4),
        ofr_depth=model_cfg.get("ofr_depth", 2),
        T=model_cfg.get("T", 1000),
        mask_ratio_max=model_cfg.get("mask_ratio_max", 0.5),
        span_len=model_cfg.get("span_len", 3),
        train_t_max=model_cfg.get("train_t_max"),
        use_refine_head=model_cfg.get("use_refine_head", False),
        refine_d_model=model_cfg.get("refine_d_model", 64),
        refine_nhead=model_cfg.get("refine_nhead", 4),
        refine_depth=model_cfg.get("refine_depth", 2),
        pretrained=model_cfg.get("pretrained"),
    )

    # ── Loss ───────────────────────────────────────────────────────────
    criterion = DFRNetLoss(
        lambda_aux=loss_cfg.get("lambda_aux", 0.5),
        beta_rec=loss_cfg.get("beta_rec", 0.1),
        blank_idx=loss_cfg.get("blank_idx", 0),
        lambda_refine=loss_cfg.get("lambda_refine", 0.5),
    )

    # ── Optimizer ──────────────────────────────────────────────────────
    optimizer, scheduler = build_optimizer(model, train_cfg)

    # ── Data ───────────────────────────────────────────────────────────
    # Re-use PaddleOCR's data pipeline for minimal code duplication
    logger = logging.getLogger("dfrnet")
    logging.basicConfig(level=logging.INFO)

    global_cfg = {
        "max_text_length": train_cfg["max_text_length"],
        "character_dict_path": train_cfg.get(
            "character_dict_path", "configs/digits_dict.txt"
        ),
        "use_space_char": False,
    }

    train_dataloader = build_dataloader(
        {
            "Global": global_cfg,
            "Train": {
                "dataset": {
                    "name": "SimpleDataSet",
                    "data_dir": train_cfg["data_dir"],
                    "label_file_list": [train_cfg["label_file"]],
                    "transforms": [
                        {"DecodeImage": {"img_mode": "BGR", "channel_first": False}},
                        {"RecAug": None},
                        {"CTCLabelEncode": None},
                        {"RecResizeImg": {"image_shape": train_cfg["image_shape"]}},
                        {"KeepKeys": {"keep_keys": ["image", "label", "length"]}},
                    ],
                },
                "loader": {
                    "shuffle": True,
                    "batch_size_per_card": train_cfg["batch_size"],
                    "drop_last": True,
                    "num_workers": train_cfg.get("num_workers", 4),
                },
            }
        },
        "Train",
        None,
        logger,
    )
    val_dataloader = build_dataloader(
        {
            "Global": global_cfg,
            "Eval": {
                "dataset": {
                    "name": "SimpleDataSet",
                    "data_dir": eval_cfg["data_dir"],
                    "label_file_list": [eval_cfg["label_file"]],
                    "transforms": [
                        {"DecodeImage": {"img_mode": "BGR", "channel_first": False}},
                        {"CTCLabelEncode": None},
                        {"RecResizeImg": {"image_shape": eval_cfg["image_shape"]}},
                        {"KeepKeys": {"keep_keys": ["image", "label", "length"]}},
                    ],
                },
                "loader": {
                    "shuffle": False,
                    "batch_size_per_card": eval_cfg["batch_size"],
                    "drop_last": False,
                    "num_workers": eval_cfg.get("num_workers", 4),
                },
            }
        },
        "Eval",
        None,
        logger,
    )

    post_process = build_post_process(
        {"name": "CTCLabelDecode", **{k: v for k, v in global_cfg.items() if k != "max_text_length"}}
    )
    metric = build_metric({"name": "RecMetric", "main_indicator": "acc"})

    cutout_cfg = train_cfg.get("image_cutout")
    occ_mix_cfg = train_cfg.get("image_occlusion_mix")

    # ── Training loop ──────────────────────────────────────────────────
    global_step = 0
    for epoch in range(train_cfg["epochs"]):
        model.train()
        for batch in train_dataloader:
            images = batch[0]
            labels = batch[1]

            if cutout_cfg:
                images = random_cutout(
                    images,
                    num_patches=cutout_cfg.get("num_patches", 2),
                    patch_size_frac=cutout_cfg.get("patch_size_frac", 0.15),
                    p=cutout_cfg.get("p", 0.5),
                )
            if occ_mix_cfg:
                images = random_occlusion_mix(
                    images,
                    p=occ_mix_cfg.get("p", 0.5),
                    frac_range=tuple(occ_mix_cfg.get("frac_range", [0.2, 0.5])),
                )

            out = model(images, labels)
            loss_dict = criterion(
                out["logits_main"],
                out["logits_aux"],
                out["F_clean"],
                out["F_hat"],
                labels,
                logits_refined=out.get("logits_refined"),
            )

            loss = loss_dict["loss"]
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            if global_step % 50 == 0:
                msg = (
                    f"[epoch {epoch+1}/{train_cfg['epochs']} step {global_step}] "
                    f"loss={loss.item():.4f} "
                    f"main={loss_dict['loss_main'].item():.4f} "
                    f"aux={loss_dict['loss_aux'].item():.4f} "
                    f"rec={loss_dict['loss_rec'].item():.4f}"
                )
                if "loss_refine" in loss_dict:
                    msg += f" refine={loss_dict['loss_refine'].item():.4f}"
                print(msg)

            if global_step % save_cfg["eval_step"] == 0 and global_step > 0:
                result = evaluate(model, val_dataloader, post_process, metric)
                print(f"  [eval step {global_step}] {result}")

            global_step += 1

        scheduler.step()

        if (epoch + 1) % save_cfg["save_epoch_step"] == 0:
            ckpt_path = os.path.join(save_cfg["save_dir"], f"epoch_{epoch+1}")
            paddle.save(model.state_dict(), ckpt_path + ".pdparams")
            paddle.save(optimizer.state_dict(), ckpt_path + ".pdopt")
            print(f"  Saved checkpoint: {ckpt_path}")

    # final eval
    result = evaluate(model, val_dataloader, post_process, metric)
    print(f"[Final eval] {result}")


if __name__ == "__main__":
    main()
