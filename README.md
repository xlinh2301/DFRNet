# DFRNet: Diffusion Feature Refinement Network for Occluded Scene Text Recognition

DFRNet augments a PPOCRv5 recognition backbone with a training-only
**Occlusion-aware Feature Refinement (OFR)** branch. The goal is to make
the encoder produce features that are robust to partial occlusion, blur,
and other feature-level degradations — without changing the inference
architecture or adding any inference-time latency.

## Idea

Standard CTC training only ever sees clean features. DFRNet additionally
corrupts the encoder output with a **structured, occlusion-aware
diffusion process** (Gaussian noise + contiguous span masking) and asks
a lightweight refinement module to recover a feature usable by the
**same, shared CTC head**. Because the head is shared, the refinement
module cannot learn a shortcut classifier — it must produce features
that live in the same space as the clean ones.

```
Image ──► Backbone ──► Neck/Encoder ──► F (B, L, d)
                                         │
                  ┌──────────────────────┴───────────────────────┐
                  │                                               │
           Shared CTC Head                       OcclusionDiffusionCorruption
                  │                                               │
              L_main                                       F_t (corrupted)
                                                                  │
                                                        OFR Module R_θ(F_t, t)
                                                                  │
                                                             F̂ (recovered)
                                                                  │
                                                 ┌────────────────┴───────────┐
                                            Shared CTC Head            L2 Rec Loss
                                                 │                          │
                                              L_aux                      L_rec
```

Training objective:

```
L = L_main + λ · L_aux + β · L_rec
```

- `L_main` — standard CTC loss on the clean feature
- `L_aux`  — CTC loss on the recovered feature, through the **same** head
- `L_rec`  — reconstruction loss (MSE), focused on the occluded token positions

At **inference**, the OFR branch is not executed:

```
Image → Backbone → Encoder → Shared CTC Head → prediction
```

so DFRNet has **zero additional inference cost** over the base PPOCRv5 model.

## Repository structure

```
DFRNet/
├── dfrnet/
│   ├── corruption.py    # OcclusionDiffusionCorruption — structured occlusion + Gaussian noise
│   ├── ofr_module.py    # OFR Module — timestep-conditioned transformer (AdaLayerNorm)
│   ├── model.py         # DFRNet — wraps PPOCRv5 backbone/encoder + shared CTC head + OFR branch
│   └── loss.py          # DFRNetLoss — L_main + λ·L_aux + β·L_rec
├── train.py              # Training entry point (PaddlePaddle)
├── infer.py              # Inference entry point
├── configs/
│   └── dfrnet.yaml       # Model / training / data configuration
└── tools/
    ├── inspect_checkpoint.py    # Inspect a PPOCRv5 .pdparams checkpoint and preview key remapping
    ├── smoke_test_torch.py      # Pure-PyTorch smoke test of corruption + OFR + loss (no Paddle needed)
    └── ablation_contribution.py # Controlled ablation: Baseline vs DFRNet on synthetic CTC data
```

## Requirements

- PaddlePaddle (for `dfrnet/`, `train.py`, `infer.py`) + PaddleOCR checked out alongside this repo
- PyTorch (only for the standalone tools in `tools/`, no Paddle dependency there)

Expected directory layout:

```
source_code/
├── DFRNet/       (this repo)
└── PaddleOCR/    (https://github.com/PaddlePaddle/PaddleOCR)
release/
└── ppocr_v5_paddle/
    └── best_accuracy.pdparams
```

## Quick start

### 1. Inspect the pretrained checkpoint

No Paddle installation required — pure Python/pickle inspection of key
names and shapes, and a preview of how they remap onto DFRNet's modules:

```bash
python tools/inspect_checkpoint.py --ckpt ../../release/ppocr_v5_paddle/best_accuracy.pdparams
```

### 2. Smoke-test the training loop (no Paddle required)

Validates that the corruption process, OFR module, and shared-head loss
converge on synthetic data:

```bash
python tools/smoke_test_torch.py
```

### 3. Run the contribution ablation

Trains a Baseline (encoder + CTC only) and DFRNet (encoder + CTC + OFR)
side by side on synthetic CTC sequences, then evaluates both under
clean / light-occlusion / heavy-occlusion conditions:

```bash
python tools/ablation_contribution.py
```

### 4. Train on real data (PaddlePaddle)

Edit `configs/dfrnet.yaml` — set `pretrained`, dataset paths, and
`num_classes` for your charset — then:

```bash
python train.py --config configs/dfrnet.yaml
```

### 5. Inference

```bash
python infer.py --config configs/dfrnet.yaml --checkpoint outputs/dfrnet/epoch_100.pdparams --image path/to/img.jpg
```

## Design notes

- **Shared CTC head, no auxiliary classifier.** The recovered feature F̂
  is decoded by the exact same head as the clean feature F. This avoids
  gradient conflict between two independent classifiers and forces OFR
  to produce features the base recognizer can already use.
- **Structured occlusion mask, not just Gaussian noise.** The corruption
  process combines a cosine-scheduled Gaussian diffusion term with a
  contiguous span mask, directly simulating character-level occlusion
  rather than generic additive noise.
- **Masked-position reconstruction loss.** `L_rec` is computed only over
  the occluded token positions, keeping the gradient signal focused on
  where recovery actually matters.
- **Zero inference overhead.** The OFR branch and corruption process are
  training-only; the deployed model is architecturally identical to the
  base PPOCRv5 recognizer.
