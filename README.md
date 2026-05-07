# Acne Classification

Object detection (Part 1) and cross-domain patch classification (Part 2) on ACNE04. ASsumed to being Google Colab for compute.

**Make sure to change directory paths in both notebooks**
---

## Datasets

| Dataset | Used in | Link |
|---------|---------|------|
| **ACNE04**| Part 1 training | [Roboflow Universe — acne04](https://universe.roboflow.com/acne-vulgaris-detection/acne04-detection/) |
| **DermNet** | Part 2 cross-domain evaluation | [Kaggle — shubhamgoel27/dermnet](https://www.kaggle.com/datasets/shubhamgoel27/dermnet) |

---

## Part 1 — Object Detection

**Notebook:** `part1_acne_detection.ipynb`
Models: YOLOv11s and DINO-DETR trained on ACNE04 bounding boxes.

### Setup

Open in Google Colab. Enable GPU runtime first:
> Runtime → Change runtime type → Recommended: **L4** or **A100** → Save

Upload dataset folders to Google Drive:
- `acne04-yolov11/` — YOLO format (images + `.txt` label files + `data.yaml`)
- `acne04-coco/` — COCO format (images + `_annotations.coco.json`)


### Section 1 (Configuration)

All tunable parameters live at the top of **Section 1**:

```python
# ── Quick test vs. full training ──────────────────────────────────────
QUICK_TEST = True    # 3 epochs — fast smoke test
QUICK_TEST = False   # full training: 100 epochs YOLO, 50 epochs DINO

# ── Batch sizes ───────────────────────────────────────────────────────
DINO_BATCH = 2   # L4 (22 GB) or T4 (16 GB)
DINO_BATCH = 4   # V100
DINO_BATCH = 8   # A100 (40 GB)
# YOLO uses YOLO_BATCH = -1 (Ultralytics autobatch — leave as-is)

# ── Data paths ────────────────────────────────────────────────────────
DATA_YOLO = "/content/drive/MyDrive/your-folder/acne04-yolov11"
DATA_COCO = "/content/drive/MyDrive/your-folder/acne04-coco"
```

> DINO-DETR's 4-scale transformer is memory-heavy — reduce `DINO_BATCH` by 1 if you get OOM.

### Compatibility Patches (Section 2)

Section 2 applies three patches automatically — no manual action needed:
- **`ms_deform_attn_cuda.cu`** — fixes PyTorch 2.x API changes in the DINO CUDA kernel
- **`slconfig.py`** — removes a deprecated `verify` argument from `yapf.FormatCode`
- **`main.py`** — registers `argparse.Namespace` as safe for PyTorch 2.6+ checkpoint loading

These are idempotent and safe to re-run after a runtime restart.

---

## Part 2 — Cross-Domain Patch Classification

**Notebook:** `part2_acne_classification.ipynb`
Trains a ResNet-18 binary classifier on ACNE04 patches, then evaluates zero-shot on DermNet.

**Pipeline:**
1. Extract positive (acne bbox) + negative (clear skin) patches from ACNE04
2. Fine-tune ResNet-18 on patches
3. Apply histogram matching for domain adaptation
4. Evaluate on DermNet test set with Grad-CAM visualisation

### Setup

Open in Google Colab. Enable GPU runtime (**A100** recommended):
> Runtime → Change runtime type → **A100** → Save

Upload ACNE04 COCO data to Google Drive. DermNet is downloaded automatically via `kagglehub` on first run and cached — no manual download needed.

### Levers — Section 1 (Configuration)

All tunable parameters live in **Section 1 (`cell-config`)**:

```python
# ── Smoke test ────────────────────────────────────────────────────────
SMOKE = True    # fast sanity check: 20 images, 2 epochs, 2 eval batches
SMOKE = False   # full run

# ── Data paths ────────────────────────────────────────────────────────
DATA_COCO   = "/content/drive/MyDrive/your-folder/acne04-coco"
PATCHES_DIR = "/content/patches"      # ephemeral Colab storage — wiped on each run
OUT_DIR     = "/content/outputs_part2"

# ── Patch extraction ─────────────────────────────────────────────────
PATCH_SIZE    = 64    # px — resize all crops to N×N before saving
PADDING       = 0.1   # fractional padding around each GT bbox (10%)
NEG_PER_IMAGE = 8     # negative (clear skin) patches sampled per image
                      # increase for more balanced training; skin filter may
                      # yield fewer than this if the image has little clear skin

# ── Training ─────────────────────────────────────────────────────────
EPOCHS     = 50    # ReduceLROnPlateau will plateau naturally before this
BATCH_SIZE = 256   # safe for A100; reduce to 64 on T4/L4

# ── Decision threshold ────────────────────────────────────────────────
THRESHOLD = 0.3   # lower than 0.5 to improve acne recall on DermNet
                  # eval cell prints a sweep from 0.20–0.50 to help pick this
```

### Skin Filter (patch extraction)

Negative patches are screened with a YCbCr skin-colour filter so that hair, teeth, and background are rejected. The filter is defined in `cell-extract`:

```python
def is_skin_patch(patch_pil, min_skin_ratio=0.40):
    ...
    # Y  ∈ (40, 220)   — excludes dark hair (low Y) and teeth (high Y)
    # Cr ∈ (133, 177)
    # Cb ∈ (77, 127)
```

Raise `min_skin_ratio` (e.g. `0.55`) for stricter filtering; lower it (e.g. `0.30`) if too few negatives are being saved (check the printed count after extraction).

### Model Architecture

- **Base**: ResNet-18, ImageNet pretrained (`timm`)
- **Head**: `nn.Linear(512, 1)` — binary logit, BCEWithLogitsLoss
- **Frozen layers**: `conv1`, `bn1`, `layer1`, `layer2`
- **Differential learning rates** (AdamW):

```python
layer3  → lr = 1e-5
layer4  → lr = 1e-4
fc head → lr = 1e-3
```

- **Scheduler**: `ReduceLROnPlateau(mode='max', patience=5, factor=0.5)` on val F1
- **Class imbalance**: `pos_weight = n_neg / n_pos` computed dynamically

### Evaluation Output

The eval cell (Section 7) prints:
- Accuracy, F1, AUROC
- Full `classification_report`
- Threshold sweep table (precision / recall / F1 at thresholds 0.20–0.50)
- Confusion matrix saved to `outputs_part2/confusion_matrix.png`

### Grad-CAM (Section 8)

Visualises model attention on 5 acne + 5 non-acne DermNet test samples. Output grid has three rows:

| Row | Content |
|-----|---------|
| Original | Raw DermNet image before any processing |
| Hist-matched | What the model actually sees after domain adaptation |
| Grad-CAM | Attention heatmap — red = high attention, blue = ignored |

Saved to `outputs_part2/gradcam_dermnet.png`.
