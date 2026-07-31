
# LPR‑MobileViT

Mobile Vision Transformer based OCR for License Plate Recognition

## 1. Project layout & key files

```
.
├── train.py
├── evaluate.py
├── infer.py
├── lpr_backend_service.py
├── lpr_mobilevit.py              # unified library: models, datasets, augs, utils
├── configs/
│   ├── base.yaml
│   └── *.yaml
└── data/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

- **`lpr_mobilevit.py`** – Single, unified module that contains:
  - backbone: `MobileViTBackbone`
  - heads: `MultiTemplateOCR` (router + per‑template heads)
  - dataset/data utils: `LPDatasetMT`, charset helpers & encoders
  - augmentation: simple letterbox + training jitter
  - decoding helpers (optional char‑level re‑score) and metrics  

- **`train.py`** – Train a multi‑template model (**router + per‑template heads**) based on your YAML. Saves `best.ckpt` and periodic `epoch_*.ckpt`.

- **`evaluate.py`** – Offline evaluation for multi‑template configs. Prints a JSON summary to stdout (format accuracy, char accuracy, plate accuracy) and can optionally dump a CSV of per‑image predictions. Optional robust decode knobs are available for multi‑template models.

- **`infer.py`** – Batch inference on folders or file lists; can write CSV/JSON predictions and optional visual overlays. Uses the same config/ckpt as training.

- **`lpr_backend_service.py`** – A small class, `PlateRecognizer`, for **runtime inference** in services (inputs as base64).

## 2. Installation

Python 3.9+ recommended.

```bash
# PyTorch (choose a build that matches your CUDA if using GPU)
pip install torch torchvision

# Core deps
pip install numpy opencv-python pyyaml pandas tqdm
```

## 3. Data format
### Multi‑template CSV
**Columns**: `path,label[,template_id]` — `template_id` is optional; if omitted the loader will regex‑match the label to a template defined in your YAML.

```csv
path,label,template_id
/path/to/ny/xyz.jpg,ABC1234,ny7
/path/to/ca/xyz.jpg,7ABC123,ca7
```

### Config files (YAML)
Configs drive *both* model shape and data parsing. The code supports optional config inheritance via `inherits:` (child overrides parent keys).

#### Common keys
- **`input_height` / `input_width` / `in_channels`** (or `model.in_h` / `model.in_w`) – resize target and channel count.  
  *Note*: For MobileViT, H and W must be divisible by the patch size (default `2`).
- **`input.mean` / `input.std`** – normalization (defaults to ImageNet stats).
- **`augment`** (train‑time only): `perspective_prob`, `max_warp`, `motion_blur_prob`, `brightness`, `contrast`.
- **`model`**: MobileViT channels/dims/depths/patch/expand; MobileNetV3 uses defaults unless overridden.
- **`multipath`**: `{fmt_hidden, hidden_dim, dropout}` (router MLP and per‑slot head MLP widths).
- **`train`**: `{batch_size, num_workers, epochs, lr, weight_decay, amp, ckpt_every, lambda_fmt, ignore_index, balance_templates}`.
- **`eval`**: `{batch_size, num_workers, amp}`.

#### Multi‑template example
```yaml
# configs/us_multi_example.yaml
separators: ["-", "•", "·", " "]

templates:
  - id: "ny7"
    pattern: "LLL-####"
    charsets:
      - "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      - "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      - "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      - "0123456789"
      - "0123456789"
      - "0123456789"
      - "0123456789"

  - id: "ca7"
    pattern: "### LLL"
    charsets:
      - "0123456789"
      - "0123456789"
      - "0123456789"
      - "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      - "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      - "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
```

> At **runtime** (service), named charsets like `"digits"` / `"letters"` can be expanded automatically, but for **training/eval** you should provide explicit strings as shown above.

## 4. Training

```bash
# Multi‑template, MobileViT (default)
python train.py \
  --train_csv data/train.csv \
  --val_csv   data/val.csv \
  --config    configs/us_multi_example.yaml \
  --out_dir   runs/us_multi_mvit
```

Useful flags:
- `--init_from path/to/ckpt` – **warm‑start** (loads matching tensors, ignores shape mismatches).
- `--freeze_backbone` – train only the OCR heads (helpful when data is small).

**Outputs**: best model is saved to `OUT_DIR/best.ckpt`, and periodic checkpoints to `OUT_DIR/epoch_*.ckpt`.

## 5. Evaluation

```bash
# Multi‑template eval (MobileViT)
python evaluate.py \
  --config configs/us_multi_example.yaml \
  --ckpt   runs/us_multi_mvit/best.ckpt \
  --val_csv data/val.csv \
  --out_csv runs/us_multi_mvit/eval_preds.csv
```

### Robust decode (multi‑template only)
For challenging data, you can enable more robust decoding via optional flags (they are **no‑ops unless provided**):  
`--topk_templates`, `--conf_gate`, `--gap_delta`, `--beam`, `--char_topk`. These are passed into the model’s `decode(...)` and char‑level re‑scoring utilities. fileciteturn0file0

Example:

```bash
python evaluate.py \
  --config configs/us_multi_example.yaml \
  --ckpt   runs/us_multi_mvit/best.ckpt \
  --val_csv data/val.csv \
  --topk_templates 3 --conf_gate 0.98 --gap_delta 0.22 --beam 3 --char_topk 3 \
  --out_csv runs/us_multi_mvit/eval_preds_robust.csv
```

**Summary fields** (printed as JSON):
- `format_acc` – template/router accuracy
- `char_acc_gt` – char accuracy measured under the **ground‑truth** template
- `plate_acc_gt` – full‑plate accuracy measured under the **ground‑truth** template
- `plate_acc_sys` – full‑plate accuracy of the **system prediction** (router + decode)


## 6. Inference

`infer.py` runs inference on images from a directory, explicit paths, or a file list. It can emit CSV/JSON and optional overlays.

```bash
# Directory of images, default backbone (MobileViT)
python infer.py \
  --config configs/us_multi_example.yaml \
  --ckpt   runs/us_multi_mvit/best.ckpt \
  --img_dir /path/to/plates \
  --out_csv runs/us_multi_mvit/preds.csv \
  --viz_dir runs/us_multi_mvit/overlays
```

Outputs include `text_raw` (no separators), `text` (with separators inserted from the template), `confidence`, and (for multi‑template) the chosen `template_id`.

## Backbone details & papers

- **MobileViT**, a light‑weight transformer‑CNN hybrid that learns global representations via a fold‑transformer‑unfold “as convolution” block (Fig. 1b, *page 2*). It typically offers better task‑level generalization than MobileNetV3 at similar parameter budgets.

- **MultiPath ViT OCR**, a per‑character multi‑head (one head per slot) design driven by a **router** for multi‑template scenarios (Fig. 1, *page 2*). This repo implements that idea with MobileViT or MobileNetV3 as the backbone.

## License / Acknowledgements

This project is released under the MIT License. It is provided to encourage research, education, and community collaboration in AI-powered residential security systems.

Code structure and training/eval approach are inspired by:
- **MultiPath ViT OCR** (ICCKE 2022) – see the architecture and results tables for context.
- **MobileViT** (ICLR 2022) – see block diagram and mobile performance notes.
