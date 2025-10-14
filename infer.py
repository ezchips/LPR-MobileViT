
import os, glob, argparse, json
from typing import List, Tuple
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

from lpr_mobilevit import load_yaml, set_seed
from lpr_mobilevit import build_val_augs
from lpr_mobilevit import MobileViTBackbone
from lpr_mobilevit import MultiPathOCR, MultiTemplateOCR
from lpr_mobilevit import load_yaml, resolve_config_inheritance, load_format_config, load_multi_format_config

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}

def list_images(img_dir=None, images: List[str]=None, list_file=None) -> List[str]:
    paths = []
    if img_dir:
        for ext in IMG_EXTS:
            paths.extend(glob.glob(os.path.join(img_dir, f"*{ext}")))
    if images:
        for p in images:
            if os.path.isdir(p):
                for ext in IMG_EXTS:
                    paths.extend(glob.glob(os.path.join(p, f"*{ext}")))
            elif os.path.splitext(p)[1].lower() in IMG_EXTS:
                paths.append(p)
    if list_file:
        with open(list_file,'r') as f:
            for line in f:
                p = line.strip()
                if p and os.path.splitext(p)[1].lower() in IMG_EXTS:
                    paths.append(p)
    paths = sorted(list({os.path.abspath(p) for p in paths}))
    if not paths:
        raise SystemExit("No images found. Use --img_dir or --images or --list_file.")
    return paths

def format_with_pattern(raw: str, pattern: str) -> str:
    if not pattern:
        return raw
    out = []
    i = 0
    for ch in pattern:
        if ch in ("L", "#"):
            if i < len(raw):
                out.append(raw[i]); i += 1
        else:
            out.append(ch)
    return "".join(out)

class InferenceDataset(Dataset):
    def __init__(self, paths: List[str], tfms):
        self.paths = paths
        self.tfms = tfms
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = self.tfms(image=img)
        return out['image'], p

def build_backbone(cfg, in_ch=3):
    return MobileViTBackbone(
        in_ch=in_ch,
        channels=tuple(cfg.get('channels',[48,64,80])),
        dims=tuple(cfg.get('dims',[96,120,144])),
        depths=tuple(cfg.get('depths',[2,4,3])),
        patch=cfg.get('patch',2),
        expand=cfg.get('expand',4),
    )

def load_single_model(config_path, device, in_ch):
    fmt = load_format_config(config_path)
    backbone = build_backbone(load_yaml(config_path).get('model', {}), in_ch)
    num_classes = [len(cs) for cs in fmt.charsets]
    model = MultiPathOCR(backbone, num_classes).to(device)
    return model, fmt

def load_multi_model(config_path, device, in_ch):
    mt_cfg = load_multi_format_config(config_path)
    num_classes_per_template = [[len(cs) for cs in t.charsets] for t in mt_cfg.templates]
    backbone = build_backbone(load_yaml(config_path).get('model', {}), in_ch)
    model = MultiTemplateOCR(backbone, num_classes_per_template,
                             fmt_hidden=256, head_hidden=512, head_dropout=0.1).to(device)
    return model, mt_cfg

def decode_single(logits: List[torch.Tensor], charsets: List[str]):
    probs = [l.softmax(dim=1) for l in logits]
    preds = [p.argmax(dim=1) for p in probs]
    B = preds[0].shape[0]
    texts, confs = [], []
    for i in range(B):
        s=[]; csum=0.0
        for pos,(p,idx) in enumerate(zip(probs,preds)):
            ci = int(idx[i].item())
            s.append(charsets[pos][ci])
            csum += float(p[i,ci].item())
        texts.append("".join(s))
        confs.append(csum/len(charsets))
    return texts, confs

def draw_overlay(img_rgb, top_left_text: str, bottom_left_text: str, ok_color=(0,180,0)):
    img = img_rgb.copy()
    h, w = img.shape[:2]
    pad = 6
    bar_h = 48
    cv2.rectangle(img, (0,0), (w, bar_h), (255,255,255), -1)
    cv2.putText(img, top_left_text, (pad, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(img, bottom_left_text, (pad, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ok_color, 2, cv2.LINE_AA)
    return img

def _resolve_io(cfg: dict):
    """
    (io_h, io_w, in_ch) with precedence:
    model.in_h/in_w -> input_height/input_width -> input.height/width -> defaults
    """
    in_ch = int(cfg.get("in_channels", 3))
    m = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    io_h = m.get("in_h", None)
    io_w = m.get("in_w", None)
    if io_h is None or io_w is None:
        io_h = cfg.get("input_height", io_h)
        io_w = cfg.get("input_width",  io_w)
    if (io_h is None or io_w is None) and isinstance(cfg.get("input"), dict):
        io_h = cfg["input"].get("height", io_h)
        io_w = cfg["input"].get("width",  io_w)
    io_h = int(io_h if io_h is not None else 128)
    io_w = int(io_w if io_w is not None else 256)
    return io_h, io_w, in_ch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--img_dir", type=str)
    grp.add_argument("--images", nargs="+")
    grp.add_argument("--list_file", type=str)
    ap.add_argument("--out_csv", type=str, default="preds.csv")
    ap.add_argument("--out_json", type=str, default=None)
    ap.add_argument("--viz_dir", type=str, default=None)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--seed", type=int, default=123)
    # Optional decode knobs (all None by default: no behavior change)
    ap.add_argument("--topk_templates", type=int, default=None, help="Number of template candidates to consider (MultiTemplate decode).")
    ap.add_argument("--conf_gate", type=float, default=None, help="If provided, use char-level re-scoring when below this confidence.")
    ap.add_argument("--gap_delta", type=float, default=None, help="Confidence gap threshold for re-scoring (e.g., 0.20).")
    ap.add_argument("--beam", type=int, default=None, help="Beam width for char-level re-scoring.")
    ap.add_argument("--char_topk", type=int, default=None, help="Top-K per-position chars for re-scoring.")
    args = ap.parse_args()

    set_seed(args.seed)

    cfg_all = load_yaml(args.config)
    cfg = cfg_all
    if 'inherits' in cfg_all:
        import yaml
        base_path = os.path.join(os.path.dirname(args.config), cfg_all['inherits'])
        with open(base_path,'r') as f:
            parent = yaml.safe_load(f)
        for k,v in cfg_all.items():
            if k in ("format","templates"): continue
            parent[k] = v
        cfg = parent

    io_h, io_w, in_ch = _resolve_io(cfg)

    paths = list_images(args.img_dir, args.images, args.list_file)
    tfms = build_val_augs(io_h, io_w, cfg)
    ds = InferenceDataset(paths, tfms)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    cfg_merged = resolve_config_inheritance(args.config)
    is_multi = ("templates" in cfg_merged)
    if is_multi:
        model, mt_cfg = load_multi_model(args.config, device, in_ch)
    else:
        model, fmt = load_single_model(args.config, device, in_ch)

    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    rows = []
    if args.viz_dir:
        os.makedirs(args.viz_dir, exist_ok=True)

    with torch.no_grad():
        for imgs, batch_paths in dl:
            imgs = imgs.to(device).float()
            if is_multi:
                fmt_logits, tpl_logits = model(imgs)
                # Build decode kwargs (filter by model.decode signature to stay compatible)
                import inspect
                dec_params = set(inspect.signature(model.decode).parameters.keys())
                _dk = {}
                if args.topk_templates is not None and 'topk_templates' in dec_params: _dk['topk_templates'] = args.topk_templates
                if args.conf_gate is not None and 'conf_gate' in dec_params: _dk['conf_gate'] = args.conf_gate
                if args.gap_delta is not None and 'gap_delta' in dec_params: _dk['gap_delta'] = args.gap_delta
                if args.beam is not None and 'beam' in dec_params: _dk['beam'] = args.beam
                if args.char_topk is not None and 'char_topk' in dec_params: _dk['char_topk'] = args.char_topk
                texts, confs, tids = model.decode(
                    (fmt_logits, tpl_logits),
                    charsets_per_template=[t.charsets for t in mt_cfg.templates],
                    **_dk
                )
                for p, raw, conf, tid in zip(batch_paths, texts, confs, tids):
                    tpl = mt_cfg.templates[int(tid)]
                    # insert separators
                    formatted = []
                    i = 0
                    for ch in tpl.pattern:
                        if ch in ("L","#"):
                            if i < len(raw): formatted.append(raw[i]); i+=1
                        else:
                            formatted.append(ch)
                    formatted = "".join(formatted)
                    rows.append({
                        "path": p,
                        "template_id": tpl.id,
                        "template_pattern": tpl.pattern,
                        "text_raw": raw,
                        "text": formatted,
                        "confidence": float(conf),
                    })
                    if args.viz_dir:
                        img0 = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
                        top = f"T:{tpl.id}  {tpl.pattern or ''}"
                        bot = f"{formatted}  ({conf:.2f})"
                        out = draw_overlay(img0, top, bot)
                        cv2.imwrite(os.path.join(args.viz_dir, os.path.basename(p)), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
            else:
                logits = model(imgs)
                text_raw, confs = decode_single(logits, fmt.charsets)
                for p, raw, conf in zip(batch_paths, text_raw, confs):
                    formatted = []
                    i = 0
                    for ch in fmt.pattern:
                        if ch in ("L","#"):
                            if i < len(raw): formatted.append(raw[i]); i+=1
                        else:
                            formatted.append(ch)
                    formatted = "".join(formatted)
                    rows.append({
                        "path": p,
                        "text_raw": raw,
                        "text": formatted,
                        "confidence": float(conf),
                    })
                    if args.viz_dir:
                        img0 = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
                        top = f"{fmt.pattern or ''}"
                        bot = f"{formatted}  ({conf:.2f})"
                        out = draw_overlay(img0, top, bot)
                        cv2.imwrite(os.path.join(args.viz_dir, os.path.basename(p)), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(rows, f, indent=2)
    print(f"Wrote {len(rows)} predictions to {args.out_csv}" + (f" and {args.out_json}" if args.out_json else ""))
    if args.viz_dir:
        print(f"Wrote overlays to {args.viz_dir}")

if __name__ == "__main__":
    main()