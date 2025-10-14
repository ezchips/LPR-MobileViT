
import os, argparse, json, collections
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict, Optional
try:
    from tqdm import tqdm
except:
    tqdm = lambda x, **k: x

import cv2

from lpr_mobilevit import load_yaml, set_seed
from lpr_mobilevit import build_val_augs
from lpr_mobilevit import MobileViTBackbone
from lpr_mobilevit import MultiPathOCR, MultiTemplateOCR
from lpr_mobilevit import LPDataset, LPDatasetMT, collate_mt
from lpr_mobilevit import load_yaml, resolve_config_inheritance, load_format_config, load_multi_format_config

def _build_backbone(cfg, in_ch=3):
    return MobileViTBackbone(
        in_ch=in_ch,
        channels=tuple(cfg.get('channels', [48,64,80])),
        dims=tuple(cfg.get('dims', [96,120,144])),
        depths=tuple(cfg.get('depths', [2,4,3])),
        patch=cfg.get('patch', 2),
        expand=cfg.get('expand', 4),
    )

def _argmax_per_pos(logits: List[torch.Tensor]) -> torch.Tensor:
    preds = [l.argmax(dim=1) for l in logits]
    return torch.stack(preds, dim=1)

def _decode_indices(indices: List[int], charsets: List[str]) -> str:
    s = []
    for i, idx in enumerate(indices):
        s.append(charsets[i][int(idx)])
    return ''.join(s)

def _overlay_text(img_rgb, gt, pred, ok, save_path):
    img = img_rgb.copy()
    h, w = img.shape[:2]
    cv2.rectangle(img, (0,0), (w, 32), (255,255,255), -1)
    color = (0,128,0) if ok else (0,0,255)
    cv2.putText(img, f"GT: {gt}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(img, f"PR: {pred}", (w//2, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def evaluate_single(model, loader, device, fmt, args):
    model.eval()
    char_total = 0
    char_correct = 0
    plate_total = 0
    plate_correct = 0
    rows = []
    err_saved = 0
    for it, (imgs, ys, paths) in enumerate(loader):
        imgs = imgs.to(device).float()
        ys = ys.to(device)
        logits = model(imgs)
        preds = _argmax_per_pos(logits)
        correct = (preds == ys)
        char_correct += correct.sum().item()
        char_total += correct.numel()
        ok_plate = correct.all(dim=1)
        plate_correct += ok_plate.sum().item()
        plate_total += imgs.size(0)
        preds_cpu = preds.cpu().numpy()
        ys_cpu = ys.cpu().numpy()
        for i in range(imgs.size(0)):
            pred_text = _decode_indices(list(preds_cpu[i]), fmt.charsets)
            gt_text   = _decode_indices(list(ys_cpu[i]), fmt.charsets)
            rows.append({
                "path": paths[i],
                "gt": gt_text,
                "pred": pred_text,
                "plate_ok": bool(ok_plate[i].item())
            })
    import pandas as pd
    df = pd.DataFrame(rows)
    summary = {
        "mode": "single",
        "char_acc": round(char_correct / max(1,char_total), 6),
        "plate_acc": round(plate_correct / max(1,plate_total), 6),
        "samples": plate_total
    }
    return {"summary": summary, "df": df}

def _decode_system_one(i, tid, tpl_logits, mt_cfg):
    positions = len(tpl_logits[tid])
    s = []
    conf_sum = 0.0
    for pos in range(positions):
        p = tpl_logits[tid][pos][i].softmax(dim=0)  # [Ci]
        conf, idx = p.max(dim=0)
        s.append(mt_cfg.templates[tid].charsets[pos][int(idx.item())])
        conf_sum += float(conf.item())
    return ''.join(s), conf_sum/positions if positions>0 else 0.0

def _decode_ground_truth_one(i, y_pad, mt_cfg, tid_gt):
    s = []
    Nt = mt_cfg.templates[tid_gt].positions
    for pos in range(Nt):
        idx = int(y_pad[i, pos].item())
        s.append(mt_cfg.templates[tid_gt].charsets[pos][idx])
    return ''.join(s)

def evaluate_multi(model, loader, device, mt_cfg, args):
    model.eval()
    T = len(mt_cfg.templates)
    Pmax = max(t.positions for t in mt_cfg.templates)
    ignore_index = -100

    fmt_correct = 0; n_samples = 0
    char_correct = 0; char_total = 0
    plate_ok_sys = 0; plate_ok_gt = 0
    tpl_confusion = np.zeros((T, T), dtype=np.int64)
    char_conf = collections.Counter()
    rows = []

    for it, batch in enumerate(loader):
        imgs, y_pad, t_idx, mask, paths = batch
        B = imgs.size(0)
        n_samples += B
        imgs = imgs.to(device).float()
        y_pad = y_pad.to(device)
        t_idx = t_idx.to(device)
        mask  = mask.to(device)
        fmt_logits, tpl_logits = model(imgs)
        fmt_pred = fmt_logits.argmax(dim=1)
        fmt_correct += (fmt_pred == t_idx).sum().item()
        for a, b in zip(t_idx.cpu().tolist(), fmt_pred.cpu().tolist()):
            tpl_confusion[a, b] += 1
        for i in range(B):
            tid_gt = int(t_idx[i].item())
            tid_pr = int(fmt_pred[i].item())
            Nt = len(tpl_logits[tid_gt])
            ok_gt = True
            for pos in range(Nt):
                gt = int(y_pad[i, pos].item())
                if gt == ignore_index: continue
                pred_idx = int(tpl_logits[tid_gt][pos][i].argmax(dim=0).item())
                char_total += 1
                if pred_idx == gt:
                    char_correct += 1
                else:
                    ok_gt = False
                    ch_gt = mt_cfg.templates[tid_gt].charsets[pos][gt]
                    ch_pr = mt_cfg.templates[tid_gt].charsets[pos][pred_idx]
                    char_conf[(ch_gt, ch_pr)] += 1
            plate_ok_gt += int(ok_gt)
            txt_pr, conf_pr = _decode_system_one(i, tid_pr, tpl_logits, mt_cfg)
            txt_gt = _decode_ground_truth_one(i, y_pad, mt_cfg, tid_gt)
            plate_ok_sys += int(txt_pr == txt_gt)
            rows.append({
                "path": paths[i],
                "template_gt": mt_cfg.templates[tid_gt].pattern or mt_cfg.templates[tid_gt].id,
                "template_pr": mt_cfg.templates[tid_pr].pattern or mt_cfg.templates[tid_pr].id,
                "format_ok": bool(tid_gt == tid_pr),
                "gt": txt_gt,
                "pred": txt_pr,
                "plate_ok_sys": bool(txt_pr == txt_gt),
                "plate_ok_gt": bool(ok_gt),
                "pred_conf": float(conf_pr),
            })

    summary = {
        "mode": "multi",
        "samples": n_samples,
        "format_acc": round(fmt_correct / max(1, n_samples), 6),
        "char_acc_gt": round(char_correct / max(1, char_total), 6),
        "plate_acc_gt": round(plate_ok_gt / max(1, n_samples), 6),
        "plate_acc_sys": round(plate_ok_sys / max(1, n_samples), 6),
    }
    import pandas as pd
    df = pd.DataFrame(rows)
    return {"summary": summary, "df": df, "tpl_confusion": tpl_confusion}


def evaluate_multi_decode(model, loader, device, mt_cfg, args):
    """
    Evaluate using model.decode with optional kwargs (topk_templates, conf_gate, gap_delta, beam, char_topk).
    Falls back to simple argmax if model.decode is unavailable.
    """
    import inspect, numpy as np, collections
    model.eval()
    T = len(mt_cfg.templates)
    ignore_index = -100

    fmt_correct = 0; n_samples = 0
    char_correct = 0; char_total = 0
    plate_ok_sys = 0; plate_ok_gt = 0
    tpl_confusion = np.zeros((T, T), dtype=np.int64)
    rows = []

    def idxs_to_str(tid, y_row):
        # convert y indices to text using template charsets
        chars = []
        cs = mt_cfg.templates[tid].charsets
        for pos in range(len(cs)):
            gt = int(y_row[pos].item())
            if gt == ignore_index: continue
            vocab = cs[pos]
            try:
                chars.append(vocab[gt])
            except Exception:
                pass
        return "".join(chars)

    with torch.no_grad():
        for batch in loader:
            imgs, y_pad, t_idx, mask, paths = batch
            imgs = imgs.to(device).float()
            fmt_logits, tpl_logits = model(imgs)
            # Build kwargs filtered by signature
            dkw = {}
            try:
                params = set(inspect.signature(model.decode).parameters.keys())
                if args.topk_templates is not None and 'topk_templates' in params: dkw['topk_templates'] = args.topk_templates
                if args.conf_gate is not None and 'conf_gate' in params: dkw['conf_gate'] = args.conf_gate
                if args.gap_delta is not None and 'gap_delta' in params: dkw['gap_delta'] = args.gap_delta
                if args.beam is not None and 'beam' in params: dkw['beam'] = args.beam
                if args.char_topk is not None and 'char_topk' in params: dkw['char_topk'] = args.char_topk

                texts, confs, tids = model.decode(
                    (fmt_logits, tpl_logits),
                    charsets_per_template=[t.charsets for t in mt_cfg.templates],
                    **dkw
                )
            except Exception:
                # Fallback: simple argmax template + per-position
                probs = torch.softmax(fmt_logits, dim=1)
                tids = probs.argmax(dim=1).tolist()
                confs = probs.max(dim=1).values.tolist()
                texts = []
                for i in range(imgs.size(0)):
                    tid = int(tids[i])
                    cs = mt_cfg.templates[tid].charsets
                    s = []
                    for pos in range(len(cs)):
                        logit = tpl_logits[tid][pos][i]  # (B,C) -> pick i-th -> (C,)
                        idx = int(logit.argmax(dim=0).item())
                        try:
                            s.append(cs[pos][idx])
                        except Exception:
                            pass
                    texts.append("".join(s))

            B = imgs.size(0)
            n_samples += B
            for i in range(B):
                tid_gt = int(t_idx[i].item())
                tid_pr = int(tids[i])
                txt_pr = texts[i]
                txt_gt = idxs_to_str(tid_gt, y_pad[i])

                fmt_correct += int(tid_pr == tid_gt)
                tpl_confusion[tid_gt, tid_pr] += 1

                # plate metrics
                ok_sys = (txt_pr == txt_gt)
                plate_ok_sys += int(ok_sys)

                # char metrics (under GT template)
                Nt = len(mt_cfg.templates[tid_gt].charsets)
                ok_gt = True
                for pos in range(Nt):
                    gt_idx = int(y_pad[i, pos].item())
                    if gt_idx == ignore_index: continue
                    # argmax per-position under GT template
                    pred_idx = int(tpl_logits[tid_gt][pos][i].argmax(dim=0).item())
                    char_total += 1
                    if pred_idx == gt_idx:
                        char_correct += 1
                    else:
                        ok_gt = False

                plate_ok_gt += int(ok_gt)

                rows.append({
                    "path": paths[i],
                    "template_gt": mt_cfg.templates[tid_gt].pattern or mt_cfg.templates[tid_gt].id,
                    "template_pr": mt_cfg.templates[tid_pr].pattern or mt_cfg.templates[tid_pr].id,
                    "format_ok": bool(tid_gt == tid_pr),
                    "gt": txt_gt,
                    "pred": txt_pr,
                    "plate_ok_sys": bool(ok_sys),
                    "plate_ok_gt": bool(ok_gt),
                    "pred_conf": float(confs[i]) if isinstance(confs, (list, tuple)) else float(confs[i].item()) if hasattr(confs, 'shape') else 0.0,
                })

    import pandas as pd
    df = pd.DataFrame(rows)
    summary = {
        "mode": "multi/decode",
        "samples": n_samples,
        "format_acc": round(fmt_correct / max(1, n_samples), 6),
        "char_acc_gt": round(char_correct / max(1, char_total), 6),
        "plate_acc_gt": round(plate_ok_gt / max(1, n_samples), 6),
        "plate_acc_sys": round(plate_ok_sys / max(1, n_samples), 6),
    }
    return {"summary": summary, "df": df, "tpl_confusion": tpl_confusion}

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
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--out_dir", default="eval_out")
    ap.add_argument("--seed", type=int, default=123)
    # Optional decode knobs (all None by default)
    ap.add_argument("--topk_templates", type=int, default=None)
    ap.add_argument("--conf_gate", type=float, default=None)
    ap.add_argument("--gap_delta", type=float, default=None)
    ap.add_argument("--beam", type=int, default=None)
    ap.add_argument("--char_topk", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    cfg_all = load_yaml(args.config)
    if 'inherits' in cfg_all:
        import yaml
        base_path = os.path.join(os.path.dirname(args.config), cfg_all['inherits'])
        with open(base_path, 'r') as f:
            base_cfg = yaml.safe_load(f)
        for k, v in cfg_all.items():
            if k in ('format','templates'): continue
            base_cfg[k] = v
        cfg = base_cfg
    else:
        cfg = cfg_all

    io_h, io_w, in_ch = _resolve_io(cfg)
    model_cfg = cfg.get('model', {})
    eval_cfg = cfg.get('eval', {})
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tfms = build_val_augs(io_h, io_w, cfg)

    cfg_merged = resolve_config_inheritance(args.config)
    is_multi = ("templates" in cfg_merged)
    if is_multi:
        mt_cfg = load_multi_format_config(args.config)
        ds = LPDatasetMT(args.val_csv, mt_cfg, transform=tfms, template_col="template_id",
                         drop_unmatched=True, uppercase=True, ignore_index=-100)
        loader = DataLoader(ds, batch_size=eval_cfg.get('batch_size', 256),
                            shuffle=False, num_workers=eval_cfg.get('num_workers', 8),
                            pin_memory=True, collate_fn=collate_mt)
        from lpr_mobilevit import MultiTemplateOCR
        num_classes_per_template = [[len(cs) for cs in t.charsets] for t in mt_cfg.templates]
        backbone = _build_backbone(model_cfg, in_ch)
        model = MultiTemplateOCR(backbone, num_classes_per_template).to(device)
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
        use_decode = any([args.topk_templates is not None, args.conf_gate is not None, args.gap_delta is not None, args.beam is not None, args.char_topk is not None])
        if use_decode:
            results = evaluate_multi_decode(model, loader, device, mt_cfg, args)
        else:
            results = evaluate_multi(model, loader, device, mt_cfg, args)
    else:
        fmt = load_format_config(args.config)
        ds = LPDataset(args.val_csv, fmt, transform=tfms, drop_invalid=True, uppercase=True)
        loader = DataLoader(ds, batch_size=eval_cfg.get('batch_size', 256),
                            shuffle=False, num_workers=eval_cfg.get('num_workers', 8),
                            pin_memory=True)
        backbone = _build_backbone(model_cfg, in_ch)
        num_classes = [len(cs) for cs in fmt.charsets]
        model = MultiPathOCR(backbone, num_classes).to(device)
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
        results = evaluate_single(model, loader, device, fmt, args)

    print(json.dumps(results['summary'], indent=2))
    if args.out_csv:
        results['df'].to_csv(args.out_csv, index=False)

if __name__ == "__main__":
    main()