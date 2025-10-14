
import os, argparse, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from lpr_mobilevit import load_yaml, set_seed
from lpr_mobilevit import build_train_augs, build_val_augs
from lpr_mobilevit import MobileViTBackbone
from lpr_mobilevit import MultiPathOCR, MultiTemplateOCR
from lpr_mobilevit import accuracy_metrics
from lpr_mobilevit import LPDataset, LPDatasetMT, collate_mt
from lpr_mobilevit import load_yaml, resolve_config_inheritance, load_format_config, load_multi_format_config

def build_backbone(cfg_model, in_channels):
    return MobileViTBackbone(
        in_ch=in_channels,
        channels=tuple(cfg_model.get('channels',[48,64,80])),
        dims=tuple(cfg_model.get('dims',[96,120,144])),
        depths=tuple(cfg_model.get('depths',[2,4,3])),
        patch=cfg_model.get('patch',2),
        expand=cfg_model.get('expand',4),
    )

def build_single(fmt_cfg, base_cfg):
    num_classes = [len(cs) for cs in fmt_cfg.charsets]
    backbone = build_backbone(base_cfg.get('model',{}), base_cfg.get('in_channels',3))
    model = MultiPathOCR(backbone, num_classes,
                         hidden=base_cfg.get('multipath',{}).get('hidden_dim',512),
                         dropout=base_cfg.get('multipath',{}).get('dropout',0.1))
    return model, num_classes

def build_multi(mt_cfg, base_cfg):
    num_classes_per_template = [[len(cs) for cs in t.charsets] for t in mt_cfg.templates]
    backbone = build_backbone(base_cfg.get('model',{}), base_cfg.get('in_channels',3))
    model = MultiTemplateOCR(
        backbone,
        num_classes_per_template,
        fmt_hidden=base_cfg.get('multipath',{}).get('fmt_hidden',256),
        head_hidden=base_cfg.get('multipath',{}).get('hidden_dim',512),
        head_dropout=base_cfg.get('multipath',{}).get('dropout',0.1),
    )
    return model, num_classes_per_template

def train_epoch_single(model, loader, device, scaler, ce_losses, optimizer):
    model.train()
    total_loss=0.0; seen=0
    for imgs, ys, _ in loader:
        imgs = imgs.to(device).float()
        ys   = ys.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            logits = model(imgs)
            loss = 0.0
            for pos, logit in enumerate(logits):
                loss = loss + ce_losses[pos](logit, ys[:,pos])
        if scaler is not None:
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()
        b = imgs.size(0); total_loss += loss.item()*b; seen += b
    return total_loss/seen

@torch.no_grad()
def eval_single(model, loader, device):
    model.eval()
    tot_char=0.0; tot_plate=0.0; n=0
    for imgs, ys, _ in loader:
        imgs = imgs.to(device).float()
        ys = ys.to(device)
        logits = model(imgs)
        cacc, pacc = accuracy_metrics(logits, ys)
        b = imgs.size(0)
        tot_char += cacc*b; tot_plate += pacc*b; n += b
    return tot_char/n, tot_plate/n

def train_epoch_multi(model, loader, device, scaler, optimizer, lambda_fmt, ignore_index):
    model.train()
    total_loss=0.0; seen=0
    for imgs, y_pad, t_idx, mask, _ in loader:
        imgs  = imgs.to(device).float()
        y_pad = y_pad.to(device)
        t_idx = t_idx.to(device)
        mask  = mask.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            outputs = model(imgs)
            loss = model.loss_multi(outputs, y_pad, t_idx, mask,
                                    lambda_fmt=lambda_fmt, ignore_index=ignore_index,
                                    class_weights=None)
        if scaler is not None:
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()
        b = imgs.size(0); total_loss += loss.item()*b; seen += b
    return total_loss/seen

@torch.no_grad()
def eval_multi(model, loader, device, ignore_index=-100):
    model.eval()
    char_correct=0; total_chars=0; plate_correct=0; n=0; fmt_correct=0
    for imgs, y_pad, t_idx, mask, _ in loader:
        imgs  = imgs.to(device).float()
        y_pad = y_pad.to(device)
        t_idx = t_idx.to(device)
        mask  = mask.to(device)

        fmt_logits, tpl_logits = model(imgs)
        fmt_pred = fmt_logits.argmax(dim=1)
        fmt_correct += (fmt_pred == t_idx).sum().item()

        B = imgs.size(0)
        for i in range(B):
            t = int(t_idx[i].item())
            ok = True
            Nt = len(tpl_logits[t])
            for pos in range(Nt):
                gt = int(y_pad[i, pos].item())
                if gt == ignore_index:
                    continue
                pred = int(tpl_logits[t][pos][i].argmax(dim=0).item())
                total_chars += 1
                if pred == gt:
                    char_correct += 1
                else:
                    ok = False
            plate_correct += int(ok)
        n += B
    char_acc  = (char_correct/total_chars) if total_chars else 0.0
    plate_acc = (plate_correct/n) if n else 0.0
    fmt_acc   = fmt_correct/n if n else 0.0
    return char_acc, plate_acc, fmt_acc

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
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", default="runs/exp1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--init_from", type=str, default="", help="Warm-start from checkpoint (safe partial load).")
    ap.add_argument("--freeze_backbone", action="store_true", help="Freeze backbone parameters during training.")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    all_cfg = load_yaml(args.config)
    if 'inherits' in all_cfg:
        import yaml
        base_path = os.path.join(os.path.dirname(args.config), all_cfg['inherits'])
        with open(base_path,'r') as f:
            base_cfg = yaml.safe_load(f)
        for k,v in all_cfg.items():
            if k == 'format' or k == 'templates': continue
            base_cfg[k] = v
        cfg = base_cfg
    else:
        cfg = all_cfg

    io_h, io_w, in_channels = _resolve_io(cfg)
    train_cfg = cfg.get('train', {})
    eval_cfg  = cfg.get('eval', {})
    model_cfg = cfg.get('model', {})
    mp_cfg    = cfg.get('multipath', {})

    # --- inside main(), BEFORE any call to load_format_config ---
    cfg_merged = resolve_config_inheritance(args.config)
    is_multi = ("templates" in cfg_merged)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if is_multi:
        mt_cfg = load_multi_format_config(args.config)
        model, num_classes_per_template = build_multi(mt_cfg, cfg)
        train_tfms = build_train_augs(io_h, io_w, cfg)
        val_tfms   = build_val_augs(io_h, io_w, cfg)
        ds_train = LPDatasetMT(args.train_csv, mt_cfg, transform=train_tfms,
                               template_col="template_id", drop_unmatched=True,
                               uppercase=True, ignore_index=train_cfg.get('ignore_index', -100))
        ds_val   = LPDatasetMT(args.val_csv, mt_cfg, transform=val_tfms,
                               template_col="template_id", drop_unmatched=True,
                               uppercase=True, ignore_index=train_cfg.get('ignore_index', -100))
        if train_cfg.get('balance_templates', True):
            t_indices = [t for (_, _, t) in ds_train.samples]
            import numpy as np
            counts = np.bincount(t_indices, minlength=len(mt_cfg.templates)).astype(np.float32)
            inv = 1.0 / np.maximum(counts, 1.0)
            weights = np.array([inv[t] for t in t_indices], dtype=np.float32)
            sampler = WeightedRandomSampler(torch.from_numpy(weights), num_samples=len(weights), replacement=True)
            dl_train = DataLoader(ds_train, batch_size=train_cfg.get('batch_size',128),
                                  sampler=sampler, num_workers=train_cfg.get('num_workers',8),
                                  pin_memory=True, collate_fn=collate_mt)
        else:
            dl_train = DataLoader(ds_train, batch_size=train_cfg.get('batch_size',128),
                                  shuffle=True, num_workers=train_cfg.get('num_workers',8),
                                  pin_memory=True, collate_fn=collate_mt)
        dl_val   = DataLoader(ds_val,   batch_size=eval_cfg.get('batch_size',256),
                              shuffle=False, num_workers=eval_cfg.get('num_workers',8),
                              pin_memory=True, collate_fn=collate_mt)
    else:
        fmt_cfg = load_format_config(args.config)
        model, num_classes = build_single(fmt_cfg, cfg)
        train_tfms = build_train_augs(io_h, io_w, cfg)
        val_tfms   = build_val_augs(io_h, io_w, cfg)
        ds_train = LPDataset(args.train_csv, fmt_cfg, transform=train_tfms, drop_invalid=True, uppercase=True)
        ds_val   = LPDataset(args.val_csv,   fmt_cfg, transform=val_tfms,   drop_invalid=True, uppercase=True)
        dl_train = DataLoader(ds_train, batch_size=train_cfg.get('batch_size',128),
                              shuffle=True, num_workers=train_cfg.get('num_workers',8), pin_memory=True)
        dl_val   = DataLoader(ds_val,   batch_size=eval_cfg.get('batch_size',256),
                              shuffle=False, num_workers=eval_cfg.get('num_workers',8), pin_memory=True)

    model.to(device)
    # --- Warm-start (safe partial load) ---
    if getattr(args, 'init_from', ''):
        _sd = torch.load(args.init_from, map_location='cpu')
        _sd = _sd.get('model', _sd) if isinstance(_sd, dict) else _sd
        model_sd = model.state_dict()
        filtered = {}
        dropped = []
        for k, v in _sd.items():
            if k in model_sd and hasattr(v, 'shape') and hasattr(model_sd[k], 'shape') and tuple(v.shape) == tuple(model_sd[k].shape):
                filtered[k] = v
            elif k in model_sd:
                try:
                    old_shape = tuple(v.shape) if hasattr(v, 'shape') else None
                    new_shape = tuple(model_sd[k].shape) if hasattr(model_sd[k], 'shape') else None
                except Exception:
                    old_shape, new_shape = None, None
                dropped.append((k, old_shape, new_shape))
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        print(f"[init_from] loaded {len(filtered)} tensors; dropped {len(dropped)} due to shape mismatch; missing={len(missing)} unexpected={len(unexpected)}")
        for _k, _os, _ns in dropped[:10]:
            print(f"  [drop] {_k}: ckpt{_os} -> model{_ns}")

    # --- Optional freeze backbone ---
    freeze_flag = bool(args.freeze_backbone) or bool(cfg.get('train', {}).get('freeze_backbone', False))
    if freeze_flag and hasattr(model, 'backbone'):
        for p in model.backbone.parameters():
            p.requires_grad = False
        print('[freeze_backbone] Backbone parameters frozen')
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.get('lr',5e-4),
                                  weight_decay=train_cfg.get('weight_decay',0.01))
    scaler = torch.amp.GradScaler('cuda', enabled=train_cfg.get('amp', True) and torch.cuda.is_available())

    if is_multi:
        lambda_fmt = float(train_cfg.get('lambda_fmt', 0.20))
        ignore_index = int(train_cfg.get('ignore_index', -100))
    else:
        ce_losses = [nn.CrossEntropyLoss() for _ in (num_classes if not is_multi else [])]

    best_plate = 0.0
    epochs = train_cfg.get('epochs', 150)

    for epoch in range(1, epochs+1):
        tic = time.time()
        if is_multi:
            tr_loss = train_epoch_multi(model, dl_train, device, scaler, optimizer, lambda_fmt, ignore_index)
            char_acc, plate_acc, fmt_acc = eval_multi(model, dl_val, device, ignore_index=ignore_index)
            print(f"[{epoch:03d}] loss={tr_loss:.4f}  char_acc={char_acc:.4f}  plate_acc={plate_acc:.4f}  fmt_acc={fmt_acc:.4f}  ({time.time()-tic:.1f}s)")
        else:
            tr_loss = train_epoch_single(model, dl_train, device, scaler, ce_losses, optimizer)
            char_acc, plate_acc = eval_single(model, dl_val, device)
            print(f"[{epoch:03d}] loss={tr_loss:.4f}  char_acc={char_acc:.4f}  plate_acc={plate_acc:.4f}  ({time.time()-tic:.1f}s)")

        if plate_acc > best_plate:
            best_plate = plate_acc
            torch.save({
                'model': model.state_dict(),
                'is_multi': is_multi,
                'meta': {
                    'input_hw': (io_h, io_w),
                    'in_channels': in_channels,
                    'lambda_fmt': train_cfg.get('lambda_fmt', 0.20) if is_multi else None,
                }
            }, os.path.join(args.out_dir, "best.ckpt"))

        if (epoch % train_cfg.get('ckpt_every', 5)) == 0:
            torch.save({'model': model.state_dict()}, os.path.join(args.out_dir, f"epoch_{epoch}.ckpt"))

if __name__ == "__main__":
    main()