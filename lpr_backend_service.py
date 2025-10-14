# lpr_backend_service.py
import base64, os
from typing import Dict, Any, List, Optional

import numpy as np
import cv2
import torch

from lpr_mobilevit import set_seed
from lpr_mobilevit import build_val_augs
from lpr_mobilevit import MobileViTBackbone
from lpr_mobilevit import MultiPathOCR, MultiTemplateOCR
from lpr_mobilevit import (
    resolve_config_inheritance,
    load_format_config,
    load_multi_format_config,
    NAMED_CHARSETS,  # added
)

# Optional (falls back gracefully if not present)
try:
    from lpr_mobilevit import charlevel_beam_rescore
except Exception:
    charlevel_beam_rescore = None


# ----------------- small helpers -----------------

def _ensure_bgr(img: np.ndarray) -> np.ndarray:
    """Normalize any input (Gray/BGRA/16-bit) to BGR uint8."""
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.dtype != np.uint8:
        maxv = 65535.0 if img.dtype == np.uint16 else float(np.max(img) or 1.0)
        img = cv2.convertScaleAbs(img, alpha=255.0 / maxv)
    return img


def _format_with_pattern(raw: str, pattern: str) -> str:
    """Insert separators according to template pattern (L/# are filled from raw)."""
    if not pattern:
        return raw
    out, i = [], 0
    for ch in pattern:
        if ch in ("L", "#"):
            if i < len(raw):
                out.append(raw[i]); i += 1
        else:
            out.append(ch)
    return "".join(out)


def _laplacian_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _preprocess_if_blurry(img_bgr: np.ndarray, blur_gate: Optional[float]) -> np.ndarray:
    """
    CLAHE + unsharp only when the image looks blurrier than 'blur_gate'
    (measured via Laplacian variance on grayscale).
    """
    if blur_gate is None:
        return img_bgr
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if _laplacian_var(gray) >= float(blur_gate):
        return img_bgr

    # CLAHE on Y channel
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(y)
    eq = cv2.cvtColor(cv2.merge((y2, cr, cb)), cv2.COLOR_YCrCb2BGR)

    # Unsharp mask
    g = cv2.GaussianBlur(eq, (0, 0), 1.0)
    sharp = cv2.addWeighted(eq, 1.6, g, -0.6, 0)
    return sharp


def _expand_slot_charsets(named_map: Dict[str, str], tpl) -> List[str]:
    """
    Expand a template's per-slot charsets using NAMED_CHARSETS (when referenced
    by name in YAML). Falls back to common aliases when needed.
    """
    out: List[str] = []
    for cs in tpl.charsets:
        if isinstance(cs, str):
            key = cs.strip()
            if key in named_map:
                out.append(named_map[key])
            else:
                lk = key.lower()
                if lk in ("digit", "digits"):
                    out.append("0123456789")
                elif lk.startswith("letters"):
                    out.append("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                elif lk in ("alnum", "letters_digits", "letters+digits"):
                    out.append("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                else:
                    # Last resort: treat the literal string as the set itself
                    # (keeps system robust even if a custom string is provided).
                    out.append("".join(dict.fromkeys(list(key))))
        elif isinstance(cs, (list, tuple)):
            out.append("".join(cs))
        else:
            out.append("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    return out
# -------------------------------------------------


class PlateRecognizer:
    """
    One model, two decode paths:
      - 'v3' = MultiTemplateOCR.decode() baseline, but we *truly* keep the best among Top‑K templates.
      - 'v4' = v3 + confidence‑gated char‑level re‑scoring (uses per‑slot logits if available).

    Key knobs
    ---------
    topk_templates : evaluate K router candidates and keep the best (robust to router mistakes).
    conf_gate      : run v4 re‑scoring only if baseline confidence < gate (set None to always rescore).
    char_topk      : per‑slot Top‑K kept for beam re‑scoring.
    beam           : flip up to N ambiguous slots.
    gap_delta      : slot is "ambiguous" if (top1 - top2) < gap_delta.
    blur_gate      : apply CLAHE+unsharp only if LapVar(gray) < blur_gate (set None to disable).
    """

    def __init__(
        self,
        config: str,
        ckpt: str,
        device: str = "auto",
        seed: int = 123,
        amp: bool = True,
        topk_templates: int = 3,
        conf_gate: Optional[float] = 0.96,
        char_topk: int = 2,
        beam: int = 2,
        gap_delta: float = 0.20,
        blur_gate: Optional[float] = 80.0,
    ):
        set_seed(seed)
        self.conf_gate = None if conf_gate is None else float(conf_gate)
        self.char_topk = int(char_topk)
        self.beam = int(beam)
        self.gap_delta = float(gap_delta)
        self.topk_templates = int(topk_templates)
        self.blur_gate = None if blur_gate is None else float(blur_gate)

        # Resolve config (respects `inherits:`)
        self.cfg_merged = resolve_config_inheritance(config)
        self.model_cfg = self.cfg_merged.get("model", {})
        self.in_h = int(self.model_cfg.get("in_h", self.cfg_merged.get("input_height", 128)))
        self.in_w = int(self.model_cfg.get("in_w", self.cfg_merged.get("input_width", 256)))
        self.in_ch = int(self.cfg_merged.get("in_channels", 3))

        # Build transforms exactly like train/eval
        cfg_for_tfms = getattr(self, "cfg_merged", getattr(self, "cfg", None))
        self.tfms = build_val_augs(self.in_h, self.in_w, cfg_for_tfms)


        # Build backbone + head
        self.is_multi = ("templates" in self.cfg_merged)
        if self.is_multi:
            self.mt_cfg = load_multi_format_config(config)
            num_classes_per_template = [[len(cs) for cs in t.charsets] for t in self.mt_cfg.templates]
            backbone = MobileViTBackbone(
                in_ch=self.in_ch,
                channels=tuple(self.model_cfg.get("channels", [48, 64, 80])),
                dims=tuple(self.model_cfg.get("dims", [96, 120, 144])),
                depths=tuple(self.model_cfg.get("depths", [2, 4, 3])),
                patch=int(self.model_cfg.get("patch", 2)),
                expand=int(self.model_cfg.get("expand", 4)),
            )
            self.model = MultiTemplateOCR(
                backbone, num_classes_per_template,
                fmt_hidden=256, head_hidden=512, head_dropout=0.1
            )
        else:
            self.fmt_cfg = load_format_config(config)
            backbone = MobileViTBackbone(
                in_ch=self.in_ch,
                channels=tuple(self.model_cfg.get("channels", [48, 64, 80])),
                dims=tuple(self.model_cfg.get("dims", [96, 120, 144])),
                depths=tuple(self.model_cfg.get("depths", [2, 4, 3])),
                patch=int(self.model_cfg.get("patch", 2)),
                expand=int(self.model_cfg.get("expand", 4)),
            )
            self.model = MultiPathOCR(backbone, [len(cs) for cs in self.fmt_cfg.charsets])

        # Device & AMP
        self.device = ("cuda" if (device == "auto" and torch.cuda.is_available()) else device)
        self.model.to(self.device).eval()
        self.use_amp = bool(self.cfg_merged.get("eval", {}).get("amp", True)) and (self.device == "cuda")

        # Load weights
        ckpt_obj = torch.load(ckpt, map_location=self.device)
        state = ckpt_obj["model"] if isinstance(ckpt_obj, dict) and "model" in ckpt_obj else ckpt_obj
        self.model.load_state_dict(state, strict=False)

        # Lookup for named charsets
        self.named_charsets = dict(NAMED_CHARSETS)  # <-- added

        # Warm-up (avoid first-hit latency spikes)
        with torch.inference_mode():
            dummy = torch.zeros(1, self.in_ch, self.in_h, self.in_w, dtype=torch.float32, device=self.device)
            _ = self.model(dummy)

    # ---------------- public API ----------------

    def infer_b64(self, b64_str: str) -> Dict[str, Any]:
        """Decode base64 (data URL or plain) -> BGR image -> run v3 & v4 from one forward pass."""
        s = b64_str.strip()
        if s.startswith("data:image"):
            s = s.split(",", 1)[1]
        s = s.replace(" ", "+")
        try:
            img_bytes = base64.b64decode(s, validate=False)
        except Exception:
            return {"error": "base64_decode_failed"}

        buf = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if img is None:
            return {"error": "cv2_imdecode_failed"}
        img = _ensure_bgr(img)
        return self._infer_bgr(img)

    # --------------- internal ---------------

    def _infer_bgr(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        # Blur‑gated enhancement (CLAHE + unsharp) — only when needed
        img_bgr = _preprocess_if_blurry(img_bgr, self.blur_gate)

        # Albumentations expects RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        x = self.tfms(image=img_rgb)["image"].unsqueeze(0).to(self.device).float()

        with torch.autocast("cuda", dtype=torch.float16, enabled=self.use_amp), torch.inference_mode():
            if self.is_multi:
                fmt_logits, tpl_logits = self.model(x)

                # ----- v3: best among Top‑K router candidates -----
                texts, confs, tids = self.model.decode(
                    (fmt_logits, tpl_logits),
                    charsets_per_template=[t.charsets for t in self.mt_cfg.templates],
                    topk_templates=max(1, self.topk_templates)
                )

                # Keep the *best* among K (not just index 0)
                if isinstance(confs, (list, tuple)):
                    best_k = int(np.argmax([float(c) for c in confs])) if len(confs) else 0
                else:
                    best_k = int(np.argmax(np.asarray(confs)))
                raw_v3, conf_v3, tid_v3 = texts[best_k], float(confs[best_k]), int(tids[best_k])
                tpl_v3 = self.mt_cfg.templates[tid_v3]
                text_v3 = _format_with_pattern(raw_v3, tpl_v3.pattern)

                # ----- v4: confidence‑gated, char‑level re‑scoring (optional) -----
                raw_v4, conf_v4, tid_v4 = raw_v3, conf_v3, tid_v3
                tpl_v4 = tpl_v3
                text_v4 = text_v3
                rescored = False

                do_rescore = (self.conf_gate is None) or (conf_v3 < self.conf_gate)
                if do_rescore and (charlevel_beam_rescore is not None):
                    # Try to slice per‑slot logits to [Pmax, Cmax] for the chosen template.
                    gl = None
                    T = len(self.mt_cfg.templates)
                    if fmt_logits.ndim == 4 and fmt_logits.shape[1] == T:
                        gl = fmt_logits[0, tid_v3]            # [Pmax, Cmax]
                    elif fmt_logits.ndim == 4 and fmt_logits.shape[-1] == T:
                        gl = fmt_logits[0, :, :, tid_v3]      # [Pmax, Cmax]
                    elif fmt_logits.ndim == 3:
                        gl = fmt_logits[0]                     # already [Pmax, Cmax] (rare)

                    if gl is not None and gl.ndim == 2:
                        slot_cs = _expand_slot_charsets(self.named_charsets, tpl_v3)  # <-- uses NAMED_CHARSETS
                        pos_mask = [ch in ("L", "#") for ch in tpl_v4.pattern]

                        res = charlevel_beam_rescore(
                            gl, slot_cs, pos_mask, tpl_v4.pattern,
                            topk=self.char_topk, max_flips=self.beam, gap_delta=self.gap_delta
                        )
                        if res and float(res["avg_conf"]) > conf_v3:
                            raw_v4 = res["text_raw"]
                            text_v4 = res["text"]
                            conf_v4 = float(res["avg_conf"])
                            rescored = True

                return {
                    "v3": {
                        "text": text_v3, "text_raw": raw_v3,
                        "confidence": conf_v3,
                        "template_id": tpl_v3.id, "pattern": tpl_v3.pattern
                    },
                    "v4": {
                        "text": text_v4, "text_raw": raw_v4,
                        "confidence": conf_v4,
                        "template_id": tpl_v4.id, "pattern": tpl_v4.pattern,
                        "rescored": rescored
                    },
                    "agree": (text_v3 == text_v4),
                    "best": ("v4" if conf_v4 >= conf_v3 else "v3"),
                    "best_confidence": (conf_v4 if conf_v4 >= conf_v3 else conf_v3),
                }

            # ---- single-format fallback (rare in your setup) ----
            logits = self.model(x)
            probs = [l.softmax(dim=1) for l in logits]
            idxs = [p.argmax(dim=1) for p in probs]
            s, csum = [], 0.0
            for p, idx, cs in zip(probs, idxs, self.fmt_cfg.charsets):
                ci = int(idx[0].item())
                s.append(cs[ci]); csum += float(p[0, ci])
            raw = "".join(s); conf = csum / len(self.fmt_cfg.charsets)
            return {
                "v3": {"text": _format_with_pattern(raw, self.fmt_cfg.pattern), "text_raw": raw,
                       "confidence": conf, "template_id": None, "pattern": self.fmt_cfg.pattern},
                "v4": {"text": _format_with_pattern(raw, self.fmt_cfg.pattern), "text_raw": raw,
                       "confidence": conf, "template_id": None, "pattern": self.fmt_cfg.pattern,
                       "rescored": False},
                "agree": True, "best": "v3", "best_confidence": conf
            }
