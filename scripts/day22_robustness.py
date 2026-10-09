"""
Gun 22 - Robustness Deneyleri (RGB/Thermal blackout, noise, misalignment)
=========================================================================
Egitilmis TUM modelleri (RGB, Thermal, Early, Feature, Adaptive, Robust)
test aninda ayni bozulmalarla degerlendirir (yeniden egitim YOK).

- Bu dosya TAMAMEN BAGIMSIZDIR: model kurucu fonksiyonlar icinde (sadece
  day6_dataset.py ve day7_evaluator.py ayni klasorde olmali).
- Bozulma, normalization SONRASINDA uygulanir - Gun 21'deki Modality
  Dropout ile AYNI nokta. "blackout" = normalize uzayinda sifir (ortalama renkli goruntu).
- Sonuclar HER MODELDEN SONRA csv'ye eklenir (kesinti olursa kayip olmaz).

Ornek (her model ayri komut, ayni --stride ve ayni --out kullan):
  python day22_robustness.py --data_root C:\\...\\data\\kaist-cvpr15 --stride 10 ^
      --models Adaptive=C:\\...\\checkpoints\\adaptive_fusion_resnet50_epoch2.pth ^
      --out C:\\...\\robustness_results.csv

Tum sonuclari tablo olarak gormek icin:
  python day22_robustness.py --summarize C:\\...\\robustness_results.csv
"""

import os
import sys
import csv
import time
import argparse
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from day6_dataset import KAISTPedestrianDataset, load_annotation_xml  # noqa: E402
from day7_evaluator import split_eval_ignore_boxes, evaluate_dataset  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Model kurucular (egitim scriptleriyle AYNI mimari - agirliklar eslesmeli)
# ---------------------------------------------------------------------------

def _fasterrcnn_fpn(backbone_name):
    if backbone_name == "mobilenet":
        return torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None)
    return torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)


def build_single_modality_model(backbone_name, num_classes=2):
    """Gun 8 (RGB) ve Gun 10 (Thermal) - ikisi de AYNI mimari."""
    model = _fasterrcnn_fpn(backbone_name)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def build_early_fusion_model(backbone_name, num_classes=2):
    """Gun 13 - 4 kanalli giris."""
    model = _fasterrcnn_fpn(backbone_name)
    if backbone_name == "mobilenet":
        old_conv = model.backbone.body['0'][0]
    else:
        old_conv = model.backbone.body.conv1
    new_conv = nn.Conv2d(4, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                         stride=old_conv.stride, padding=old_conv.padding, bias=False)
    if backbone_name == "mobilenet":
        model.backbone.body['0'][0] = new_conv
    else:
        model.backbone.body.conv1 = new_conv
    model.transform.image_mean = [0.485, 0.456, 0.406, 0.5]
    model.transform.image_std = [0.229, 0.224, 0.225, 0.5]
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


class AdaptiveGate(nn.Module):
    """Gun 18 - alpha = RGB agirligi."""
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels * 2, max(channels // 4, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // 4, 8), 1),
            nn.Sigmoid(),
        )

    def forward(self, rgb_feat, thermal_feat):
        b, c, _, _ = rgb_feat.shape
        s = torch.cat([self.pool(rgb_feat).view(b, c), self.pool(thermal_feat).view(b, c)], dim=1)
        alpha = self.fc(s).view(b, 1, 1, 1)
        return alpha * rgb_feat + (1 - alpha) * thermal_feat, alpha.view(b)


class DualStreamFPNBackbone(nn.Module):
    """fusion_type='concat' -> Gun 15 Feature Fusion | 'gate' -> Gun 18/21 Adaptive/Robust."""
    def __init__(self, backbone_name, fusion_type):
        super().__init__()
        self.rgb_backbone = _fasterrcnn_fpn(backbone_name).backbone
        self.thermal_backbone = _fasterrcnn_fpn(backbone_name).backbone
        self.out_channels = self.rgb_backbone.out_channels
        self.fusion_type = fusion_type

        with torch.no_grad():
            sample = self.rgb_backbone(torch.zeros(1, 3, 128, 128))
        self.level_keys = list(sample.keys())

        if fusion_type == "concat":
            self.fusion_convs = nn.ModuleDict()
            for k in self.level_keys:
                ch = sample[k].shape[1]
                self.fusion_convs[k] = nn.Sequential(nn.Conv2d(ch * 2, ch, kernel_size=1),
                                                     nn.ReLU(inplace=True))
        else:
            self.gates = nn.ModuleDict()
            for k in self.level_keys:
                self.gates[k] = AdaptiveGate(sample[k].shape[1])
        self.last_alphas = {}

    def forward(self, x):
        rgb_feats = self.rgb_backbone(x[:, :3])
        thermal_feats = self.thermal_backbone(x[:, 3:6])
        fused, alphas = OrderedDict(), {}
        for k in self.level_keys:
            if self.fusion_type == "concat":
                fused[k] = self.fusion_convs[k](torch.cat([rgb_feats[k], thermal_feats[k]], dim=1))
            else:
                fused[k], a = self.gates[k](rgb_feats[k], thermal_feats[k])
                alphas[k] = a.detach().cpu()
        self.last_alphas = alphas
        return fused


def build_dual_stream_model(backbone_name, fusion_type, num_classes=2):
    backbone = DualStreamFPNBackbone(backbone_name, fusion_type)
    n = len(backbone.level_keys)
    sizes = tuple((s,) for s in [32, 64, 128, 256, 512][:n])
    anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=((0.5, 1.0, 2.0),) * n)
    roi_pooler = MultiScaleRoIAlign(featmap_names=[str(k) for k in backbone.level_keys],
                                    output_size=7, sampling_ratio=2)
    model = FasterRCNN(backbone, num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)
    model.transform.image_mean = [0.485, 0.456, 0.406] * 2
    model.transform.image_std = [0.229, 0.224, 0.225] * 2
    return model


# ---------------------------------------------------------------------------
# 2. Bozulma sarmalayicisi (TUM modeller icin ayni kod)
# ---------------------------------------------------------------------------

def shift_width(t, s):
    """Yatay kaydirma; bosluklari 0 ile doldurur."""
    s = int(s)
    if s == 0:
        return t
    out = torch.zeros_like(t)
    out[..., s:] = t[..., :-s]
    return out


class InputDegradeBackbone(nn.Module):
    """Orijinal backbone'u sarar (agirliklar degismez). rgb/thermal = giris tensorundeki
    kanal dilimleri (slice) ya da None. Bozulma normalization SONRASI uygulanir."""

    def __init__(self, inner, rgb=None, thermal=None):
        super().__init__()
        self.inner = inner
        self.out_channels = inner.out_channels
        self.rgb, self.thermal = rgb, thermal
        self.degrade_mode = None
        self.degrade_level = 0.0

    @property
    def last_alphas(self):
        return getattr(self.inner, "last_alphas", {})

    def forward(self, x):
        mode, lv = self.degrade_mode, self.degrade_level
        if mode is None:
            return self.inner(x)
        x = x.clone()
        r, t = self.rgb, self.thermal
        if mode == "rgb_blackout" and r is not None:
            x[:, r] = 0
        elif mode == "thermal_blackout" and t is not None:
            x[:, t] = 0
        elif mode == "rgb_noise" and r is not None:
            x[:, r] = x[:, r] + torch.randn_like(x[:, r]) * lv
        elif mode == "thermal_noise" and t is not None:
            x[:, t] = x[:, t] + torch.randn_like(x[:, t]) * lv
        elif mode == "misalign" and r is not None and t is not None:
            x[:, t] = shift_width(x[:, t], lv)
        return self.inner(x)


def build_model_for(kind, args, ckpt):
    """Model kur -> checkpoint yukle -> bozulma sarmalayicisiyla sar."""
    if kind == "rgb":
        model = build_single_modality_model(args.rgb_backbone)
        wrap = dict(rgb=slice(0, 3))
    elif kind == "thermal":
        model = build_single_modality_model(args.thermal_backbone)
        wrap = dict(thermal=slice(0, 3))
    elif kind == "early":
        model = build_early_fusion_model(args.early_backbone)
        wrap = dict(rgb=slice(0, 3), thermal=slice(3, 4))
    elif kind == "feature":
        model = build_dual_stream_model(args.feature_backbone, "concat")
        wrap = dict(rgb=slice(0, 3), thermal=slice(3, 6))
    elif kind == "adaptive":
        model = build_dual_stream_model(args.backbone, "gate")
        wrap = dict(rgb=slice(0, 3), thermal=slice(3, 6))
    else:
        raise ValueError(kind)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.backbone = InputDegradeBackbone(model.backbone, **wrap)  # yuklemeden SONRA sar
    return model


def applicable(kind, mode):
    """Tek-modaliteli modelde kullanmadigi kanali bozmak etkisizdir (= clean)."""
    if mode == "clean":
        return True
    if kind == "rgb":
        return mode.startswith("rgb_")
    if kind == "thermal":
        return mode.startswith("thermal_")
    return True


# ---------------------------------------------------------------------------
# 3. Veri (sadece goruntuler; GT bir kez hazirlanir)
# ---------------------------------------------------------------------------

class TestImages(Dataset):
    """rgb -> [3,H,W] | thermal -> [3,H,W] | early -> [4,H,W] | 6ch -> [6,H,W]
    Goruntuler ORIJINAL olcekte (640x512) - GT ile ayni koordinat sistemi."""

    def __init__(self, base, frame_ids, kind):
        self.base, self.frame_ids, self.kind = base, frame_ids, kind

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, i):
        fid = self.frame_ids[i]
        if self.kind == "rgb":
            return self.base._load_image(fid, "visible")
        if self.kind == "thermal":
            return self.base._load_image(fid, "lwir")
        vis = self.base._load_image(fid, "visible")
        lwir = self.base._load_image(fid, "lwir")
        if self.kind == "early":
            lwir = lwir[0:1]
        return torch.cat((vis, lwir), dim=0)


def collate(batch):
    return list(batch)


def load_gt(base, frame_ids):
    eval_b, ignore_b = [], []
    for fid in frame_ids:
        s, v, n = fid.split("/")
        raw = load_annotation_xml(os.path.join(base.ann_root, s, v, n + ".xml"))
        e, ig = split_eval_ignore_boxes(raw)
        eval_b.append(e)
        ignore_b.append(ig)
    return eval_b, ignore_b


def mr_subset(dets, eval_b, ignore_b, idxs, max_thresholds):
    if not idxs:
        return float("nan")
    r = evaluate_dataset([dets[i] for i in idxs], [eval_b[i] for i in idxs],
                         [ignore_b[i] for i in idxs], num_images=len(idxs),
                         max_thresholds=max_thresholds)
    return 100.0 if r is None else float(r["log_average_miss_rate"]) * 100.0


# ---------------------------------------------------------------------------
# 4. Deney dongusu
# ---------------------------------------------------------------------------

MODES = [
    ("clean", 0),
    ("rgb_blackout", 0),
    ("thermal_blackout", 0),
    ("rgb_noise", 0.25), ("rgb_noise", 0.5), ("rgb_noise", 1.0),
    ("thermal_noise", 0.25), ("thermal_noise", 0.5), ("thermal_noise", 1.0),
    ("misalign", 2), ("misalign", 4), ("misalign", 8),
]
METRIC_KEYS = ["MR_all", "MR_day", "MR_night", "alpha_all", "alpha_day", "alpha_night"]
FIELDS = ["model", "mode", "level", "note"] + METRIC_KEYS
LOADER_KIND = {"rgb": "rgb", "thermal": "thermal", "early": "early",
               "feature": "6ch", "adaptive": "6ch"}


@torch.no_grad()
def run_one_mode(model, loader, device, mode, level):
    model.eval()
    bb = model.backbone
    bb.degrade_mode = None if mode == "clean" else mode
    bb.degrade_level = level
    torch.manual_seed(0)  # tum modellerde ayni gurultu
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)

    dets, alphas = [], []
    for i, images in enumerate(loader):
        images = [im.to(device) for im in images]
        preds = model(images)
        a = bb.last_alphas
        if a:
            alphas += torch.stack(list(a.values())).mean(0).tolist()
        else:
            alphas += [float("nan")] * len(images)
        for p in preds:
            dets.append(list(zip(p["boxes"].cpu().tolist(), p["scores"].cpu().tolist())))
        if (i + 1) % 100 == 0:
            print(f"      {i + 1}/{len(loader)}")
    bb.degrade_mode = None
    return dets, alphas


def append_rows(path, rows):
    exists = os.path.isfile(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def print_summary(rows):
    clean_mr = {r["model"]: _num(r["MR_all"]) for r in rows if r["mode"] == "clean"}
    print("\n| Model | Mod | Seviye | MR-All | dMR-All | MR-Day | MR-Night | alpha |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r.get("note"):
            print(f"| {r['model']} | {r['mode']} | {r['level']} | — | — | — | — | — |")
            continue
        mr = _num(r["MR_all"])
        d = mr - clean_mr.get(r["model"], float("nan"))
        print(f"| {r['model']} | {r['mode']} | {r['level']} | {mr:.2f} | {d:+.2f} | "
              f"{_num(r['MR_day']):.2f} | {_num(r['MR_night']):.2f} | {_num(r['alpha_all']):.3f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root")
    # Her tur icin "Isim=checkpoint.pth" listesi
    ap.add_argument("--rgb", nargs="*", default=[], help="Gun 8 RGB-only")
    ap.add_argument("--thermal", nargs="*", default=[], help="Gun 10 Thermal-only")
    ap.add_argument("--early", nargs="*", default=[], help="Gun 13 Early Fusion")
    ap.add_argument("--feature", nargs="*", default=[], help="Gun 15 Feature Fusion")
    ap.add_argument("--models", nargs="*", default=[], help="Gun 18/21 Adaptive / Robust (gate'li)")
    # Backbone'lar SENIN gercekten egittiklerinle ayni (RGB/Thermal/Early: mobilenet, Feature/Adaptive: resnet50)
    ap.add_argument("--rgb_backbone", choices=["resnet50", "mobilenet"], default="mobilenet")
    ap.add_argument("--thermal_backbone", choices=["resnet50", "mobilenet"], default="mobilenet")
    ap.add_argument("--early_backbone", choices=["resnet50", "mobilenet"], default="mobilenet")
    ap.add_argument("--feature_backbone", choices=["resnet50", "mobilenet"], default="resnet50")
    ap.add_argument("--backbone", choices=["resnet50", "mobilenet"], default="resnet50",
                    help="Adaptive / Robust modeller icin")
    ap.add_argument("--stride", type=int, default=5, help="her N. test frame'ini kullan (TUM modellerde AYNI olmali)")
    ap.add_argument("--score_thresh", type=float, default=0.05)
    ap.add_argument("--max_thresholds", type=int, default=200)
    ap.add_argument("--out", default="robustness_results.csv")
    ap.add_argument("--summarize", default=None, help="Mevcut csv'yi tablo olarak yazdir ve cik")
    args = ap.parse_args()

    if args.summarize:
        with open(args.summarize, newline="") as f:
            print_summary(list(csv.DictReader(f)))
        return
    if not args.data_root:
        sys.exit("--data_root gerekli")

    specs = ([("rgb", s) for s in args.rgb] + [("thermal", s) for s in args.thermal]
             + [("early", s) for s in args.early] + [("feature", s) for s in args.feature]
             + [("adaptive", s) for s in args.models])
    if not specs:
        sys.exit("En az bir model ver: --rgb / --thermal / --early / --feature / --models")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")

    base = KAISTPedestrianDataset(args.data_root, split="test", only_reasonable=False)
    frame_ids = base.frame_ids[::args.stride]
    day_idx = [i for i, f in enumerate(frame_ids) if base.day_night[f] == "day"]
    night_idx = [i for i, f in enumerate(frame_ids) if base.day_night[f] == "night"]
    all_idx = list(range(len(frame_ids)))
    print(f"Frame: {len(frame_ids)} (day={len(day_idx)}, night={len(night_idx)})")
    eval_b, ignore_b = load_gt(base, frame_ids)

    all_rows = []
    for kind, spec in specs:
        name, ckpt = spec.split("=", 1)
        print(f"\n=== Model: {name} [{kind}] ({ckpt}) ===")
        t0 = time.time()
        model = build_model_for(kind, args, ckpt)
        model.roi_heads.score_thresh = args.score_thresh
        model.to(device)
        loader = DataLoader(TestImages(base, frame_ids, LOADER_KIND[kind]),
                            batch_size=1, shuffle=False, collate_fn=collate, num_workers=0)

        rows, clean_row = [], None
        for mode, level in MODES:
            if not applicable(kind, mode):
                row = {"model": name, "mode": mode, "level": level, "note": "n/a (= clean)"}
                row.update({k: clean_row[k] for k in METRIC_KEYS})
                rows.append(row)
                print(f"  -> {mode} ({level}): uygulanamaz, atlandi")
                continue
            print(f"  -> {mode} (level={level})")
            dets, alphas = run_one_mode(model, loader, device, mode, level)
            row = {
                "model": name, "mode": mode, "level": level, "note": "",
                "MR_all": mr_subset(dets, eval_b, ignore_b, all_idx, args.max_thresholds),
                "MR_day": mr_subset(dets, eval_b, ignore_b, day_idx, args.max_thresholds),
                "MR_night": mr_subset(dets, eval_b, ignore_b, night_idx, args.max_thresholds),
                "alpha_all": float(np.mean(alphas)),
                "alpha_day": float(np.mean([alphas[i] for i in day_idx])) if day_idx else float("nan"),
                "alpha_night": float(np.mean([alphas[i] for i in night_idx])) if night_idx else float("nan"),
            }
            if mode == "clean":
                clean_row = row
            print("     ", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()})
            rows.append(row)

        append_rows(args.out, rows)  # her modelden SONRA kaydet
        all_rows += rows
        print(f"  [{name}] bitti ({(time.time() - t0) / 60:.1f} dk) - sonuclar eklendi: {args.out}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print_summary(all_rows)


if __name__ == "__main__":
    main()