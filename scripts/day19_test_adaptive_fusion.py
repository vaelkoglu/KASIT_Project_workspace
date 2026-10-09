"""
Gun 19-20 - Adaptive Fusion Modelini Test Etme ve Alpha Degerlerini Day/Night
Kirilimiyla Analiz Etme
========================================================================
Modeli yukler, test setindeki goruntuler uzerinde calistirir, ve her
goruntu icin ogrenilen Alpha (RGB'ye guven) degerini DAY/NIGHT etiketiyle
birlikte kaydeder - boylece "model gunduz mu gecede mi RGB'ye daha fazla
guveniyor?" sorusunu somut verilerle cevaplayabiliriz (Gun 20'nin tam
hedefi budur).
"""

import os
import time
import json
import random
import argparse
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from day6_dataset import KAISTPedestrianDataset

# venv2\Scripts\python.exe scripts\day19_test_adaptive_fusion.py --data_root C:\Users\acer5\OneDrive\Desktop\kaist-project\data\kaist-cvpr15 --checkpoint C:\Users\acer5\OneDrive\Desktop\kaist-project\checkpoints\adaptive_fusion_resnet50_epoch2.pth --max_images 800


# ---------------------------------------------------------------------------
# 1. Model Mimarisi (Gun 18 ile AYNI - agirliklarin eslesmesi icin zorunlu)
# ---------------------------------------------------------------------------
class AdaptiveGate(nn.Module):
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
        b, c, h, w = rgb_feat.shape
        rgb_summary = self.pool(rgb_feat).view(b, c)
        thermal_summary = self.pool(thermal_feat).view(b, c)
        combined = torch.cat([rgb_summary, thermal_summary], dim=1)
        alpha = self.fc(combined).view(b, 1, 1, 1)
        fused = alpha * rgb_feat + (1 - alpha) * thermal_feat
        return fused, alpha.view(b)


class AdaptiveFusionFPNBackbone(nn.Module):
    def __init__(self, backbone_name="resnet50"):
        super().__init__()
        if backbone_name == "mobilenet":
            rgb_full = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None)
            thermal_full = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None)
        else:
            rgb_full = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
            thermal_full = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)

        self.rgb_backbone = rgb_full.backbone
        self.thermal_backbone = thermal_full.backbone
        self.out_channels = self.rgb_backbone.out_channels

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 128, 128)
            sample_feats = self.rgb_backbone(dummy)

        self.level_keys = list(sample_feats.keys())
        self.gates = nn.ModuleDict()
        for key in self.level_keys:
            ch = sample_feats[key].shape[1]
            self.gates[key] = AdaptiveGate(ch)

        self.last_alphas = {}

    def forward(self, x):
        rgb = x[:, :3, :, :]
        thermal = x[:, 3:6, :, :]
        rgb_feats = self.rgb_backbone(rgb)
        thermal_feats = self.thermal_backbone(thermal)
        fused = OrderedDict()
        alphas = {}
        for key in self.level_keys:
            fused_feat, alpha = self.gates[key](rgb_feats[key], thermal_feats[key])
            fused[key] = fused_feat
            alphas[key] = alpha.detach().cpu()
        self.last_alphas = alphas
        return fused


def build_adaptive_fusion_model(num_classes=2, backbone_name="resnet50"):
    backbone = AdaptiveFusionFPNBackbone(backbone_name=backbone_name)
    num_levels = len(backbone.level_keys)
    base_sizes = [32, 64, 128, 256, 512]
    sizes = tuple((s,) for s in base_sizes[:num_levels])
    aspect_ratios = ((0.5, 1.0, 2.0),) * num_levels
    anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=aspect_ratios)
    roi_pooler = MultiScaleRoIAlign(featmap_names=[str(k) for k in backbone.level_keys],
                                     output_size=7, sampling_ratio=2)
    model = FasterRCNN(backbone, num_classes=num_classes,
                        rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)
    model.transform.image_mean = [0.485, 0.456, 0.406, 0.485, 0.456, 0.406]
    model.transform.image_std = [0.229, 0.224, 0.225, 0.229, 0.224, 0.225]
    return model


def make_fusion_image(visible_img, lwir_img):
    return torch.cat((visible_img, lwir_img), dim=0)


# ---------------------------------------------------------------------------
# 2. Test + Alpha cikarma (ARTIK day_night ve frame_id ile birlikte)
# ---------------------------------------------------------------------------
def test_and_extract_alphas(model, dataset, frame_ids, device, output_json):
    model.eval()
    results = []

    print(f"\nTest basliyor... Toplam test goruntusu: {len(frame_ids)}")
    start = time.time()

    frame_id_to_idx = {fid: i for i, fid in enumerate(dataset.frame_ids)}

    with torch.no_grad():
        for i, frame_id in enumerate(frame_ids):
            ds_idx = frame_id_to_idx[frame_id]
            vis_img, lwir_img, target = dataset[ds_idx]
            fusion_img = make_fusion_image(vis_img, lwir_img).to(device)

            predictions = model([fusion_img])

            alpha_val = 0.5
            if hasattr(model.backbone, "last_alphas") and model.backbone.last_alphas:
                first_level_key = next(iter(model.backbone.last_alphas))
                alpha_val = model.backbone.last_alphas[first_level_key].mean().item()

            num_boxes = len(predictions[0]["boxes"])
            results.append({
                "frame_id": frame_id,
                "day_night": dataset.day_night.get(frame_id, "unknown"),  # ONEMLI EKLENEN ALAN
                "alpha": round(alpha_val, 4),
                "detected_pedestrians": num_boxes,
            })

            if (i + 1) % 50 == 0:
                print(f"  [Test] {i + 1}/{len(frame_ids)} islendi - Alpha: {alpha_val:.3f} - Yaya: {num_boxes}")

    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nTest tamamlandi! Gecen sure: {time.time() - start:.1f} saniye.")
    print(f"Alpha verileri JSON olarak kaydedildi: {output_json}")
    return results


# ---------------------------------------------------------------------------
# 3. Gun 20: Day/Night kirilimiyla analiz + grafik
# ---------------------------------------------------------------------------
def analyze_and_plot(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    day_alphas = [r["alpha"] for r in results if r["day_night"] == "day"]
    night_alphas = [r["alpha"] for r in results if r["day_night"] == "night"]

    print("\n" + "=" * 50)
    print("GUN 20 - ALPHA (RGB GUVEN DEGERI) DAY/NIGHT ANALIZI")
    print("=" * 50)
    if day_alphas:
        print(f"[DAY]   n={len(day_alphas)}  ort.alpha={np.mean(day_alphas):.4f}  "
              f"std={np.std(day_alphas):.4f}  min={min(day_alphas):.4f}  max={max(day_alphas):.4f}")
    else:
        print("[DAY] veri yok")
    if night_alphas:
        print(f"[NIGHT] n={len(night_alphas)}  ort.alpha={np.mean(night_alphas):.4f}  "
              f"std={np.std(night_alphas):.4f}  min={min(night_alphas):.4f}  max={max(night_alphas):.4f}")
    else:
        print("[NIGHT] veri yok")

    if day_alphas and night_alphas:
        diff = np.mean(day_alphas) - np.mean(night_alphas)
        print(f"\nFark (Day - Night): {diff:+.4f}")
        if diff > 0.02:
            print("-> Model gunduz RGB'ye, gecede Thermal'e daha fazla guveniyor (BEKLENEN DAVRANIS).")
        elif diff < -0.02:
            print("-> Model BEKLENENIN TERSINE davraniyor - incelenmeli.")
        else:
            print("-> Day/Night arasinda belirgin bir fark yok - Gate henuz yeterince ayirt edici degil.")
    print("=" * 50)

    # ---- Grafik: histogram (day vs night dagilimi) ----
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 30)
    if day_alphas:
        ax.hist(day_alphas, bins=bins, alpha=0.6, label=f"Day (n={len(day_alphas)})", color="#f39c12")
    if night_alphas:
        ax.hist(night_alphas, bins=bins, alpha=0.6, label=f"Night (n={len(night_alphas)})", color="#2c3e50")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="alpha=0.5 (notr)")
    ax.set_xlabel("Alpha (1=RGB'ye tam guven, 0=Thermal'e tam guven)")
    ax.set_ylabel("Goruntu sayisi")
    ax.set_title("Gun 20: Ogrenilen Alpha Degerlerinin Day/Night Dagilimi")
    ax.legend()
    plt.tight_layout()
    hist_path = os.path.join(output_dir, "day20_alpha_histogram.png")
    plt.savefig(hist_path, dpi=150)
    plt.close()

    # ---- Grafik: ortalama alpha bar chart ----
    fig, ax = plt.subplots(figsize=(5, 5))
    means = [np.mean(day_alphas) if day_alphas else 0, np.mean(night_alphas) if night_alphas else 0]
    bars = ax.bar(["Day", "Night"], means, color=["#f39c12", "#2c3e50"])
    ax.bar_label(bars, fmt="%.3f")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("Ortalama Alpha")
    ax.set_title("Gun 20: Ortalama Ogrenilen Guven Agirligi\n(alpha_day vs alpha_night)")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    bar_path = os.path.join(output_dir, "day20_alpha_mean_bar.png")
    plt.savefig(bar_path, dpi=150)
    plt.close()

    print(f"\nGrafikler kaydedildi:\n  - {hist_path}\n  - {bar_path}")


# ---------------------------------------------------------------------------
# 4. Ana calisma
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # ONEMLI DUZELTME: eksik 'data' klasoru eklendi
    DEFAULT_DATA_ROOT = r"C:\Users\acer5\OneDrive\Desktop\kaist-project\data\kaist-cvpr15"
    DEFAULT_CHECKPOINT = r"C:\Users\acer5\OneDrive\Desktop\kaist-project\checkpoints\adaptive_fusion_resnet50_epoch2.pth"
    DEFAULT_OUTPUT_JSON = r"C:\Users\acer5\OneDrive\Desktop\kaist-project\alpha_logs.json"
    DEFAULT_OUTPUT_DIR = r"C:\Users\acer5\OneDrive\Desktop\kaist-project\comparison_results"

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--max_images", type=int, default=800,
                         help="Test edilecek goruntu sayisi (dengeli day/night ornegi icin random.sample kullanilir)")
    parser.add_argument("--output_json", type=str, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanilan cihaz: {device}")

    print("\nTest veri seti yukleniyor (split='test')...")
    # image_size KASITLI OLARAK None - orijinal olcekte yukleniyor (Gun 16'da
    # ogrendigimiz olcek-uyumsuzlugu hatasindan kacinmak icin).
    dataset = KAISTPedestrianDataset(args.data_root, split="test", only_reasonable=True, image_size=None)

    random.seed(42)
    all_ids = dataset.frame_ids
    if args.max_images < len(all_ids):
        frame_ids = random.sample(all_ids, args.max_images)  # dengeli day/night ornegi icin rastgele
    else:
        frame_ids = all_ids
    print(f"Toplam test frame: {len(all_ids)} | Test edilecek: {len(frame_ids)}")

    print(f"\nModel kuruluyor ({args.backbone})...")
    model = build_adaptive_fusion_model(num_classes=2, backbone_name=args.backbone)

    print(f"Agirliklar yukleniyor: {args.checkpoint}")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.to(device)

    results = test_and_extract_alphas(model, dataset, frame_ids, device, args.output_json)
    analyze_and_plot(results, args.output_dir)