"""
Gun 21 - Robust Fusion via Modality Dropout
======================================================================
Bu kod, Gun 18'deki Adaptive Fusion (alpha-weighted) modelinin uzerine
"Modality Dropout" (rastgele sensor kapatma) mekanizmasini ekler.
Amac: Egitim sirasinda %15 ihtimalle RGB kamerasini, %15 ihtimalle
Termal kamerasini tamamen karartarak (sifirlayarak) modelin tek bir
sensore asiri bagimli (over-reliant) olmasini engellemek ve donanim
hatalarina (hardware failure) karsi direncli (robust) hale getirmektir.

Kullanim:
  python day21_train_robust_fusion.py --data_root C:\...\kaist-cvpr15 --backbone resnet50 --epochs 3 --batch_size 1 --output_dir ./checkpoints
"""

import os
import time
import argparse
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from day6_dataset import KAISTPedestrianDataset  # noqa: E402


# ---------------------------------------------------------------------------
# 1. RGB+Thermal'i 6 kanalli tek tensor olarak dondüren wrapper
# ---------------------------------------------------------------------------

class FeatureFusionWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        visible_img, lwir_img, target = self.base[idx]
        stacked = torch.cat((visible_img, lwir_img), dim=0)  # [6, H, W]
        clean_target = {"boxes": target["boxes"], "labels": target["labels"]}
        return stacked, clean_target


def detection_collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


# ---------------------------------------------------------------------------
# 2. Adaptive Gate ve Modality Dropout içeren Dual-Stream Backbone
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
    def __init__(self, backbone_name="mobilenet"):
        super().__init__()
        if backbone_name == "mobilenet":
            rgb_full = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
            thermal_full = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
        else:
            rgb_full = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
            thermal_full = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

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

        # ---------------------------------------------------------
        # DAY 21 - MODALITY DROPOUT (SENSÖR KARARTMA) EKLENTİSİ
        # Sadece eğitim sırasında aktif olur, test sırasında pasiftir.
        # ---------------------------------------------------------
        if self.training:
            rand_val = torch.rand(1).item()
            if rand_val < 0.15:
                rgb = torch.zeros_like(rgb)  # %15 ihtimalle RGB'yi karart
            elif rand_val < 0.30:
                thermal = torch.zeros_like(thermal)  # %15 ihtimalle Thermali karart
            # Kalan %70 ihtimalle iki sensör de normal çalışır.
        # ---------------------------------------------------------

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
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=[str(k) for k in backbone.level_keys],
        output_size=7, sampling_ratio=2,
    )

    model = FasterRCNN(backbone, num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)
    model.transform.image_mean = [0.485, 0.456, 0.406, 0.485, 0.456, 0.406]
    model.transform.image_std = [0.229, 0.224, 0.225, 0.229, 0.224, 0.225]

    if backbone_name == "mobilenet":
        pretrained_ref = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
    else:
        pretrained_ref = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    try:
        model.rpn.load_state_dict(pretrained_ref.rpn.state_dict())
        model.roi_heads.box_head.load_state_dict(pretrained_ref.roi_heads.box_head.state_dict())
        print("RPN ve box_head COCO onceden-egitilmis agirliklarla baslatildi.")
    except Exception as e:
        print(f"UYARI: pretrained RPN/box_head agirliklari yuklenemedi: {e}")
    del pretrained_ref

    return model


# ---------------------------------------------------------------------------
# 3. Egitim dongusu
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, epoch, output_dir, print_every=10, max_grad_norm=1.0,
                    save_every=100):
    model.train()
    total_loss = 0.0
    skipped_batches = 0
    start = time.time()
    alpha_log = []

    for i, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        if not torch.isfinite(loss):
            skipped_batches += 1
            print(f"  UYARI: batch {i + 1} - loss NaN/Inf cikti, bu batch atlaniyor.")
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        total_loss += loss.item()

        if hasattr(model.backbone, "last_alphas") and model.backbone.last_alphas:
            first_level_key = next(iter(model.backbone.last_alphas))
            alpha_log.append(model.backbone.last_alphas[first_level_key].mean().item())

        if (i + 1) % print_every == 0:
            elapsed = time.time() - start
            avg_alpha = sum(alpha_log[-print_every:]) / max(len(alpha_log[-print_every:]), 1)
            print(f"  [epoch {epoch}] batch {i + 1}/{len(loader)} "
                  f"- loss={loss.item():.4f} - ort.alpha(son {print_every})={avg_alpha:.3f} "
                  f"- gecen sure={elapsed:.1f}s")

        if (i + 1) % save_every == 0:
            mid_ckpt_path = os.path.join(output_dir, f"robust_fusion_epoch{epoch}_batch{i + 1}.pth")
            torch.save(model.state_dict(), mid_ckpt_path)
            print(f"  [ara checkpoint kaydedildi: {mid_ckpt_path}]")

    valid_batches = max(len(loader) - skipped_batches, 1)
    avg_loss = total_loss / valid_batches
    print(f"Epoch {epoch} tamamlandi. Ortalama loss: {avg_loss:.4f} ({skipped_batches} batch atlandi)")
    return avg_loss


# ---------------------------------------------------------------------------
# 4. Ana calisma
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gun 21 Modality Dropout egitimi")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max_train_images", type=int, default=2500)
    parser.add_argument("--backbone", type=str, choices=["resnet50", "mobilenet"], default="resnet50")
    parser.add_argument("--image_size", type=int, nargs=2, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanilan cihaz: {device}")

    print("\nDataset yukleniyor...")
    base_train_ds = KAISTPedestrianDataset(
        args.data_root, split="train", only_reasonable=True,
        image_size=tuple(args.image_size) if args.image_size else None
    )

    if args.max_train_images < len(base_train_ds):
        indices = list(range(0, len(base_train_ds), max(1, len(base_train_ds) // args.max_train_images)))
        indices = indices[:args.max_train_images]
        base_train_ds = Subset(base_train_ds, indices)

    train_ds = FeatureFusionWrapper(base_train_ds)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=detection_collate_fn, num_workers=0
    )

    print(f"\nModel kuruluyor (Robust Fusion / Modality Dropout {args.backbone})...")
    model = build_adaptive_fusion_model(backbone_name=args.backbone)
    if args.resume_from:
        print(f"Checkpoint'ten devam ediliyor: {args.resume_from}")
        model.load_state_dict(torch.load(args.resume_from, map_location="cpu", weights_only=True))
    model.to(device)

    # 50x LR carpanli Gate Optimizasyonu aynen korunuyor (Degistirmeyin)
    gate_params = list(model.backbone.gates.parameters())
    gate_param_ids = {id(p) for p in gate_params}
    other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in gate_param_ids]

    optimizer = torch.optim.SGD([
        {"params": other_params, "lr": args.lr},
        {"params": gate_params, "lr": args.lr * 50},
    ], momentum=0.9, weight_decay=0.0005)

    print(f"\nEgitim basliyor: {args.epochs} epoch, batch_size={args.batch_size}\n")
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, optimizer, device, epoch, args.output_dir)
        ckpt_path = os.path.join(args.output_dir, f"robust_fusion_{args.backbone}_epoch{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint kaydedildi: {ckpt_path}\n")