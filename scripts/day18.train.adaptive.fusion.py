"""
Gun 18 - Adaptive Fusion / Gating Module (alpha-weighted fusion)
======================================================================
Gun 15/16'daki Feature Fusion'da RGB ve Thermal feature map'leri HER ZAMAN
ayni sabit sekilde (concat + 1x1 conv) birlestiriliyordu. Burada, her
goruntu icin "RGB'ye mi Thermal'e mi daha fazla guvenmeliyim?" sorusunu
otomatik cevaplayan kucuk bir "Gate" (kapi) agi ekliyoruz:

    alpha = Gate(ortalama(RGB_features), ortalama(Thermal_features))  # 0-1 arasi
    fused = alpha * RGB_features + (1 - alpha) * Thermal_features

alpha goruntunun GENEL BAGLAMINA (global context, ortalama havuzlama ile)
gore hesaplanir - gunduz/parlak sahnelerde RGB'ye, gece/karanlik sahnelerde
Thermal'e kaymasi beklenir. Bu degerler Gun 20'de day/night kirilimiyla
analiz edilecek (backbone.last_alphas uzerinden erisilebilir).

Kullanim:
  python day18_train_adaptive_fusion.py --data_root C:\\...\\kaist-cvpr15 --backbone resnet50 --epochs 3 --max_train_images 2500 --batch_size 1 --image_size 320 256 --output_dir C:\\...\\checkpoints
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
# 1. RGB+Thermal'i 6 kanalli tek tensor olarak dondüren wrapper (Gun 15 ile ayni)
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
# 2. Adaptive Gate: alpha-weighted fusion (Gun 15'teki sabit concat+conv YERINE)
# ---------------------------------------------------------------------------

class AdaptiveGate(nn.Module):
    """
    RGB ve Thermal feature map'lerinin GLOBAL ortalamasindan (global average
    pooling) tek bir skaler guven degeri (alpha, goruntu basina) ogrenir.
    alpha=1 -> tamamen RGB'ye guven, alpha=0 -> tamamen Thermal'e guven.
    """
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
        alpha = self.fc(combined).view(b, 1, 1, 1)  # yayilabilir (broadcastable) sekil

        fused = alpha * rgb_feat + (1 - alpha) * thermal_feat
        return fused, alpha.view(b)  # alpha'yi da dondur (analiz icin, Gun 20)


class AdaptiveFusionFPNBackbone(nn.Module):
    """
    Gun 15'teki DualStreamFPNBackbone ile AYNI dual-stream FPN temeli, TEK
    FARK: sabit concat+1x1conv fusion yerine ogrenilebilir AdaptiveGate
    kullanilir - her piramit seviyesinde kendi gate'i var.
    """
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

        # En son forward cagrisindaki alpha degerlerini saklar - egitim/test
        # sirasinda backbone.last_alphas uzerinden okunabilir (Gun 20 analizi icin).
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


def build_adaptive_fusion_model(num_classes=2, backbone_name="mobilenet"):
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

    # Gun 15'teki AYNI iyilestirme: RPN ve box_head'i rastgele degil, hazir
    # COCO-onceden-egitilmis agirliklarla baslat.
    if backbone_name == "mobilenet":
        pretrained_ref = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
    else:
        pretrained_ref = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    try:
        model.rpn.load_state_dict(pretrained_ref.rpn.state_dict())
        model.roi_heads.box_head.load_state_dict(pretrained_ref.roi_heads.box_head.state_dict())
        print("RPN ve box_head COCO onceden-egitilmis agirliklarla baslatildi (rastgele degil).")
    except Exception as e:
        print(f"UYARI: pretrained RPN/box_head agirliklari yuklenemedi, rastgele init ile devam ediliyor: {e}")
    del pretrained_ref

    return model


# ---------------------------------------------------------------------------
# 3. Egitim dongusu (Gun 8/10/13/15 ile ayni guvenlik onlemleri)
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, epoch, output_dir, print_every=10,
                     max_grad_norm=1.0, save_every=100):
    model.train()
    total_loss = 0.0
    skipped_batches = 0
    start = time.time()
    alpha_log = []  # bu epoch'ta gorulen alpha degerlerini topla (bilgi amacli)

    for i, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        if not torch.isfinite(loss):
            skipped_batches += 1
            print(f"  UYARI: batch {i+1} - loss NaN/Inf cikti, bu batch atlaniyor.")
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        total_loss += loss.item()

        # model.backbone uzerinden son alpha degerlerine eris (bilgi/log amacli)
        if hasattr(model.backbone, "last_alphas") and model.backbone.last_alphas:
            first_level_key = next(iter(model.backbone.last_alphas))
            alpha_log.append(model.backbone.last_alphas[first_level_key].mean().item())

        if (i + 1) % print_every == 0:
            elapsed = time.time() - start
            avg_alpha = sum(alpha_log[-print_every:]) / max(len(alpha_log[-print_every:]), 1)
            print(f"  [epoch {epoch}] batch {i+1}/{len(loader)} "
                  f"- loss={loss.item():.4f} - ort.alpha(son {print_every})={avg_alpha:.3f} "
                  f"- gecen sure={elapsed:.1f}s")

        if (i + 1) % save_every == 0:
            mid_ckpt_path = os.path.join(output_dir, f"adaptive_fusion_epoch{epoch}_batch{i+1}.pth")
            torch.save(model.state_dict(), mid_ckpt_path)
            print(f"  [ara checkpoint kaydedildi: {mid_ckpt_path}]")

    valid_batches = max(len(loader) - skipped_batches, 1)
    avg_loss = total_loss / valid_batches
    print(f"Epoch {epoch} tamamlandi. Ortalama loss: {avg_loss:.4f} "
          f"({skipped_batches} batch NaN/Inf nedeniyle atlandi)")
    return avg_loss


# ---------------------------------------------------------------------------
# 4. Ana calisma
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gun 18 Adaptive Fusion/Gating egitimi")
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

    print("=" * 60)
    print("KULLANILAN AYARLAR:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanilan cihaz: {device}")

    print("\nDataset yukleniyor...")
    base_train_ds = KAISTPedestrianDataset(
        args.data_root, split="train", only_reasonable=True,
        image_size=tuple(args.image_size) if args.image_size else None
    )
    print(f"Toplam train frame sayisi: {len(base_train_ds)}")

    if args.max_train_images < len(base_train_ds):
        indices = list(range(0, len(base_train_ds), max(1, len(base_train_ds) // args.max_train_images)))
        indices = indices[:args.max_train_images]
        base_train_ds = Subset(base_train_ds, indices)
        print(f"Egitim icin {len(base_train_ds)} goruntuye indirgendi.")

    train_ds = FeatureFusionWrapper(base_train_ds)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=detection_collate_fn, num_workers=0
    )

    print(f"\nModel kuruluyor (Adaptive Gating {args.backbone} - RGB ve Thermal icin ayri backbone + ogrenilebilir gate)...")
    model = build_adaptive_fusion_model(backbone_name=args.backbone)
    if args.resume_from:
        print(f"Checkpoint'ten devam ediliyor: {args.resume_from}")
        model.load_state_dict(torch.load(args.resume_from, map_location="cpu", weights_only=True))
    model.to(device)

    # ONEMLI DUZELTME: Gate agi (kucuk, sifirdan baslatilmis) ile geri kalan
    # model (buyuk, COCO onceden-egitilmis) ayni lr ile egitilirse, Gate'e
    # ulasan gradyan sinyali cok zayif kalir ve alpha ~0.5'te "sikisir" (tam
    # olarak Gun 18 sonucunda gozlemledigimiz sorun). Cozum: Gate parametrelerine
    # ayri ve çok daha YUKSEK bir ogrenme orani (lr) veriyoruz - boylece az
    # sayida epoch icinde bile anlamli sekilde ogrenebilir.
    # ONEMLI DUZELTME: ...
    gate_params = list(model.backbone.gates.parameters())
    gate_param_ids = {id(p) for p in gate_params}
    other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in gate_param_ids]

    # التعديل هنا: جعلنا سرعة تعلم البوابة 50 ضعف سرعة النموذج الأساسي
    optimizer = torch.optim.SGD([
        {"params": other_params, "lr": args.lr},
        {"params": gate_params, "lr": args.lr * 50},  # تم التعديل من 10 إلى 50
    ], momentum=0.9, weight_decay=0.0005)

    # تحديث نصوص الطباعة لتتطابق مع التعديل الجديد
    print(f"Gate parametre sayisi: {sum(p.numel() for p in gate_params)} (lr={args.lr * 50})")
    print(f"Diger parametre sayisi: {sum(p.numel() for p in other_params)} (lr={args.lr})")
    print(f"\nEgitim basliyor: {args.epochs} epoch, batch_size={args.batch_size}\n")
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, optimizer, device, epoch, args.output_dir)

        ckpt_path = os.path.join(args.output_dir, f"adaptive_fusion_{args.backbone}_epoch{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint kaydedildi: {ckpt_path}\n")

    print("Gun 18 Adaptive Fusion egitimi tamamlandi. Sonraki adim (Gun 19-20):")
    print("bu checkpoint'i test edip, ogrenilen alpha degerlerini day/night kirilimiyla analiz etmek.")