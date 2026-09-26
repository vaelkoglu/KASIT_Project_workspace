"""
Gun 15 - Feature/Halfway Fusion Egitimi (Dual-Stream FPN Backbone)
======================================================================
Gun 13'teki (Early Fusion) piksel seviyesinde birlestirmenin aksine, burada
RGB ve Thermal goruntuler ONCE ayri ayri iki TAM backbone+FPN'den geciyor
(Gun 8/10/13'te kullandigimiz AYNI kanitlanmis FPN mimarisi - fasterrcnn_
mobilenet_v3_large_fpn / fasterrcnn_resnet50_fpn'in backbone kismi), SONRA
her piramit seviyesindeki (coklu olcek) feature map'ler ayri ayri birlestiriliyor
(concat + 1x1 conv). Bu, Gun 14'te gozlemledigimiz "erken fusion'in thermal
kanali neredeyse yok sayması" sorununu cozmeyi hedefler, ve tum baseline'larla
(RGB/Thermal/Early Fusion) AYNI FPN cok-olcekli mimariyi koruyarak adil bir
karsilastirma saglar.

Kullanim:
  python day15_train_feature_fusion.py --data_root C:\\...\\kaist-cvpr15 --backbone mobilenet --epochs 3 --max_train_images 1500 --batch_size 2 --image_size 480 384 --output_dir C:\\...\\checkpoints
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
    """
    visible_img [3,H,W] + lwir_img [3,H,W] -> [6,H,W] tek tensor.
    Thermal'in 3 kanalli (duplicate) hali korunuyor - boylece ImageNet
    onceden-egitilmis agirliklari thermal akiminda da tam olarak kullanabiliriz.
    """
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        visible_img, lwir_img, target = self.base[idx]
        stacked = torch.cat((visible_img, lwir_img), dim=0)  # [6, H, W]
        clean_target = {
            "boxes": target["boxes"],
            "labels": target["labels"],
        }
        return stacked, clean_target


def detection_collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


# ---------------------------------------------------------------------------
# 2. Dual-Stream FPN Backbone: RGB ve Thermal icin AYRI backbone+FPN,
#    her piramit seviyesinde feature fusion
# ---------------------------------------------------------------------------

class DualStreamFPNBackbone(nn.Module):
    """
    Gun 8/10/13'teki AYNI kanitlanmis FPN backbone'unu (torchvision'in kendi
    fasterrcnn_*_fpn fonksiyonlarindan cikarilmis .backbone alani) iki ayri
    kopya olarak kullanir - biri RGB, biri Thermal icin. Ozel/riskli dahili
    API'lere (private return_layers, in_channels vs.) girmek yerine, zaten
    dogrulugu kanitlanmis hazir FPN backbone'u iki kere kurup ciktilarini
    seviye seviye birlestiriyoruz - hem daha guvenilir hem daha az hataya acik.
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
        self.out_channels = self.rgb_backbone.out_channels  # FPN standardi: genelde 256

        # Her piramit seviyesi (orn. '0','1','2','pool') icin ayri bir fusion
        # katmani - kanal sayilarini gercek bir forward pass ile otomatik tespit
        # ediyoruz (versiyon farkliliklarina karsi guvenli).
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 128, 128)
            sample_feats = self.rgb_backbone(dummy)

        self.level_keys = list(sample_feats.keys())
        self.fusion_convs = nn.ModuleDict()
        for key in self.level_keys:
            ch = sample_feats[key].shape[1]
            self.fusion_convs[key] = nn.Sequential(
                nn.Conv2d(ch * 2, ch, kernel_size=1),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        rgb = x[:, :3, :, :]
        thermal = x[:, 3:6, :, :]

        rgb_feats = self.rgb_backbone(rgb)
        thermal_feats = self.thermal_backbone(thermal)

        fused = OrderedDict()
        for key in self.level_keys:
            cat = torch.cat([rgb_feats[key], thermal_feats[key]], dim=1)
            fused[key] = self.fusion_convs[key](cat)
        return fused


def build_feature_fusion_model(num_classes=2, backbone_name="mobilenet"):
    backbone = DualStreamFPNBackbone(backbone_name=backbone_name)

    # Piramit seviye sayisina gore anchor generator ve ROI pooler'i otomatik
    # olustur - mobilenet ve resnet50 FPN'leri farkli sayida seviye
    # donduregebilir (biri 4, digeri 5 gibi), sabit varsaymak yerine
    # backbone.level_keys uzerinden dinamik kuruyoruz.
    num_levels = len(backbone.level_keys)
    base_sizes = [32, 64, 128, 256, 512]
    sizes = tuple((s,) for s in base_sizes[:num_levels])
    aspect_ratios = ((0.5, 1.0, 2.0),) * num_levels

    anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=aspect_ratios)
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=[str(k) for k in backbone.level_keys],
        output_size=7, sampling_ratio=2,
    )

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
    )

    # ONEMLI: girisimiz 6 kanalli (3 RGB + 3 thermal-duplicate), varsayilan
    # transform 3 kanallik ImageNet mean/std bekler - 6 kanala genisletiyoruz.
    model.transform.image_mean = [0.485, 0.456, 0.406, 0.485, 0.456, 0.406]
    model.transform.image_std = [0.229, 0.224, 0.225, 0.229, 0.224, 0.225]

    return model


# ---------------------------------------------------------------------------
# 3. Egitim dongusu (Gun 8/10/13 ile ayni guvenlik onlemleri)
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, epoch, output_dir, print_every=10,
                     max_grad_norm=1.0, save_every=100):
    model.train()
    total_loss = 0.0
    skipped_batches = 0
    start = time.time()

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

        if (i + 1) % print_every == 0:
            elapsed = time.time() - start
            print(f"  [epoch {epoch}] batch {i+1}/{len(loader)} "
                  f"- loss={loss.item():.4f} - gecen sure={elapsed:.1f}s")

        if (i + 1) % save_every == 0:
            mid_ckpt_path = os.path.join(output_dir, f"feature_fusion_epoch{epoch}_batch{i+1}.pth")
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
    parser = argparse.ArgumentParser(description="Gun 15 Feature/Halfway Fusion egitimi")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max_train_images", type=int, default=1500)
    parser.add_argument("--backbone", type=str, choices=["resnet50", "mobilenet"], default="mobilenet",
                         help="DIKKAT: iki ayri backbone yuklendigi icin bellek kullanimi Gun 8/10/13'e "
                              "gore ~2 kat fazladir. Zayif GPU'da mobilenet + kucuk image_size sart.")
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

    print(f"\nModel kuruluyor (Dual-Stream {args.backbone} - RGB ve Thermal icin ayri backbone)...")
    model = build_feature_fusion_model(backbone_name=args.backbone)
    if args.resume_from:
        print(f"Checkpoint'ten devam ediliyor: {args.resume_from}")
        model.load_state_dict(torch.load(args.resume_from, map_location="cpu", weights_only=True))
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)

    print(f"\nEgitim basliyor: {args.epochs} epoch, batch_size={args.batch_size}\n")
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, optimizer, device, epoch, args.output_dir)

        ckpt_path = os.path.join(args.output_dir, f"feature_fusion_{args.backbone}_epoch{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint kaydedildi: {ckpt_path}\n")

    print(" Feature Fusion egitimi tamamlandi.")
    print("bu checkpoint'i test edip RGB/Thermal/Early Fusion ile karsilastirmak.")