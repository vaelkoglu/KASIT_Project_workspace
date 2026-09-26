"""
Gun 10 - Thermal-only Baseline Egitimi (Faster R-CNN, transfer learning)
===========================================================================
Gun 8'in (RGB-only) birebir ayni mantigi, tek fark: goruntu kaynagi olarak
visible (RGB) yerine lwir (thermal) goruntuler kullaniliyor. Boylece
gece/gunduz performansini Gun 11'de RGB ile adil sekilde karsilastirabiliriz.

Kullanim:
  python day10_train_thermal.py --data_root C:\\...\\kaist-cvpr15 --backbone mobilenet --epochs 5 --max_train_images 1500 --batch_size 2 --image_size 480 384 --output_dir C:\\...\\checkpoints
"""

import os
import argparse
import time

import torch
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from day6_dataset import KAISTPedestrianDataset  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Sadece Thermal (LWIR) donduren wrapper Dataset
# ---------------------------------------------------------------------------

class ThermalOnlyWrapper(torch.utils.data.Dataset):
    """
    KAISTPedestrianDataset (visible, lwir, target) donduruyor.
    Bu wrapper sadece lwir + target'i alir - Gun 8'deki RGBOnlyWrapper'in
    aynisi, tek fark: visible_img yerine lwir_img donduruluyor.
    """
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        _visible_img, lwir_img, target = self.base[idx]
        clean_target = {
            "boxes": target["boxes"],
            "labels": target["labels"],
        }
        return lwir_img, clean_target


def detection_collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


# ---------------------------------------------------------------------------
# 2. Model kurulumu (Gun 8 ile birebir ayni)
# ---------------------------------------------------------------------------

def build_model(num_classes=2, backbone="mobilenet"):
    if backbone == "mobilenet":
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
    else:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# 3. Egitim dongusu (Gun 8 ile birebir ayni - NaN koruma + ara checkpoint)
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
            mid_ckpt_path = os.path.join(output_dir, f"thermal_baseline_epoch{epoch}_batch{i+1}.pth")
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
    parser = argparse.ArgumentParser(description="Gun 10 Thermal-only baseline egitimi")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max_train_images", type=int, default=1500)
    parser.add_argument("--backbone", type=str, choices=["resnet50", "mobilenet"], default="mobilenet")
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
        print(f"Egitim icin {len(base_train_ds)} goruntuye indirgendi (--max_train_images).")

    train_ds = ThermalOnlyWrapper(base_train_ds)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=detection_collate_fn, num_workers=0
    )

    print(f"\nModel kuruluyor (COCO onceden-egitilmis Faster R-CNN [{args.backbone}], person icin uyarlaniyor)...")
    model = build_model(num_classes=2, backbone=args.backbone)
    if args.resume_from:
        print(f"Checkpoint'ten devam ediliyor: {args.resume_from}")
        model.load_state_dict(torch.load(args.resume_from, map_location="cpu"))
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)

    print(f"\nEgitim basliyor: {args.epochs} epoch, batch_size={args.batch_size}\n")
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, optimizer, device, epoch, args.output_dir)

        ckpt_path = os.path.join(args.output_dir, f"thermal_baseline_epoch{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint kaydedildi: {ckpt_path}\n")

    print("Gun 10 egitimi tamamlandi. Sonraki adim (Gun 11): RGB ve Thermal")
    print("sonuclarini day/night kirilimiyla karsilastirmak.")