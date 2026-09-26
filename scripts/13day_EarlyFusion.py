import os
import time
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Subset
import sys
import argparse

# 1. Bilgisayarinizin gercek yollarini ayarlama (Masaustu yollari)
sys.path.append(r'C:\Users\acer5\OneDrive\Desktop\kaist-project\scripts')

from day6_dataset import KAISTPedestrianDataset
from day8_train_rgb import detection_collate_fn


# ---------------------------------------------------------
# 2. Early Fusion Dataset Wrapper (4 Kanalli Goruntu Uretici)
# ---------------------------------------------------------
class EarlyFusionWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        visible_img, lwir_img, target = self.base[idx]

        # Thermal goruntu 3 kanal olarak geliyor (day6_dataset RGB'ye cevirdigi
        # icin) - sadece ilk kanali aliyoruz, aksi halde 3+3=6 kanal olurdu.
        if lwir_img.shape[0] == 3:
            lwir_img = lwir_img[0:1, :, :]

        fusion_img = torch.cat((visible_img, lwir_img), dim=0)  # [4, H, W]

        clean_target = {
            "boxes": target["boxes"],
            "labels": target["labels"],
        }
        return fusion_img, clean_target


# ---------------------------------------------------------
# 3. Model Mimarisi: 4 Kanal Okuyabilen Ozel Backbone
# ---------------------------------------------------------
def build_early_fusion_model(num_classes=2, backbone_name="mobilenet"):
    if backbone_name == "mobilenet":
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
        old_conv = model.backbone.body['0'][0]
    elif backbone_name == "resnet50":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        old_conv = model.backbone.body.conv1
    else:
        raise ValueError("Backbone 'mobilenet' veya 'resnet50' olmalidir")

    new_conv = nn.Conv2d(4, old_conv.out_channels,
                         kernel_size=old_conv.kernel_size,
                         stride=old_conv.stride,
                         padding=old_conv.padding,
                         bias=False)

    with torch.no_grad():
        new_conv.weight[:, :3, :, :] = old_conv.weight
        new_conv.weight[:, 3:4, :, :] = old_conv.weight.mean(dim=1, keepdim=True)

    if backbone_name == "mobilenet":
        model.backbone.body['0'][0] = new_conv
    elif backbone_name == "resnet50":
        model.backbone.body.conv1 = new_conv

    # Transform katmanini tam olarak 4 degere sabitliyoruz (3 RGB + 1 Thermal)
    model.transform.image_mean = [0.485, 0.456, 0.406, 0.5]
    model.transform.image_std = [0.229, 0.224, 0.225, 0.5]

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# ---------------------------------------------------------
# 4. Egitim Dongusu - Gun 8/10'daki GUVENLIK ONLEMLERI eklendi:
#    - NaN/Inf loss korumasi (gradient patlamasi onlemi)
#    - Ara checkpoint (elektrik kesintisi/donma durumunda ilerleme kaybolmasin)
# ---------------------------------------------------------
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

#NaN 
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
            print(f"  [Epoch {epoch}] Batch {i + 1}/{len(loader)} - Loss: {loss.item():.4f} - Sure: {elapsed:.1f}s")

        if (i + 1) % save_every == 0:
            mid_ckpt_path = os.path.join(output_dir, f"early_fusion_epoch{epoch}_batch{i+1}.pth")
            torch.save(model.state_dict(), mid_ckpt_path)
            print(f"  [ara checkpoint kaydedildi: {mid_ckpt_path}]")

    valid_batches = max(len(loader) - skipped_batches, 1)
    avg_loss = total_loss / valid_batches
    print(f"Epoch {epoch} Tamamlandi. Ortalama Loss: {avg_loss:.4f} "
          f"({skipped_batches} batch NaN/Inf nedeniyle atlandi)")
    return avg_loss


# ---------------------------------------------------------
# 5. Ana Calisma Blogu
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gun 13 Early Fusion Egitimi")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--backbone", type=str, choices=["resnet50", "mobilenet"], default="mobilenet",
                         help="mobilenet onerilir - zayif VRAM'li GPU'lar icin daha guvenli")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001,
                         help="Gun 8/10'da kararlilik icin 0.001'e dusurduk - ayni degeri koruyoruz")
    parser.add_argument("--max_train_images", type=int, default=1500)
    parser.add_argument("--image_size", type=int, nargs=2, default=None,
                         help="orn: --image_size 480 384 - RAM/VRAM kullanimini azaltir")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)

    args, _ = parser.parse_known_args()

    print("=" * 60)
    print("KULLANILAN AYARLAR:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Kullanilan Cihaz: {device}")
    print("Veri seti yukleniyor (Early Fusion Modu)...")

    base_train_ds = KAISTPedestrianDataset(
        args.data_root, split="train", only_reasonable=True,
        image_size=tuple(args.image_size) if args.image_size else None
    )

    if args.max_train_images < len(base_train_ds):
        indices = list(range(0, len(base_train_ds), max(1, len(base_train_ds) // args.max_train_images)))
        indices = indices[:args.max_train_images]
        base_train_ds = Subset(base_train_ds, indices)
        print(f"Egitim icin {len(base_train_ds)} goruntuye indirgendi.")

    train_ds = EarlyFusionWrapper(base_train_ds)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=detection_collate_fn)

    print(f"{args.backbone} modeli 4 kanalli (RGB+Thermal) giris icin hazirlaniyor...")
    model = build_early_fusion_model(backbone_name=args.backbone)
    if args.resume_from:
        print(f"Checkpoint'ten devam ediliyor: {args.resume_from}")
        model.load_state_dict(torch.load(args.resume_from, map_location="cpu"))
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)

    print(f"\nEgitim Basliyor: {args.epochs} Epoch, Batch Size: {args.batch_size}\n")

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, optimizer, device, epoch, args.output_dir)

        ckpt_path = os.path.join(args.output_dir, f"early_fusion_{args.backbone}_epoch{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Agirliklar kaydedildi (Checkpoint): {ckpt_path}\n")

    print("Gun 13 Early Fusion egitimi basariyla tamamlandi!")
    print("Sonraki adim (Gun 14): bu checkpoint'i test edip RGB/Thermal ile karsilastirmak.")