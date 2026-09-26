"""
Gun 6 (bolum 2) - KAIST Multispectral Pedestrian Dataset icin PyTorch Dataset/DataLoader
=========================================================================================
Bu modul, Gun 5'te dogruladigimiz XML semasini ve imageSets split dosyalarini kullanarak
tam bir PyTorch Dataset olusturur. Her ornek RGB (visible) goruntu + thermal (lwir)
goruntu + bounding box etiketlerini dondurur.

Kullanim (test icin, dogrudan calistirilirsa):
  python day6_dataset.py --data_root C:\\kaist-project\\data\\kaist-cvpr15 --split test

Egitimde import ederek kullanim:
  from day6_dataset import KAISTPedestrianDataset
  train_ds = KAISTPedestrianDataset(data_root, split="train")
  train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=kaist_collate_fn)
"""

import os
import argparse
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
# 1. imageSets'ten frame listesi + day/night lookup
# ---------------------------------------------------------------------------

def load_image_list(imagesets_dir, filename):
    path = os.path.join(imagesets_dir, filename)
    if not os.path.exists(path):
        fallback_path = os.path.join(imagesets_dir, filename.replace("-20", ""))
        if os.path.exists(fallback_path):
            with open(fallback_path, "r") as f:
                return [line.strip() for line in f if line.strip()]
        print(f"UYARI: {path} bulunamadi!")
        return []

    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def build_frame_list(imagesets_dir, split):
    """
    split='train' -> train-all-02.txt (day+night hepsi)
    split='test'  -> test-all-20.txt
    Her frame icin day/night bilgisi de doner: {frame_id: 'day'/'night'}
    """
    if split == "train":
        all_file, day_file, night_file = "train-all-02.txt", "train-day-02.txt", "train-night-02.txt"
    else:
        all_file, day_file, night_file = "test-all-20.txt", "test-day-20.txt", "test-night-20.txt"

    all_ids = load_image_list(imagesets_dir, all_file)
    day_ids = set(load_image_list(imagesets_dir, day_file))
    night_ids = set(load_image_list(imagesets_dir, night_file))

    day_night = {}
    for fid in all_ids:
        if fid in day_ids:
            day_night[fid] = "day"
        elif fid in night_ids:
            day_night[fid] = "night"
        else:
            day_night[fid] = "unknown"

    return all_ids, day_night


# ---------------------------------------------------------------------------
# 2. XML annotation okuma (Gun 5'teki fonksiyonla ayni)
# ---------------------------------------------------------------------------

def load_annotation_xml(xml_path):
    boxes = []
    if not os.path.isfile(xml_path):
        return boxes
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for obj in root.findall("object"):
        name_el = obj.find("name")
        bnd = obj.find("bndbox")
        occ_el = obj.find("occlusion")
        if name_el is None or bnd is None:
            continue
        try:
            x = float(bnd.find("x").text)
            y = float(bnd.find("y").text)
            w = float(bnd.find("w").text)
            h = float(bnd.find("h").text)
        except (AttributeError, TypeError, ValueError):
            continue
        occlusion = int(occ_el.text) if occ_el is not None and occ_el.text else 0
        boxes.append({
            "label": name_el.text.strip() if name_el.text else "unknown",
            "x": x, "y": y, "w": w, "h": h,
            "occlusion": occlusion,
        })
    return boxes


# ---------------------------------------------------------------------------
# 3. PyTorch Dataset
# ---------------------------------------------------------------------------

# Reasonable protokolu: sadece 'person' etiketi degerlendirilir.
LABEL_TO_ID = {"person": 1}  # 0 = background (Faster R-CNN tarzi modeller icin ayrilir)


class KAISTPedestrianDataset(Dataset):
    """
    Her __getitem__ cagrisi doner:
      visible_img: FloatTensor [3, H, W], deger araligi [0, 1]
      lwir_img:    FloatTensor [3, H, W], deger araligi [0, 1]
      target: dict {
          'boxes':  FloatTensor [N, 4]  -> (x_min, y_min, x_max, y_max) formatinda
          'labels': LongTensor [N]      -> hepsi 1 (person)
          'occlusion': LongTensor [N]
          'frame_id': str
          'day_night': str
      }

    only_reasonable=True ise (varsayilan), sadece Reasonable protokolune uyan
    kutular (person, occlusion 0-1, height>=55px) dondurulur - Gun 4'te
    belirledigimiz degerlendirme kriteri.
    """

    def __init__(self, data_root, split="train", only_reasonable=True, image_size=None):
        self.data_root = data_root
        self.split = split
        self.only_reasonable = only_reasonable
        self.image_size = image_size  # (W, H) - None ise orijinal boyutta birakir

        imagesets_dir = os.path.join(data_root, "imageSets")
        self.ann_root = os.path.join(data_root, "annotations-xml-new-sanitized")
        self.img_root = os.path.join(data_root, "images")

        self.frame_ids, self.day_night = build_frame_list(imagesets_dir, split)

    def __len__(self):
        return len(self.frame_ids)

    def _load_image(self, frame_id, modality):
        set_name, video_name, image_name = frame_id.split("/")
        img_path = os.path.join(self.img_root, set_name, video_name, modality, image_name + ".jpg")
        img = Image.open(img_path).convert("RGB")
        if self.image_size is not None:
            img = img.resize(self.image_size, Image.BILINEAR)
        return TF.to_tensor(img)  # [3, H, W], [0, 1]

    def _load_target(self, frame_id, orig_size, resized_size):
        set_name, video_name, image_name = frame_id.split("/")
        xml_path = os.path.join(self.ann_root, set_name, video_name, image_name + ".xml")
        raw_boxes = load_annotation_xml(xml_path)

        scale_x = resized_size[0] / orig_size[0] if self.image_size is not None else 1.0
        scale_y = resized_size[1] / orig_size[1] if self.image_size is not None else 1.0

        boxes, labels, occlusions = [], [], []
        for b in raw_boxes:
            if b["label"] not in LABEL_TO_ID:
                continue
            if self.only_reasonable:
                if b["occlusion"] == 2:  # heavy occlusion -> Reasonable'da yok sayilir
                    continue
                if b["h"] < 55:  # kucuk/uzak yayalar -> Reasonable'da yok sayilir
                    continue

            x_min = b["x"] * scale_x
            y_min = b["y"] * scale_y
            x_max = (b["x"] + b["w"]) * scale_x
            y_max = (b["y"] + b["h"]) * scale_y

            # Goruntu sinirlarina kirp (clip): XML'deki bazi kutular goruntu
            # disina tasabiliyor (annotation hatasi) - bu durum Faster R-CNN'de
            # negatif/sifir alanli kutulara ve dolayisiyla NaN loss'a yol acabilir.
            img_w, img_h = resized_size
            x_min = max(0.0, min(x_min, img_w))
            y_min = max(0.0, min(y_min, img_h))
            x_max = max(0.0, min(x_max, img_w))
            y_max = max(0.0, min(y_max, img_h))

            # Sifir/negatif alan, NaN veya Inf iceren kutulari tamamen at -
            # bunlar modelin loss hesaplamasini bozabilir (en olasi NaN kaynagi).
            if x_max <= x_min or y_max <= y_min:
                continue
            if not all(np.isfinite([x_min, y_min, x_max, y_max])):
                continue

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(LABEL_TO_ID[b["label"]])
            occlusions.append(b["occlusion"])

        if len(boxes) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            occ_t = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
            occ_t = torch.tensor(occlusions, dtype=torch.int64)

        return {
            "boxes": boxes_t,
            "labels": labels_t,
            "occlusion": occ_t,
            "frame_id": frame_id,
            "day_night": self.day_night[frame_id],
        }

    def __getitem__(self, idx):
        frame_id = self.frame_ids[idx]

        # orijinal boyutu ogrenmek icin visible goruntuyu once ac (KAIST: 640x512)
        set_name, video_name, image_name = frame_id.split("/")
        visible_path = os.path.join(self.img_root, set_name, video_name, "visible", image_name + ".jpg")
        with Image.open(visible_path) as im:
            orig_size = im.size  # (W, H)

        visible_img = self._load_image(frame_id, "visible")
        lwir_img = self._load_image(frame_id, "lwir")
        resized_size = self.image_size if self.image_size is not None else orig_size

        target = self._load_target(frame_id, orig_size, resized_size)

        return visible_img, lwir_img, target


def kaist_collate_fn(batch):
    """
    Her goruntude farkli sayida kutu oldugu icin (object detection'da standart),
    varsayilan collate yerine bunu kullan: DataLoader(..., collate_fn=kaist_collate_fn)
    """
    visible_imgs = torch.stack([item[0] for item in batch], dim=0)
    lwir_imgs = torch.stack([item[1] for item in batch], dim=0)
    targets = [item[2] for item in batch]
    return visible_imgs, lwir_imgs, targets


# ---------------------------------------------------------------------------
# 4. Hizli test (dogrudan calistirilirsa)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KAIST Dataset/DataLoader hizli test")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["train", "test"], default="test")
    parser.add_argument("--image_size", type=int, nargs=2, default=None,
                         help="orn: --image_size 640 512 (verilmezse orijinal boyut korunur)")
    args = parser.parse_args()

    image_size = tuple(args.image_size) if args.image_size else None
    dataset = KAISTPedestrianDataset(args.data_root, split=args.split, image_size=image_size)
    print(f"Dataset boyutu ({args.split}): {len(dataset)} frame")

    visible_img, lwir_img, target = dataset[0]
    print(f"\nIlk ornek:")
    print(f"  frame_id       : {target['frame_id']}")
    print(f"  day_night      : {target['day_night']}")
    print(f"  visible_img    : shape={tuple(visible_img.shape)}, dtype={visible_img.dtype}")
    print(f"  lwir_img       : shape={tuple(lwir_img.shape)}, dtype={lwir_img.dtype}")
    print(f"  boxes          : {target['boxes'].shape[0]} kutu")
    if target["boxes"].shape[0] > 0:
        print(f"    ornek kutu (x_min,y_min,x_max,y_max): {target['boxes'][0].tolist()}")

    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=kaist_collate_fn, num_workers=0)
    visible_batch, lwir_batch, targets_batch = next(iter(loader))
    print(f"\nDataLoader batch testi:")
    print(f"  visible_batch shape: {tuple(visible_batch.shape)}")
    print(f"  lwir_batch shape   : {tuple(lwir_batch.shape)}")
    print(f"  batch icindeki frame'ler: {[t['frame_id'] for t in targets_batch]}")
    print(f"  batch icindeki kutu sayilari: {[t['boxes'].shape[0] for t in targets_batch]}")

    print("\nDataset ve DataLoader basariyla calisti.")