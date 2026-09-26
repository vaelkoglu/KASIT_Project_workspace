"""
Gün 5 - KAIST Multispectral Pedestrian Dataset EDA (Exploratory Data Analysis)
================================================================================
Bu script şunları analiz eder:
  1. Day/Night dağılımı (kaç görüntü/annotation gündüz vs gece)
  2. Bounding box yükseklik dağılımı (küçük/uzak vs büyük/yakın yayalar)
  3. Occlusion (kapanma) oranları
  4. Genel istatistik özeti (annotation sayısı, kişi başına ortalama, vs.)

Kullanım:
  python day5_eda.py --data_root C:\\kaist-project\\data\\kaist-cvpr15 --split test --output_dir ./eda_results
  python day5_eda.py --data_root C:\\kaist-project\\data\\kaist-cvpr15 --split train --output_dir ./eda_results_train

Gerçek klasör yapısı (kullanıcının indirdiği veri setinden doğrulandı):
  data_root/
    images/
      set00/V000/{lwir,visible}/I00000.jpg ...
    annotations-xml-new/                  <- orijinal (temizlenmemiş) annotation
    annotations-xml-new-sanitized/        <- sanitized annotation (Gün 4'te karar verdiğimiz gibi bunu kullanıyoruz)
      set00/V000/I00000.xml
    imageSets/
      train-all-02.txt, train-day-02.txt, train-night-02.txt,
      test-all-20.txt,  test-day-20.txt,  test-night-20.txt, ...

XML formatı (gerçek örnekten doğrulandı):
  <annotation>
    <object>
      <name>person</name>          <!-- person / people / cyclist / person? -->
      <bndbox><x>..</x><y>..</y><w>..</w><h>..</h></bndbox>
      <occlusion>0</occlusion>      <!-- 0=yok, 1=kısmi, 2=ağır -->
    </object>
    ...
  </annotation>
"""

import os
import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")  # ekran olmadan da PNG kaydedebilmek için
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. imageSets üzerinden day/night ground-truth listesi oluşturma
# ---------------------------------------------------------------------------

def load_image_list(imagesets_dir, filename):
    """imageSets/<filename> içindeki satırları (örn. 'set00/V000/I00980') okur."""
    path = os.path.join(imagesets_dir, filename)
    if not os.path.isfile(path):
        print(f"UYARI: '{path}' bulunamadı, atlanıyor.")
        return set()
    with open(path, "r") as f:
        return set(line.strip() for line in f if line.strip())


def build_day_night_lookup(imagesets_dir, split):
    """
    'split' = 'train' veya 'test'. imageSets/*.txt dosyalarındaki
    day/night listelerinden {frame_id: 'day'/'night'} sözlüğü kurar.
    Frame id formatı: 'set00/V000/I00980' (uzantısız).
    train için 02 (her 2. frame), test için 20 (her 20. frame) örnekleme dosyaları kullanılır -
    bunlar KAIST'in standart deney protokolüdür (Gün 4'te belirlediğimiz gibi).
    """
    if split == "train":
        day_file, night_file = "train-day-02.txt", "train-night-02.txt"
    else:
        day_file, night_file = "test-day-20.txt", "test-night-20.txt"

    day_ids = load_image_list(imagesets_dir, day_file)
    night_ids = load_image_list(imagesets_dir, night_file)

    lookup = {}
    for fid in day_ids:
        lookup[fid] = "day"
    for fid in night_ids:
        lookup[fid] = "night"
    return lookup


# ---------------------------------------------------------------------------
# 2. XML annotation okuma
# ---------------------------------------------------------------------------

def load_annotation_xml(xml_path):
    """
    Doğrulanmış gerçek KAIST XML şemasını okur:
      <object><name>..</name><bndbox><x>,<y>,<w>,<h></bndbox><occlusion>..</occlusion></object>
    """
    boxes = []
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return boxes
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
        occlusion = int(occ_el.text) if occ_el is not None and occ_el.text is not None else 0
        boxes.append({
            "label": name_el.text.strip() if name_el.text else "unknown",
            "x": x, "y": y, "w": w, "h": h,
            "occlusion": occlusion,
        })
    return boxes


# ---------------------------------------------------------------------------
# 3. Ana analiz
# ---------------------------------------------------------------------------

def run_eda(data_root, split, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    ann_root = os.path.join(data_root, "annotations-xml-new-sanitized")
    imagesets_dir = os.path.join(data_root, "imageSets")

    if not os.path.isdir(ann_root):
        raise FileNotFoundError(f"'{ann_root}' bulunamadı. --data_root parametresini kontrol et.")
    if not os.path.isdir(imagesets_dir):
        raise FileNotFoundError(f"'{imagesets_dir}' bulunamadı. --data_root parametresini kontrol et.")

    day_night_lookup = build_day_night_lookup(imagesets_dir, split)
    print(f"[{split}] imageSets'ten toplam {len(day_night_lookup)} frame day/night olarak etiketlendi.")
    if len(day_night_lookup) == 0:
        print("UYARI: day/night listesi boş çıktı - imageSets dosya adlarını kontrol et.")
        return

    stats = {"day": defaultdict(int), "night": defaultdict(int)}
    heights = {"day": [], "night": []}
    occlusion_counts = {"day": defaultdict(int), "night": defaultdict(int)}
    images_with_person = {"day": 0, "night": 0}
    total_images = {"day": 0, "night": 0}
    label_counts = {"day": defaultdict(int), "night": defaultdict(int)}  # person/people/cyclist/person? dağılımı
    missing_xml = 0

    # Reasonable protokolüne göre sadece 'person' etiketi değerlendirmeye girer
    # (cyclist, people, person? Reasonable'da ignore edilir). Bunu ayrı raporluyoruz.
    EVAL_LABEL = "person"

    for frame_id, dn in day_night_lookup.items():
        total_images[dn] += 1
        xml_path = os.path.join(ann_root, frame_id + ".xml")
        if not os.path.isfile(xml_path):
            missing_xml += 1
            continue

        boxes = load_annotation_xml(xml_path)
        for b in boxes:
            label_counts[dn][b["label"]] += 1

        person_boxes = [b for b in boxes if b["label"] == EVAL_LABEL]
        if person_boxes:
            images_with_person[dn] += 1

        for b in person_boxes:
            stats[dn]["num_annotations"] += 1
            heights[dn].append(b["h"])
            occlusion_counts[dn][b["occlusion"]] += 1

    if missing_xml:
        print(f"UYARI: {missing_xml} frame için XML dosyası bulunamadı (atlandı).")

    # --- Özet tablo yazdır ---
    print("\n=== ÖZET ===")
    for dn in ["day", "night"]:
        n_img = total_images[dn]
        n_ann = stats[dn]["num_annotations"]
        n_img_with_person = images_with_person[dn]
        avg_h = np.mean(heights[dn]) if heights[dn] else 0
        print(f"\n[{dn.upper()}]")
        print(f"  Görüntü sayısı           : {n_img}")
        print(f"  Etiket dağılımı (tüm obj): {dict(label_counts[dn])}")
        print(f"  Yaya (person) içeren gör.: {n_img_with_person}")
        print(f"  Toplam 'person' annot.   : {n_ann}  (Reasonable protokolü sadece bu etiketi kullanır)")
        print(f"  Ortalama bbox yüksekliği : {avg_h:.1f}px")
        if heights[dn]:
            small = sum(1 for h in heights[dn] if h < 55)   # KAIST'te tipik "far/small" eşiği ~55px
            medium = sum(1 for h in heights[dn] if 55 <= h < 115)
            large = sum(1 for h in heights[dn] if h >= 115)
            print(f"  Küçük (<55px)            : {small} ({100*small/n_ann:.1f}%)")
            print(f"  Orta (55-115px)          : {medium} ({100*medium/n_ann:.1f}%)")
            print(f"  Büyük (>=115px)          : {large} ({100*large/n_ann:.1f}%)")
        occ = occlusion_counts[dn]
        total_occ = sum(occ.values())
        if total_occ:
            for level in sorted(occ.keys()):
                pct = 100 * occ[level] / total_occ
                print(f"  Occlusion={level}              : {occ[level]} ({pct:.1f}%)")

    # --- Grafikler ---
    # 1) Day/Night annotation sayısı
    plt.figure(figsize=(6, 4))
    plt.bar(["Day", "Night"],
            [stats["day"]["num_annotations"], stats["night"]["num_annotations"]],
            color=["#f4a261", "#264653"])
    plt.title(f"Annotation Sayısı: Day vs Night ({split} split)")
    plt.ylabel("Annotation sayısı")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "day_night_distribution.png"), dpi=150)
    plt.close()

    # 2) BBox yükseklik histogramı (day vs night üst üste)
    plt.figure(figsize=(7, 4))
    bins = np.linspace(0, 300, 40)
    if heights["day"]:
        plt.hist(heights["day"], bins=bins, alpha=0.6, label="Day", color="#f4a261")
    if heights["night"]:
        plt.hist(heights["night"], bins=bins, alpha=0.6, label="Night", color="#264653")
    plt.axvline(55, color="gray", linestyle="--", linewidth=1)
    plt.title(f"Bounding Box Yükseklik Dağılımı ({split} split, label=person)")
    plt.xlabel("Yükseklik (px)")
    plt.ylabel("Frekans")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bbox_height_histogram.png"), dpi=150)
    plt.close()

    # 3) Occlusion dağılımı
    all_levels = sorted(set(list(occlusion_counts["day"].keys()) + list(occlusion_counts["night"].keys())))
    day_vals = [occlusion_counts["day"].get(lvl, 0) for lvl in all_levels]
    night_vals = [occlusion_counts["night"].get(lvl, 0) for lvl in all_levels]
    x = np.arange(len(all_levels))
    width = 0.35
    plt.figure(figsize=(6, 4))
    plt.bar(x - width/2, day_vals, width, label="Day", color="#f4a261")
    plt.bar(x + width/2, night_vals, width, label="Night", color="#264653")
    plt.xticks(x, [f"Level {l}" for l in all_levels])
    plt.title("Occlusion Dağılımı")
    plt.ylabel("Annotation sayısı")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "occlusion_distribution.png"), dpi=150)
    plt.close()

    print(f"\nGrafikler kaydedildi: {output_dir}/")
    print("  - day_night_distribution.png")
    print("  - bbox_height_histogram.png")
    print("  - occlusion_distribution.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KAIST dataset Gün 5 EDA scripti")
    parser.add_argument("--data_root", type=str, required=True,
                         help="Veri setinin kök klasörü, örn: C:\\kaist-project\\data\\kaist-cvpr15 "
                              "(içinde 'annotations-xml-new-sanitized/' ve 'imageSets/' olmalı)")
    parser.add_argument("--split", type=str, choices=["train", "test"], default="test",
                         help="Hangi split analiz edilsin: 'train' (train-day/night-02) veya 'test' (test-day/night-20)")
    parser.add_argument("--output_dir", type=str, default="./eda_results",
                         help="Grafiklerin kaydedileceği klasör")
    args = parser.parse_args()

    run_eda(args.data_root, args.split, args.output_dir)
