"""
Gün 6 (bölüm 1) - KAIST Dataset: Tek bir frame'i annotation'larıyla görselleştirme
====================================================================================
Bu script, verilen bir frame_id (örn. 'set00/V000/I00980') için:
  - visible (RGB) ve lwir (thermal) görüntülerini yan yana açar
  - annotations-xml-new-sanitized'daki bounding box'ları her iki görüntü üzerine çizer
  - sonucu bir PNG olarak kaydeder

Kullanım:
  python day6_visualize_bbox.py --data_root C:\\kaist-project\\data\\kaist-cvpr15 --frame_id set00/V000/I00980
  python day6_visualize_bbox.py --data_root C:\\kaist-project\\data\\kaist-cvpr15 --frame_id set00/V000/I00980 --output_dir ./vis_results

Label'a göre kutu rengi:
  person   -> yeşil   (Reasonable protokolünde değerlendirilen asıl sınıf)
  people   -> turuncu (grup halinde, ayırt edilemeyen kişiler)
  cyclist  -> sarı
  person?  -> kırmızı (belirsiz/etiketleyici emin değil)
"""

import os
import argparse
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


LABEL_COLORS = {
    "person": "#2ecc71",
    "people": "#e67e22",
    "cyclist": "#f1c40f",
    "person?": "#e74c3c",
}


def load_annotation_xml(xml_path):
    """Gün 5'teki fonksiyonla aynı: gerçek KAIST XML şemasını okur."""
    boxes = []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for obj in root.findall("object"):
        name_el = obj.find("name")
        bnd = obj.find("bndbox")
        occ_el = obj.find("occlusion")
        if name_el is None or bnd is None:
            continue
        x = float(bnd.find("x").text)
        y = float(bnd.find("y").text)
        w = float(bnd.find("w").text)
        h = float(bnd.find("h").text)
        occlusion = int(occ_el.text) if occ_el is not None and occ_el.text else 0
        boxes.append({
            "label": name_el.text.strip() if name_el.text else "unknown",
            "x": x, "y": y, "w": w, "h": h,
            "occlusion": occlusion,
        })
    return boxes


def draw_boxes_on_axis(ax, image, boxes, title):
    ax.imshow(image)
    ax.set_title(title, fontsize=11)
    ax.axis("off")

    for b in boxes:
        color = LABEL_COLORS.get(b["label"], "#ffffff")
        rect = patches.Rectangle(
            (b["x"], b["y"]), b["w"], b["h"],
            linewidth=2, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)
        occ_txt = {0: "no-occ", 1: "partial", 2: "heavy"}.get(b["occlusion"], "?")
        ax.text(
            b["x"], max(b["y"] - 4, 0),
            f"{b['label']} ({occ_txt})",
            color=color, fontsize=8, fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.5, pad=0, edgecolor="none"),
        )


def visualize_frame(data_root, frame_id, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    parts = frame_id.split("/")  # örn: ['set00', 'V000', 'I00980']
    if len(parts) != 3:
        raise ValueError(f"frame_id 'setXX/VYYY/IZZZZZ' formatında olmalı, alınan: {frame_id}")
    set_name, video_name, image_name = parts

    xml_path = os.path.join(data_root, "annotations-xml-new-sanitized", set_name, video_name, image_name + ".xml")
    visible_path = os.path.join(data_root, "images", set_name, video_name, "visible", image_name + ".jpg")
    lwir_path = os.path.join(data_root, "images", set_name, video_name, "lwir", image_name + ".jpg")

    for p, desc in [(xml_path, "annotation"), (visible_path, "visible görüntü"), (lwir_path, "lwir görüntü")]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"{desc} bulunamadı: {p}")

    boxes = load_annotation_xml(xml_path)
    print(f"'{frame_id}' için {len(boxes)} obje bulundu:")
    for b in boxes:
        print(f"  - {b['label']}: x={b['x']}, y={b['y']}, w={b['w']}, h={b['h']}, occlusion={b['occlusion']}")

    visible_img = Image.open(visible_path).convert("RGB")
    lwir_img = Image.open(lwir_path).convert("RGB")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    draw_boxes_on_axis(axes[0], visible_img, boxes, f"Visible (RGB) - {frame_id}")
    draw_boxes_on_axis(axes[1], lwir_img, boxes, f"LWIR (Thermal) - {frame_id}")
    plt.tight_layout()

    out_name = frame_id.replace("/", "_") + "_bbox.png"
    out_path = os.path.join(output_dir, out_name)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nKaydedildi: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tek bir KAIST frame'ini annotation'larıyla görselleştir")
    parser.add_argument("--data_root", type=str, required=True,
                         help="örn: C:\\kaist-project\\data\\kaist-cvpr15")
    parser.add_argument("--frame_id", type=str, required=True,
                         help="örn: set00/V000/I00980  (uzantısız, / ile)")
    parser.add_argument("--output_dir", type=str, default="./vis_results")
    args = parser.parse_args()

    visualize_frame(args.data_root, args.frame_id, args.output_dir)
