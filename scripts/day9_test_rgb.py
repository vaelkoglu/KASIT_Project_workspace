import os
import argparse
import random
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# venv2\Scripts\python.exe scripts\day9_test_rgb.py --data_root C:\Users\acer5\OneDrive\Desktop\kaist-project\data\kaist-cvpr15 --checkpoint C:\Users\acer5\OneDrive\Desktop\kaist-project\checkpoints\rgb_baseline_epoch3_batch300.pth --max_images 2252
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from day6_dataset import KAISTPedestrianDataset, load_annotation_xml
from day7_evaluator import split_eval_ignore_boxes, evaluate_dataset, compute_precision_recall_ap

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day 9: RGB Model Day/Night Evaluation")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_images", type=int, default=2252,
                         help="Test edilecek maksimum goruntu. Tum test seti 2252 - "
                              "kucultursen dengeli day/night ornekleme icin random.sample kullanilir.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanilan cihaz: {device}")

    # 1. Model ve agirliklarin yuklenmesi (Gun 8'de mobilenet kullandik)
    print(f"Model yukleniyor... ({args.checkpoint})")
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    dataset = KAISTPedestrianDataset(args.data_root, split=args.split, only_reasonable=False)

    # ONEMLI DUZELTME: test-all-20.txt set'e gore sirali olabilir (once tum
    # day set'leri, sonra night). Listeyi baştan kesmek (frame_ids[:N]) gece
    # verisini tamamen atlamana neden olabilirdi. Bunun yerine ONCE dengeli
    # rastgele bir ornek al, SONRA isle - boylece max_images kucuk olsa bile
    # hem day hem night temsil edilir.
    random.seed(42)
    all_ids = dataset.frame_ids
    if args.max_images < len(all_ids):
        frame_ids = random.sample(all_ids, args.max_images)
    else:
        frame_ids = all_ids
    print(f"Toplam test frame: {len(all_ids)} | Degerlendirilecek: {len(frame_ids)}")

    # Kac gorseli manuel indeksle degil, frame_id uzerinden dogrudan cekmek
    # icin bir yardimci fonksiyon kullaniyoruz (frame_ids karistirildigi icin
    # dataset[i] ile frame_ids[i] artik birbirine denk gelmeyebilir).
    frame_id_to_idx = {fid: i for i, fid in enumerate(dataset.frame_ids)}

    # Gun 5'te dogrulanan ayni day/night atamasini (imageSets tabanli) kullan -
    # set adindan tahmin etmek yerine dataset.day_night sozlugune guveniyoruz,
    # cunku bu zaten resmi sayilarla (1455 day / 797 night) dogrulanmisti.
    day_dets, day_eval, day_ignore = [], [], []
    night_dets, night_eval, night_ignore = [], [], []

    print(f"Test basliyor ({len(frame_ids)} goruntu isleniyor)...")
    with torch.no_grad():
        for i, frame_id in enumerate(frame_ids):
            ds_idx = frame_id_to_idx[frame_id]
            vis_img, _, _ = dataset[ds_idx]
            vis_img = vis_img.to(device)

            output = model([vis_img])[0]
            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()

            dets = [(b.tolist(), float(s)) for b, s, l in zip(boxes, scores, labels) if l == 1]

            set_name, video_name, image_name = frame_id.split("/")
            xml_path = os.path.join(dataset.ann_root, set_name, video_name, image_name + ".xml")
            raw_boxes = load_annotation_xml(xml_path)
            eval_boxes, ignore_boxes = split_eval_ignore_boxes(raw_boxes)

            day_night = dataset.day_night.get(frame_id, "unknown")

            if day_night == "day":
                day_dets.append(dets); day_eval.append(eval_boxes); day_ignore.append(ignore_boxes)
            elif day_night == "night":
                night_dets.append(dets); night_eval.append(eval_boxes); night_ignore.append(ignore_boxes)

            if (i + 1) % 100 == 0:
                print(f"  [{i + 1}/{len(frame_ids)}] islendi...")

    # 2. Sonuclari hesapla ve tabloyu yazdir
    print("\n" + "=" * 50)
    print("GUN 9 - RGB MODEL DEGERLENDIRMESI (DAY vs NIGHT)")
    print("=" * 50)

    def print_metrics(name, dets, evals, ignores):
        if len(dets) == 0:
            print(f"{name} verisi bulunamadi.")
            return
        res = evaluate_dataset(dets, evals, ignores, num_images=len(dets))
        ap_res = compute_precision_recall_ap(dets, evals, ignores)
        mr = res['log_average_miss_rate'] * 100 if res else 100.0
        ap = ap_res['ap'] * 100
        print(f"[{name.upper()}] Goruntu: {len(dets)} | Ground-Truth: {sum(len(b) for b in evals)}")
        print(f"  -> MR (Miss Rate) : %{mr:.2f} (Dusuk daha iyi)")
        print(f"  -> AP (Precision) : %{ap:.2f} (Yuksek daha iyi)\n")

    print_metrics("Tum Zamanlar (All)", day_dets + night_dets, day_eval + night_eval, day_ignore + night_ignore)
    print_metrics("Gunduz (Day)", day_dets, day_eval, day_ignore)
    print_metrics("Gece (Night)", night_dets, night_eval, night_ignore)
    print("=" * 50)