import os
import argparse
import random
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from day6_dataset import KAISTPedestrianDataset, load_annotation_xml
from day7_evaluator import split_eval_ignore_boxes, evaluate_dataset, compute_precision_recall_ap

# for making test #
# venv2\Scripts\python.exe scripts\day14_test_early_fusion.py --data_root C:\Users\acer5\OneDrive\Desktop\kaist-project\data\kaist-cvpr15 --checkpoint C:\Users\acer5\OneDrive\Desktop\kaist-project\checkpoints\early_fusion_mobilenet_epoch3.pth --max_images 2252
def build_early_fusion_model(num_classes=2, backbone_name="mobilenet"):
    """day13_early_fusion.py'deki fonksiyonun ayni kopyasi - model mimarisi
    egitim ve test'te BIREBIR ayni olmali, yoksa agirliklar yuklenemez."""
    if backbone_name == "mobilenet":
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None)
        old_conv = model.backbone.body['0'][0]
    else:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
        old_conv = model.backbone.body.conv1

    new_conv = nn.Conv2d(4, old_conv.out_channels,
                         kernel_size=old_conv.kernel_size,
                         stride=old_conv.stride,
                         padding=old_conv.padding,
                         bias=False)
    if backbone_name == "mobilenet":
        model.backbone.body['0'][0] = new_conv
    else:
        model.backbone.body.conv1 = new_conv

    model.transform.image_mean = [0.485, 0.456, 0.406, 0.5]
    model.transform.image_std = [0.229, 0.224, 0.225, 0.5]

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def make_fusion_image(visible_img, lwir_img):
    """day13'teki EarlyFusionWrapper ile birebir ayni mantik: thermal'in
    sadece 1 kanalini al, RGB'nin 3 kanaliyla birlestir -> [4, H, W]."""
    if lwir_img.shape[0] == 3:
        lwir_img = lwir_img[0:1, :, :]
    return torch.cat((visible_img, lwir_img), dim=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gun 14: Early Fusion Model Day/Night Evaluation")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--backbone", type=str, choices=["resnet50", "mobilenet"], default="mobilenet")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_images", type=int, default=2252)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanilan cihaz: {device}")

    print(f"Model yukleniyor... ({args.checkpoint})")
    model = build_early_fusion_model(backbone_name=args.backbone)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    dataset = KAISTPedestrianDataset(args.data_root, split=args.split, only_reasonable=False)

    random.seed(42)
    all_ids = dataset.frame_ids
    if args.max_images < len(all_ids):
        frame_ids = random.sample(all_ids, args.max_images)
    else:
        frame_ids = all_ids
    print(f"Toplam test frame: {len(all_ids)} | Degerlendirilecek: {len(frame_ids)}")

    frame_id_to_idx = {fid: i for i, fid in enumerate(dataset.frame_ids)}

    day_dets, day_eval, day_ignore = [], [], []
    night_dets, night_eval, night_ignore = [], [], []

    print(f"Test basliyor ({len(frame_ids)} goruntu isleniyor)...")
    with torch.no_grad():
        for i, frame_id in enumerate(frame_ids):
            ds_idx = frame_id_to_idx[frame_id]
            vis_img, lwir_img, _ = dataset[ds_idx]

            fusion_img = make_fusion_image(vis_img, lwir_img).to(device)

            output = model([fusion_img])[0]
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

    print("\n" + "=" * 50)
    print("GUN 14 - EARLY FUSION MODEL DEGERLENDIRMESI (DAY vs NIGHT)")
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