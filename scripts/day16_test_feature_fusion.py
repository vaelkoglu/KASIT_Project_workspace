import os
import argparse
import random
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
# venv2\Scripts\python.exe scripts\day16_test_feature_fusion.py --data_root C:\Users\acer5\OneDrive\Desktop\kaist-project\data\kaist-cvpr15 --checkpoint C:\Users\acer5\OneDrive\Desktop\kaist-project\checkpoints\feature_fusion_resnet50_epoch1.pth --backbone resnet50 --max_images 2252
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from day6_dataset import KAISTPedestrianDataset, load_annotation_xml
from day7_evaluator import split_eval_ignore_boxes, evaluate_dataset, compute_precision_recall_ap


class DualStreamFPNBackbone(nn.Module):
    """day15_train_feature_fusion.py'deki sinifin ayni kopyasi - mimari
    egitim ve test'te BIREBIR ayni olmali, yoksa agirliklar yuklenemez."""
    def __init__(self, backbone_name="mobilenet"):
        super().__init__()
        if backbone_name == "mobilenet":
            rgb_full = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None)
            thermal_full = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None)
        else:
            rgb_full = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
            thermal_full = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)

        self.rgb_backbone = rgb_full.backbone
        self.thermal_backbone = thermal_full.backbone
        self.out_channels = self.rgb_backbone.out_channels

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
    return model


def make_fusion_image(visible_img, lwir_img):
    """day15'teki FeatureFusionWrapper ile birebir ayni mantik: RGB (3ch) +
    Thermal duplicate (3ch) -> [6, H, W]."""
    return torch.cat((visible_img, lwir_img), dim=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gun 16: Feature Fusion Model Day/Night Evaluation")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--backbone", type=str, choices=["resnet50", "mobilenet"], default="mobilenet")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_images", type=int, default=2252)
    parser.add_argument("--image_size", type=int, nargs=2, default=None,
                         help="ONERILMEZ (kullanma): test sirasinda goruntuyu kucultmek, "
                              "raw XML koordinatlarindaki ground-truth kutularla olcek "
                              "uyusmazligina yol acar (tum kutular yanlislikla 'kacirilmis' "
                              "gorunur). Faster R-CNN herhangi bir giris boyutunu otomatik "
                              "isleyebildigi icin bu parametreyi bos birak.")
    args = parser.parse_args()

    if args.image_size:
        print("UYARI: --image_size test asamasinda kullanilmamali (ground-truth kutularla "
              "olcek uyusmazligi yaratir). Bu deger YOK SAYILACAK, orijinal (640x512) "
              "boyut kullanilacak.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanilan cihaz: {device}")

    print(f"Model yukleniyor... ({args.checkpoint})")
    model = build_feature_fusion_model(backbone_name=args.backbone)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # ONEMLI: image_size KASITLI OLARAK None birakiliyor - goruntuler orijinal
    # (640x512) olcekte yukleniyor, boylece model ciktisi ile ground-truth
    # kutular (raw XML'den okunan) AYNI koordinat sisteminde kaliyor.
    dataset = KAISTPedestrianDataset(args.data_root, split=args.split, only_reasonable=False, image_size=None)

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
    print("GUN 16 - FEATURE FUSION MODEL DEGERLENDIRMESI (DAY vs NIGHT)")
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