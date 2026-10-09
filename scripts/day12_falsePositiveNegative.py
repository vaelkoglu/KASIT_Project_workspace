import os
import cv2
import torch
import warnings
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
import sys


sys.path.append(r'C:\Users\acer5\OneDrive\Desktop\kaist-project\scripts')

data_root = r"C:\Users\acer5\OneDrive\Desktop\kaist-project\data\kaist-cvpr15"
checkpoint_path = r"C:\Users\acer5\OneDrive\Desktop\kaist-project\checkpoints\rgb_baseline_epoch2.pth"

# 2. Veri seti ve model dosyalarını çağırma (Collate fonksiyonu eklendi)
from day6_dataset import KAISTPedestrianDataset
from day8_train_rgb import build_model, RGBOnlyWrapper, detection_collate_fn

# Gereksiz uyarı mesajlarını gizleme
warnings.filterwarnings("ignore", category=FutureWarning)


def analyze_failures(model, loader, device, max_failures=20):
    model.eval()
    fp_cases, fn_cases = [], []

    with torch.no_grad():
        for images, targets in loader:
            if len(fp_cases) >= max_failures and len(fn_cases) >= max_failures:
                break

            images = [img.to(device) for img in images]
            outputs = model(images)

            for img, target, out in zip(images, targets, outputs):
                # Görüntüyü göstermek için hazırlama
                img_disp = img.cpu().permute(1, 2, 0).numpy()
                img_disp = (img_disp * 255).clip(0, 255).astype(np.uint8)
                img_disp = np.ascontiguousarray(img_disp)

                # Artık target bir string değil, doğru bir sözlük (dictionary)
                gt_boxes = target['boxes'].cpu().numpy()
                pred_boxes = out['boxes'].cpu().numpy()
                scores = out['scores'].cpu().numpy()

                # Tahminleri filtreleme (Threshold = 0.5 or 0.3)scores > 0.5. هذا يعني أن النموذج لا يرسم المربع إلا إذا كان متأكداً بنسبة 50%. إ
                valid_preds = pred_boxes[scores > 0.3]

                # 1. Yanlış Negatif (FN): Gerçekte bir yaya var ama model tahmin edemedi
                if len(gt_boxes) > 0 and len(valid_preds) == 0:
                    fn_cases.append((img_disp, gt_boxes, valid_preds))

                # 2. Yanlış Pozitif (FP): Model boş bir alanda yaya olduğunu sanıyor
                elif len(gt_boxes) == 0 and len(valid_preds) > 0:
                    fp_cases.append((img_disp, gt_boxes, valid_preds))
    return fp_cases, fn_cases


def display_and_save_results(cases, title, filename):
    limit = min(len(cases), 20)
    fig, axes = plt.subplots(5, 4, figsize=(20, 25))
    fig.suptitle(title, fontsize=18, fontweight='bold')

    for idx, (img, gt, pred) in enumerate(cases[:limit]):
        ax = axes[idx // 4, idx % 4]
        display_img = img.copy()

        # Gerçek kutuları (Ground Truth) yeşil renkte çizme
        for box in gt:
            cv2.rectangle(display_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

        # Yanlış tahmin edilen kutuları (Prediction) kırmızı renkte çizme
        for box in pred:
            cv2.rectangle(display_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)

        ax.imshow(display_img)
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    # ---------------------------------------------------------
    # المشكلة 1 تم حلها: حفظ الصور كملف PNG قبل إغلاقها
    # ---------------------------------------------------------
    save_dir = r"C:\Users\acer5\OneDrive\Desktop\kaist-project\error_analysis_results"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=150)
    print(f"Görseller başarıyla kaydedildi: {save_path}")

    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test veri setini doğru yoldan yükleme
    test_ds = KAISTPedestrianDataset(data_root, split="test", only_reasonable=True)

    # ---------------------------------------------------------
    # اختيار 2000 صورة عشوائياً لضمان تنوع الليل والنهار
    # ---------------------------------------------------------
    total_images = len(test_ds)
    sample_size = min(2000, total_images)
    random_indices = random.sample(range(total_images), sample_size)
    test_ds = Subset(test_ds, random_indices)

    # collate_fn=detection_collate_fn parametresi eklendi
    test_loader = DataLoader(RGBOnlyWrapper(test_ds), batch_size=1, shuffle=False, collate_fn=detection_collate_fn)

    # Modeli MobileNet mimarisiyle başlatma
    model = build_model(backbone="mobilenet")

    if os.path.exists(checkpoint_path):
        # Ağırlıkları güvenli bir şekilde yükleme
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        print("Model ağırlıkları başarıyla yüklendi.")
    else:
        print("UYARI: Ağırlık dosyası bulunamadı.")

    model.to(device)
    print("Model hataları (Yanlış Pozitifler ve Yanlış Negatifler) aranıyor...")

    fp_cases, fn_cases = analyze_failures(model, test_loader, device)

    print(f"{len(fp_cases)} adet Yanlış Pozitif (False Positive) durumu bulundu.")
    if fp_cases:
        display_and_save_results(fp_cases, "False Positives (Kirmizi: Tahmin, Yesil: Gercek)",
                                 "false_positives_report.png")

    print(f"{len(fn_cases)} adet Yanlış Negatif (False Negative) durumu bulundu.")
    if fn_cases:
        display_and_save_results(fn_cases, "False Negatives (Kirmizi: Tahmin, Yesil: Gercek)",
                                 "false_negatives_report.png")