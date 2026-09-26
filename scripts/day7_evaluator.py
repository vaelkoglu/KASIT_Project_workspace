"""
Gun 7 - KAIST Pedestrian Detection Evaluator
==============================================
MR^-2 (log-average miss rate), FPPI, Precision, Recall ve AP hesaplayan
degerlendirme modulu. KAIST/Caltech Reasonable protokolune uygun sekilde
tasarlandi (Gun 4'te belirledigimiz kriterler).

Iki tur kutu var:
  - "eval" kutulari : Reasonable kriterine uyan (person, occlusion<2, h>=55px)
                       -> bunlar gercek degerlendirmeye girer
  - "ignore" kutulari: person disi etiketler (cyclist, people, person?) veya
                       Reasonable disina dusen person kutulari (cok kucuk/cok
                       kapali) -> bu bolgelere denk gelen detection'lar ne TP
                       ne de FP sayilir, sadece yok sayilir (goz ardi edilir).
                       Bu, modelin "ignore" bolgelerini yanlislikla FP olarak
                       cezalandirmamak icindir - standart protokol boyle.

Kullanim (dogrudan calistirilirsa, ground-truth'u "mukemmel detection" olarak
kullanip evaluator'i test eder):
  python day7_evaluator.py --data_root C:\\kaist-project\\data\\kaist-cvpr15 --split test --max_images 300
"""

import os
import argparse
import random
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from day6_dataset import KAISTPedestrianDataset, load_annotation_xml  # noqa: E402


# ---------------------------------------------------------------------------
# 1. IoU hesaplama
# ---------------------------------------------------------------------------

def compute_iou(box_a, box_b):
    """box format: [x_min, y_min, x_max, y_max]"""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


# ---------------------------------------------------------------------------
# 2. Ground-truth'u eval / ignore olarak ayirma (Reasonable protokolu)
# ---------------------------------------------------------------------------

def split_eval_ignore_boxes(raw_boxes):
    """
    raw_boxes: day6_dataset.load_annotation_xml() ciktisi (x,y,w,h formatinda)
    Doner: eval_boxes, ignore_boxes -> her ikisi de [x_min,y_min,x_max,y_max] listesi
    """
    eval_boxes, ignore_boxes = [], []
    for b in raw_boxes:
        x1, y1 = b["x"], b["y"]
        x2, y2 = b["x"] + b["w"], b["y"] + b["h"]
        if b["label"] == "person" and b["occlusion"] < 2 and b["h"] >= 55:
            eval_boxes.append([x1, y1, x2, y2])
        else:
            # cyclist / people / person? / heavy-occluded / cok kucuk person -> ignore
            ignore_boxes.append([x1, y1, x2, y2])
    return eval_boxes, ignore_boxes


# ---------------------------------------------------------------------------
# 3. Tek goruntu icin eslestirme (matching)
# ---------------------------------------------------------------------------

def match_detections(detections, eval_boxes, ignore_boxes, iou_thresh=0.5):
    """
    detections: [(box[x1,y1,x2,y2], score), ...] -- skora gore ONCEDEN azalan
                sirada siralanmis olmali (yuksek guven once degerlendirilir)
    Doner: her detection icin bir etiket listesi: 'tp', 'fp', veya 'ignore'
           ve hangi gt kutularinin yakalandigini (matched_gt) belirten bir set.
    """
    gt_matched = [False] * len(eval_boxes)
    labels = []

    for box, score in detections:
        best_iou = 0.0
        best_idx = -1
        for i, gt in enumerate(eval_boxes):
            if gt_matched[i]:
                continue
            iou = compute_iou(box, gt)
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_iou >= iou_thresh:
            gt_matched[best_idx] = True
            labels.append("tp")
            continue

        # eval kutusuyla eslesmedi -> ignore bolgesine dusuyor mu diye bak
        is_ignored = False
        for ig in ignore_boxes:
            if compute_iou(box, ig) >= iou_thresh:
                is_ignored = True
                break

        labels.append("ignore" if is_ignored else "fp")

    num_missed = gt_matched.count(False)
    return labels, num_missed


# ---------------------------------------------------------------------------
# 4. Tum dataset uzerinde MR-FPPI egrisi + log-average miss rate
# ---------------------------------------------------------------------------

def evaluate_dataset(all_detections, all_eval_boxes, all_ignore_boxes, num_images, iou_thresh=0.5):
    """
    all_detections   : liste (goruntu basina) [(box, score), ...]
    all_eval_boxes   : liste (goruntu basina) [box, ...]  (Reasonable ground truth)
    all_ignore_boxes : liste (goruntu basina) [box, ...]

    Skor esigini degistirerek MR vs FPPI egrisini olusturur, sonra
    9 noktali log-average miss rate (MR^-2) hesaplar - standart Caltech/KAIST protokolu.
    """
    total_gt = sum(len(b) for b in all_eval_boxes)

    # Tum detection skorlarini topla, esik adaylari olarak kullan
    all_scores = sorted(set(s for dets in all_detections for _, s in dets), reverse=True)
    if not all_scores:
        print("UYARI: hic detection yok, MR-FPPI hesaplanamiyor.")
        return None

    fppi_list, mr_list = [], []

    for thresh in all_scores:
        total_tp, total_fp, total_missed = 0, 0, 0
        for dets, eval_boxes, ignore_boxes in zip(all_detections, all_eval_boxes, all_ignore_boxes):
            filtered = [(b, s) for b, s in dets if s >= thresh]
            filtered.sort(key=lambda x: -x[1])
            labels, num_missed = match_detections(filtered, eval_boxes, ignore_boxes, iou_thresh)
            total_tp += labels.count("tp")
            total_fp += labels.count("fp")
            total_missed += num_missed

        fppi = total_fp / num_images
        miss_rate = total_missed / total_gt if total_gt > 0 else 0.0
        fppi_list.append(fppi)
        mr_list.append(miss_rate)

    fppi_arr = np.array(fppi_list)
    mr_arr = np.array(mr_list)

    log_avg_mr = log_average_miss_rate(mr_arr, fppi_arr)

    return {
        "fppi": fppi_arr,
        "miss_rate": mr_arr,
        "log_average_miss_rate": log_avg_mr,
        "total_gt": total_gt,
    }


def log_average_miss_rate(mr_arr, fppi_arr):
    """
    Standart Caltech/KAIST protokolu: FPPI ekseninde [1e-2, 1e0] araliginda
    logaritmik olarak esit araliklanmis 9 nokta al, her noktada (o FPPI'ya en
    yakin, kucuk esitsizlikte) miss rate degerini bul, 9 degerin log-ortalamasini al.
    """
    ref_fppi = np.logspace(-2, 0, 9)

    # fppi azalan sirada olmali (esik azaldikca fppi artar) - once sirala
    order = np.argsort(fppi_arr)
    fppi_sorted = fppi_arr[order]
    mr_sorted = mr_arr[order]

    interpolated_mr = []
    for rf in ref_fppi:
        idx = np.searchsorted(fppi_sorted, rf, side="left")
        if idx == 0:
            mr_val = mr_sorted[0]
        elif idx >= len(fppi_sorted):
            mr_val = mr_sorted[-1]
        else:
            mr_val = mr_sorted[idx]
        # miss rate 0 olursa log(0) patlar, kucuk epsilon ile sinirla
        mr_val = max(mr_val, 1e-10)
        interpolated_mr.append(mr_val)

    log_avg = np.exp(np.mean(np.log(interpolated_mr)))
    return log_avg


# ---------------------------------------------------------------------------
# 5. Precision / Recall / AP (bonus - genel object detection metrikleri)
# ---------------------------------------------------------------------------

def compute_precision_recall_ap(all_detections, all_eval_boxes, all_ignore_boxes, iou_thresh=0.5):
    """VOC tarzi 11-nokta olmayan (alan altindaki) AP hesaplama."""
    total_gt = sum(len(b) for b in all_eval_boxes)

    scored = []  # (score, image_idx, box)
    for img_idx, dets in enumerate(all_detections):
        for box, score in dets:
            scored.append((score, img_idx, box))
    scored.sort(key=lambda x: -x[0])

    gt_matched = [[False] * len(b) for b in all_eval_boxes]
    tps, fps = [], []

    for score, img_idx, box in scored:
        eval_boxes = all_eval_boxes[img_idx]
        ignore_boxes = all_ignore_boxes[img_idx]

        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(eval_boxes):
            if gt_matched[img_idx][j]:
                continue
            iou = compute_iou(box, gt)
            if iou > best_iou:
                best_iou, best_j = iou, j

        if best_iou >= iou_thresh:
            gt_matched[img_idx][best_j] = True
            tps.append(1); fps.append(0)
            continue

        is_ignored = any(compute_iou(box, ig) >= iou_thresh for ig in ignore_boxes)
        if is_ignored:
            continue  # ne tp ne fp

        tps.append(0); fps.append(1)

    tps = np.cumsum(tps)
    fps = np.cumsum(fps)
    recalls = tps / max(total_gt, 1)
    precisions = tps / np.maximum(tps + fps, 1e-10)

    # alan altindaki AP (recall'e gore siralanmis precision zarfi)
    ap = 0.0
    for r in np.linspace(0, 1, 101):
        prec_at_r = precisions[recalls >= r].max() if np.any(recalls >= r) else 0.0
        ap += prec_at_r / 101

    return {"precision": precisions, "recall": recalls, "ap": ap}


# ---------------------------------------------------------------------------
# 6. Hizli test: ground-truth'u "mukemmel model" gibi kullanarak dogrulama
# ---------------------------------------------------------------------------

def make_synthetic_detections(raw_boxes, eval_boxes, drop_prob=0.0, jitter=0.0, add_fp_prob=0.0, img_w=640, img_h=512):
    """
    Gercek bir model henuz yok (Gun 8'de geliyor), o yuzden evaluator'i
    ground-truth uzerinden turetilmis "sahte" detection'larla test ediyoruz:
      drop_prob=0, jitter=0, add_fp_prob=0 -> mukemmel model (MR ~ 0 beklenir)
      drop_prob>0 -> bazi yayalari kacirir (miss rate artar)
      add_fp_prob>0 -> rastgele yanlis kutular ekler (FPPI artar)
    """
    detections = []
    for box in eval_boxes:
        if random.random() < drop_prob:
            continue
        x1, y1, x2, y2 = box
        if jitter > 0:
            dx = random.uniform(-jitter, jitter)
            dy = random.uniform(-jitter, jitter)
            x1, y1, x2, y2 = x1 + dx, y1 + dy, x2 + dx, y2 + dy
        score = random.uniform(0.7, 1.0)
        detections.append(([x1, y1, x2, y2], score))

    if random.random() < add_fp_prob:
        fx = random.uniform(0, img_w - 50)
        fy = random.uniform(0, img_h - 50)
        detections.append(([fx, fy, fx + 30, fy + 80], random.uniform(0.3, 0.6)))

    return detections


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gun 7 evaluator hizli test")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_images", type=int, default=300,
                         help="Hiz icin sadece ilk N goruntuyu kullan (tum test seti 2252 goruntu)")
    args = parser.parse_args()

    random.seed(42)

    dataset = KAISTPedestrianDataset(args.data_root, split=args.split, only_reasonable=False)
    # NOT: test-all-20.txt seteye gore sirali olabilir (once tum day set'leri,
    # sonra night). max_images ile listeyi kesmek gece verisini tamamen
    # atlamana neden olabilir. Bu yuzden ONCE random.sample ile dengeli bir
    # ornek al, SONRA kes - boylece hem day hem night garanti temsil edilir.
    import random
    random.seed(42)
    all_ids = dataset.frame_ids
    if args.max_images < len(all_ids):
        frame_ids = random.sample(all_ids, args.max_images)
    else:
        frame_ids = all_ids

    all_detections, all_eval_boxes, all_ignore_boxes = [], [], []

    for frame_id in frame_ids:
        set_name, video_name, image_name = frame_id.split("/")
        xml_path = os.path.join(dataset.ann_root, set_name, video_name, image_name + ".xml")
        raw_boxes = load_annotation_xml(xml_path)
        eval_boxes, ignore_boxes = split_eval_ignore_boxes(raw_boxes)

        # "mukemmel model" senaryosu: gt = detection (kucuk jitter ile gercekci)
        dets = make_synthetic_detections(raw_boxes, eval_boxes, drop_prob=0.0, jitter=1.5, add_fp_prob=0.05)

        all_detections.append(dets)
        all_eval_boxes.append(eval_boxes)
        all_ignore_boxes.append(ignore_boxes)

    print(f"Test edilen goruntu sayisi: {len(frame_ids)}")
    print(f"Toplam Reasonable ground-truth kutu: {sum(len(b) for b in all_eval_boxes)}")

    result = evaluate_dataset(all_detections, all_eval_boxes, all_ignore_boxes, num_images=len(frame_ids))
    if result:
        print(f"\n[Neredeyse mukemmel model senaryosu - sanity check]")
        print(f"  Log-Average Miss Rate (MR^-2) : {result['log_average_miss_rate']*100:.2f}%")
        print(f"  (Bu deger dusuk cikmali, cunku detection'lar gt'den turetildi)")

    pr_result = compute_precision_recall_ap(all_detections, all_eval_boxes, all_ignore_boxes)
    print(f"\n  AP (IoU=0.5) : {pr_result['ap']*100:.2f}%")

    print("\nEvaluator basariyla calisti. Gun 8'den itibaren gercek model")
    print("ciktilarini 'all_detections' formatinda ([box, score]) besleyerek")
    print("bu ayni fonksiyonlari kullanabilirsin.")