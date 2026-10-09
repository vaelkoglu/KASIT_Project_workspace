"""
Gun 24 - Nitel (Qualitative) Ornekler: Her model test setinde ne gordu?
=======================================================================
Her model icin test setinden 10 gunduz + 10 gece goruntusu secer, modeli
calistirir ve sonucu renklerle cizer:

  YESIL        = gercek yaya, model BULDU
  SARI (kesik) = gercek yaya, model KACIRDI (False Negative)
  CYAN         = modelin dogru tespiti (True Positive)
  KIRMIZI      = yanlis alarm (False Positive)

Secim modlari (--mode):
  random  : rastgele (ayni goruntuler TUM modellerde) -> ADIL karsilastirma
  success : modelin her seyi buldugu ve yanlis alarm vermedigi en iyi ornekler
  failure : modelin en cok hata yaptigi ornekler (FP/FN analizi)

DURUST NOT: 'success' modu en iyi ornekleri SECER (cherry-pick). Raporda mutlaka
"en iyi ornekler" diye etiketle; adil gorsel icin 'random' modunu da kullan.

Bu dosya day22_robustness.py'ye bagimlidir (model kurma/yukleme kodunu oradan kullanir),
ayni klasorde olmalidir (day6_dataset.py ve day7_evaluator.py de).

Ornek:
  python day24_qualitative_examples.py --data_root C:\\...\\data\\kaist-cvpr15 --mode success ^
     --rgb RGB=C:\\...\\rgb_baseline_epoch3_batch300.pth ^
     --models Adaptive=C:\\...\\adaptive_fusion_resnet50_epoch2.pth ^
     --out_dir C:\\...\\qualitative_results
"""

import os
import sys
import math
import random
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from day6_dataset import KAISTPedestrianDataset, load_annotation_xml  # noqa: E402
from day7_evaluator import compute_iou, split_eval_ignore_boxes  # noqa: E402
from day22_robustness import build_model_for, TestImages, LOADER_KIND, collate  # noqa: E402

C_FOUND, C_MISSED, C_TP, C_FP = "#2ecc71", "#f1c40f", "#00bcd4", "#e74c3c"


# ---------------------------------------------------------------------------
# 1. Havuz: GT'si (Reasonable yaya) olan, dengeli day/night goruntuler
# ---------------------------------------------------------------------------

def build_pool(base, pool_size, seed=42):
    rng = random.Random(seed)
    ids = list(base.frame_ids)
    rng.shuffle(ids)
    quota = {"day": pool_size // 2, "night": pool_size - pool_size // 2}
    counts = {"day": 0, "night": 0}
    pool = []
    for fid in ids:
        cond = base.day_night.get(fid, "unknown")
        if cond not in quota or counts[cond] >= quota[cond]:
            continue
        s, v, n = fid.split("/")
        raw = load_annotation_xml(os.path.join(base.ann_root, s, v, n + ".xml"))
        eval_b, ignore_b = split_eval_ignore_boxes(raw)
        if not eval_b:  # en az 1 Reasonable yaya olmali
            continue
        pool.append({"fid": fid, "cond": cond, "eval": eval_b, "ignore": ignore_b})
        counts[cond] += 1
        if counts["day"] >= quota["day"] and counts["night"] >= quota["night"]:
            break
    return pool


# ---------------------------------------------------------------------------
# 2. Eslestirme (Gun 7 evaluator ile ayni mantik: IoU>=0.5, ignore bolgeleri)
# ---------------------------------------------------------------------------

def match_frame(dets, eval_boxes, ignore_boxes, iou_thr=0.5):
    dets = sorted(dets, key=lambda d: -d[1])
    found = [False] * len(eval_boxes)
    tp, fp = [], []
    for box, _score in dets:
        best, bi = 0.0, -1
        for i, g in enumerate(eval_boxes):
            if found[i]:
                continue
            iou = compute_iou(box, g)
            if iou > best:
                best, bi = iou, i
        if best >= iou_thr:
            found[bi] = True
            tp.append(box)
        elif any(compute_iou(box, ig) >= iou_thr for ig in ignore_boxes):
            continue
        else:
            fp.append(box)
    return tp, fp, found


def select(results, mode, n):
    def errors(r):
        return (len(r["eval"]) - sum(r["found"])) + len(r["fp"])

    out = []
    for cond in ("day", "night"):
        rs = [r for r in results if r["cond"] == cond]
        if mode == "success":
            rs = [r for r in rs if all(r["found"]) and not r["fp"]]
            rs.sort(key=lambda r: -len(r["eval"]))  # cok yayali basarili sahneler once
        elif mode == "failure":
            rs = [r for r in rs if errors(r) > 0]
            rs.sort(key=lambda r: -errors(r))
        if len(rs) < n:
            print(f"    UYARI: {cond} icin {mode} kosuluna uyan sadece {len(rs)} goruntu var (istenen {n}). "
                  f"--pool degerini artirabilirsin.")
        out += rs[:n]
    return out


# ---------------------------------------------------------------------------
# 3. Cizim
# ---------------------------------------------------------------------------

def load_display_image(base, fid, cond, background):
    view = background if background != "auto" else ("visible" if cond == "day" else "thermal")
    img = base._load_image(fid, "visible" if view == "visible" else "lwir")
    return img.permute(1, 2, 0).numpy(), ("RGB" if view == "visible" else "Thermal")


def draw_boxes(ax, r, lw=2.0):
    for g, ok in zip(r["eval"], r["found"]):
        x1, y1, x2, y2 = g
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=lw,
                                       edgecolor=C_FOUND if ok else C_MISSED,
                                       linestyle="-" if ok else "--"))
    for b in r["tp"]:
        x1, y1, x2, y2 = b
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=1.0,
                                       edgecolor=C_TP))
    for b in r["fp"]:
        x1, y1, x2, y2 = b
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=lw,
                                       edgecolor=C_FP))


def legend_handles():
    return [patches.Patch(edgecolor=C_FOUND, facecolor="none", label="Gercek yaya - bulundu"),
            patches.Patch(edgecolor=C_MISSED, facecolor="none", linestyle="--", label="Gercek yaya - KACIRILDI (FN)"),
            patches.Patch(edgecolor=C_TP, facecolor="none", label="Dogru tespit (TP)"),
            patches.Patch(edgecolor=C_FP, facecolor="none", label="Yanlis alarm (FP)")]


def save_grid(base, sel, title, path, background, cols=5):
    n = len(sel)
    if n == 0:
        print("    Cizilecek goruntu yok, grid atlandi.")
        return
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.9 * rows), squeeze=False)
    axes = axes.reshape(-1)
    for ax, r in zip(axes, sel):
        img, view = load_display_image(base, r["fid"], r["cond"], background)
        ax.imshow(img)
        draw_boxes(ax, r)
        ax.set_title(f"{r['fid']}\n{r['cond']} ({view}) | GT {len(r['eval'])} | "
                     f"bulundu {sum(r['found'])} | FP {len(r['fp'])}", fontsize=8)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.legend(handles=legend_handles(), loc="lower center", ncol=4, fontsize=10)
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"    Kaydedildi: {path}")


def save_comparison(base, names, all_results, fids, path, background):
    nrows, ncols = len(fids), len(names)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.4 * nrows), squeeze=False)
    for i, fid in enumerate(fids):
        for j, name in enumerate(names):
            r = all_results[name][fid]
            ax = axes[i][j]
            img, view = load_display_image(base, fid, r["cond"], background)
            ax.imshow(img)
            draw_boxes(ax, r)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(f"bulundu {sum(r['found'])}/{len(r['eval'])} | FP {len(r['fp'])}", fontsize=9)
            if i == 0:
                ax.set_title(name, fontsize=12, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{r['cond']} ({view})\n{fid.split('/')[-1]}", fontsize=8)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=4, fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"Kaydedildi: {path}")


def print_stats(name, results, thr):
    print(f"  [{name}] havuz istatistigi (skor>={thr}) - kucuk orneklem, resmi MR degildir:")
    for cond in ("day", "night"):
        rs = [r for r in results if r["cond"] == cond]
        gt = sum(len(r["eval"]) for r in rs)
        fd = sum(sum(r["found"]) for r in rs)
        fp = sum(len(r["fp"]) for r in rs)
        print(f"     {cond:5s}: goruntu={len(rs):3d}  GT={gt:3d}  bulunan={fd:3d} ({100 * fd / max(gt, 1):.0f}%)  yanlis alarm={fp}")


# ---------------------------------------------------------------------------
# 4. Ana akis
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--rgb", nargs="*", default=[])
    ap.add_argument("--thermal", nargs="*", default=[])
    ap.add_argument("--early", nargs="*", default=[])
    ap.add_argument("--feature", nargs="*", default=[])
    ap.add_argument("--models", nargs="*", default=[], help="Adaptive / Robust (gate'li)")
    ap.add_argument("--rgb_backbone", choices=["resnet50", "mobilenet"], default="mobilenet")
    ap.add_argument("--thermal_backbone", choices=["resnet50", "mobilenet"], default="mobilenet")
    ap.add_argument("--early_backbone", choices=["resnet50", "mobilenet"], default="mobilenet")
    ap.add_argument("--feature_backbone", choices=["resnet50", "mobilenet"], default="resnet50")
    ap.add_argument("--backbone", choices=["resnet50", "mobilenet"], default="resnet50")
    ap.add_argument("--mode", choices=["random", "success", "failure"], default="random")
    ap.add_argument("--num_per_condition", type=int, default=10, help="gunduz ve gece icin ayri ayri")
    ap.add_argument("--pool", type=int, default=120, help="aday goruntu sayisi (yari gunduz, yari gece)")
    ap.add_argument("--score_thr", type=float, default=0.5)
    ap.add_argument("--background", choices=["auto", "visible", "thermal"], default="auto",
                    help="auto: gunduz=RGB, gece=Thermal (geceleyin RGB cok karanlik)")
    ap.add_argument("--compare_n", type=int, default=6, help="karsilastirma figurundeki satir sayisi (random modunda)")
    ap.add_argument("--out_dir", default="./qualitative_results")
    args = ap.parse_args()

    specs = ([("rgb", s) for s in args.rgb] + [("thermal", s) for s in args.thermal]
             + [("early", s) for s in args.early] + [("feature", s) for s in args.feature]
             + [("adaptive", s) for s in args.models])
    if not specs:
        sys.exit("En az bir model ver: --rgb / --thermal / --early / --feature / --models")

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")

    base = KAISTPedestrianDataset(args.data_root, split="test", only_reasonable=False)
    pool = build_pool(base, args.pool)
    print(f"Havuz: {len(pool)} goruntu (day={sum(p['cond'] == 'day' for p in pool)}, "
          f"night={sum(p['cond'] == 'night' for p in pool)}) - hepsinde en az 1 Reasonable yaya var")

    all_results, names = {}, []
    for kind, spec in specs:
        name, ckpt = spec.split("=", 1)
        print(f"\n=== Model: {name} [{kind}] ===")
        model = build_model_for(kind, args, ckpt)
        model.to(device)
        model.eval()
        loader = DataLoader(TestImages(base, [p["fid"] for p in pool], LOADER_KIND[kind]),
                            batch_size=1, shuffle=False, collate_fn=collate, num_workers=0)
        results = []
        with torch.no_grad():
            for i, (p, images) in enumerate(zip(pool, loader)):
                pred = model([im.to(device) for im in images])[0]
                dets = [(b, s) for b, s in zip(pred["boxes"].cpu().tolist(), pred["scores"].cpu().tolist())
                        if s >= args.score_thr]
                tp, fp, found = match_frame(dets, p["eval"], p["ignore"])
                results.append({**p, "tp": tp, "fp": fp, "found": found})
                if (i + 1) % 40 == 0:
                    print(f"    {i + 1}/{len(pool)}")

        print_stats(name, results, args.score_thr)
        sel = select(results, args.mode, args.num_per_condition)
        save_grid(base, sel,
                  f"{name} - {args.mode.upper()} ornekler ({args.num_per_condition} gunduz + {args.num_per_condition} gece, skor>={args.score_thr})",
                  os.path.join(args.out_dir, f"examples_{args.mode}_{name}.png"), args.background)
        all_results[name] = {r["fid"]: r for r in results}
        names.append(name)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Ayni goruntuler uzerinde yan yana karsilastirma (sadece random modunda adil)
    if args.mode == "random" and len(names) >= 2:
        half = max(args.compare_n // 2, 1)
        day_f = [p["fid"] for p in pool if p["cond"] == "day"][:half]
        night_f = [p["fid"] for p in pool if p["cond"] == "night"][:half]
        save_comparison(base, names, all_results, day_f + night_f,
                        os.path.join(args.out_dir, "comparison_random.png"), args.background)
    elif args.mode != "random":
        print("\nNot: yan yana karsilastirma figuru sadece --mode random ile uretilir "
              "(cunku diger modlarda her model icin FARKLI goruntuler secilir).")


if __name__ == "__main__":
    main()