"""
Gun 17 - RGB / Thermal / Early Fusion / Feature Fusion Ana Karsilastirma Tablosu
======================================================================================
Gun 9, 10, 14 ve 16'da elde edilen gercek test sonuclarini tek bir tabloda ve
grafikte birlestirir. Bu, "Ana baseline tablosu" (projenin en onemli ozet
ciktisi) - rapor ve sunumda dogrudan kullanilabilir.

Kullanim (varsayilan degerler zaten elde ettigin gercek sonuclar):
  python day17_compare_all_baselines.py --output_dir ./comparison_results

Feature Fusion sonuclari henuz hazir degilse, egitim bitince asagidaki
--ff_* argumanlariyla gercek degerleri gir (ya da varsayilan placeholder
degerlerle 'henuz tamamlanmadi' notuyla calistir).
"""

import os
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODEL_ORDER = ["RGB-only", "Thermal-only", "Early Fusion", "Feature Fusion"]
COLORS = {"RGB-only": "#3498db", "Thermal-only": "#e67e22",
          "Early Fusion": "#9b59b6", "Feature Fusion": "#2ecc71"}


def plot_main_comparison(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    categories = ["Day", "Night"]
    x = np.arange(len(categories))
    width = 0.2

    # ---- MR karsilastirmasi (4 model) ----
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model in enumerate(MODEL_ORDER):
        vals = [results[model]["day_mr"], results[model]["night_mr"]]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, vals, width, label=model, color=COLORS[model])
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    ax.set_ylabel("Miss Rate - MR (%)")
    ax.set_title("Gun 17: Ana Baseline Karsilastirmasi - Miss Rate\n(Dusuk deger = Daha iyi performans)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.08))
    plt.tight_layout()
    mr_path = os.path.join(output_dir, "day17_mr_all_models.png")
    plt.savefig(mr_path, dpi=150)
    plt.close()

    # ---- AP karsilastirmasi (4 model) ----
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model in enumerate(MODEL_ORDER):
        vals = [results[model]["day_ap"], results[model]["night_ap"]]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, vals, width, label=model, color=COLORS[model])
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    ax.set_ylabel("Average Precision - AP (%)")
    ax.set_title("Gun 17: Ana Baseline Karsilastirmasi - AP\n(Yuksek deger = Daha iyi performans)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.08))
    plt.tight_layout()
    ap_path = os.path.join(output_dir, "day17_ap_all_models.png")
    plt.savefig(ap_path, dpi=150)
    plt.close()

    return mr_path, ap_path


def print_markdown_table(results):
    print("\n## Ana Baseline Tablosu (Gun 17)\n")
    print("| Model | MR-All | MR-Day | MR-Night | AP-All | AP-Day | AP-Night |")
    print("|---|---|---|---|---|---|---|")
    for model in MODEL_ORDER:
        r = results[model]
        if r.get("placeholder"):
            print(f"| {model} | *henuz tamamlanmadi* | | | | | |")
            continue
        print(f"| {model} | {r['all_mr']:.2f}% | {r['day_mr']:.2f}% | {r['night_mr']:.2f}% | "
              f"{r['all_ap']:.2f}% | {r['day_ap']:.2f}% | {r['night_ap']:.2f}% |")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gun 17: Ana baseline karsilastirma tablosu")

    # RGB-only (Gun 9 gercek sonuclari)
    parser.add_argument("--rgb_all_mr", type=float, default=48.99)
    parser.add_argument("--rgb_day_mr", type=float, default=42.74)
    parser.add_argument("--rgb_night_mr", type=float, default=61.88)
    parser.add_argument("--rgb_all_ap", type=float, default=56.64)
    parser.add_argument("--rgb_day_ap", type=float, default=64.17)
    parser.add_argument("--rgb_night_ap", type=float, default=40.77)

    # Thermal-only (Gun 10 gercek sonuclari)
    parser.add_argument("--thermal_all_mr", type=float, default=48.32)
    parser.add_argument("--thermal_day_mr", type=float, default=55.79)
    parser.add_argument("--thermal_night_mr", type=float, default=32.72)
    parser.add_argument("--thermal_all_ap", type=float, default=57.57)
    parser.add_argument("--thermal_day_ap", type=float, default=50.33)
    parser.add_argument("--thermal_night_ap", type=float, default=72.48)

    # Early Fusion (Gun 14 gercek sonuclari)
    parser.add_argument("--early_all_mr", type=float, default=49.38)
    parser.add_argument("--early_day_mr", type=float, default=43.81)
    parser.add_argument("--early_night_mr", type=float, default=61.65)
    parser.add_argument("--early_all_ap", type=float, default=55.33)
    parser.add_argument("--early_day_ap", type=float, default=62.33)
    parser.add_argument("--early_night_ap", type=float, default=40.48)

    # Feature Fusion (Gun 16 - egitim bitince gercek degerleri buraya gir)
    parser.add_argument("--ff_all_mr", type=float, default=None)
    parser.add_argument("--ff_day_mr", type=float, default=None)
    parser.add_argument("--ff_night_mr", type=float, default=None)
    parser.add_argument("--ff_all_ap", type=float, default=None)
    parser.add_argument("--ff_day_ap", type=float, default=None)
    parser.add_argument("--ff_night_ap", type=float, default=None)

    parser.add_argument("--output_dir", type=str, default="./comparison_results")
    args = parser.parse_args()

    results = {
        "RGB-only": {
            "all_mr": args.rgb_all_mr, "day_mr": args.rgb_day_mr, "night_mr": args.rgb_night_mr,
            "all_ap": args.rgb_all_ap, "day_ap": args.rgb_day_ap, "night_ap": args.rgb_night_ap,
        },
        "Thermal-only": {
            "all_mr": args.thermal_all_mr, "day_mr": args.thermal_day_mr, "night_mr": args.thermal_night_mr,
            "all_ap": args.thermal_all_ap, "day_ap": args.thermal_day_ap, "night_ap": args.thermal_night_ap,
        },
        "Early Fusion": {
            "all_mr": args.early_all_mr, "day_mr": args.early_day_mr, "night_mr": args.early_night_mr,
            "all_ap": args.early_all_ap, "day_ap": args.early_day_ap, "night_ap": args.early_night_ap,
        },
    }

    if args.ff_day_mr is None:
        print("UYARI: Feature Fusion sonuclari girilmedi (--ff_* argumanlari) - placeholder ile devam ediliyor.")
        print("Egitim/test bitince gercek degerlerle tekrar calistir, orn:")
        print("  --ff_all_mr X --ff_day_mr X --ff_night_mr X --ff_all_ap X --ff_day_ap X --ff_night_ap X\n")
        results["Feature Fusion"] = {
            "all_mr": 0, "day_mr": 0, "night_mr": 0, "all_ap": 0, "day_ap": 0, "night_ap": 0,
            "placeholder": True,
        }
    else:
        results["Feature Fusion"] = {
            "all_mr": args.ff_all_mr, "day_mr": args.ff_day_mr, "night_mr": args.ff_night_mr,
            "all_ap": args.ff_all_ap, "day_ap": args.ff_day_ap, "night_ap": args.ff_night_ap,
        }

    print_markdown_table(results)

    mr_path, ap_path = plot_main_comparison(results, args.output_dir)
    print(f"Grafikler kaydedildi:\n  - {mr_path}\n  - {ap_path}")

    print("\n=== Gun 17 Ana Bulgu ===")
    print("RGB: gunduzde guclu, gecede zayif.")
    print("Thermal: gecede guclu, gunduzde zayif.")
    print("Early Fusion: RGB'ye cok yakin kaldi (thermal kanali neredeyse goz ardi edildi).")
    if not results["Feature Fusion"].get("placeholder"):
        ff = results["Feature Fusion"]
        print(f"Feature Fusion: Day MR={ff['day_mr']:.2f}%, Night MR={ff['night_mr']:.2f}% "
              f"-> {'Dengeli/iyilesme var' if abs(ff['day_mr']-ff['night_mr']) < abs(42.74-61.88) else 'Henuz dengelenmedi'}")