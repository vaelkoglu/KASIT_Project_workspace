"""
Gun 11 - RGB ve Thermal Sonuclarinin Karsilastirilmasi (Day/Night)
======================================================================
Gun 9 (RGB) ve Gun 10 (Thermal) test sonuclarini yan yana gosteren
karsilastirma grafikleri olusturur. Bu grafikler raporun ana bulgusunu
(RGB gunduzde, Thermal gecede daha iyi) gorsel olarak kanitlar.

Kullanim (varsayilan degerler zaten senin gercek sonuclarin - direkt calistir):
  python day11_compare_rgb_thermal.py --output_dir ./comparison_results

Farkli sayilarla denemek istersen argumanlarla degistirebilirsin, orn:
  python day11_compare_rgb_thermal.py --rgb_day_mr 42.74 --thermal_day_mr 55.79 ...
"""

import os
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_comparison(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    categories = ["Day", "Night"]
    x = np.arange(len(categories))
    width = 0.35

    # ---- Grafik 1: Miss Rate (MR) karsilastirmasi ----
    fig, ax = plt.subplots(figsize=(7, 5))
    rgb_mr = [results["rgb_day_mr"], results["rgb_night_mr"]]
    thermal_mr = [results["thermal_day_mr"], results["thermal_night_mr"]]

    bars1 = ax.bar(x - width/2, rgb_mr, width, label="RGB-only", color="#3498db")
    bars2 = ax.bar(x + width/2, thermal_mr, width, label="Thermal-only", color="#e67e22")

    ax.set_ylabel("Miss Rate - MR (%)")
    ax.set_title("RGB vs Thermal: Miss Rate Karsilastirmasi\n(Dusuk deger = Daha iyi performans)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.bar_label(bars1, fmt="%.1f%%", padding=3)
    ax.bar_label(bars2, fmt="%.1f%%", padding=3)
    plt.tight_layout()
    mr_path = os.path.join(output_dir, "mr_comparison.png")
    plt.savefig(mr_path, dpi=150)
    plt.close()

    # ---- Grafik 2: AP (Average Precision) karsilastirmasi ----
    fig, ax = plt.subplots(figsize=(7, 5))
    rgb_ap = [results["rgb_day_ap"], results["rgb_night_ap"]]
    thermal_ap = [results["thermal_day_ap"], results["thermal_night_ap"]]

    bars1 = ax.bar(x - width/2, rgb_ap, width, label="RGB-only", color="#3498db")
    bars2 = ax.bar(x + width/2, thermal_ap, width, label="Thermal-only", color="#e67e22")

    ax.set_ylabel("Average Precision - AP (%)")
    ax.set_title("RGB vs Thermal: AP Karsilastirmasi\n(Yuksek deger = Daha iyi performans)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.bar_label(bars1, fmt="%.1f%%", padding=3)
    ax.bar_label(bars2, fmt="%.1f%%", padding=3)
    plt.tight_layout()
    ap_path = os.path.join(output_dir, "ap_comparison.png")
    plt.savefig(ap_path, dpi=150)
    plt.close()

    # ---- Grafik 3: Tek bir ozet grafik (2x1, MR ve AP ustte) - rapor icin ideal ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].bar(x - width/2, rgb_mr, width, label="RGB-only", color="#3498db")
    axes[0].bar(x + width/2, thermal_mr, width, label="Thermal-only", color="#e67e22")
    axes[0].set_ylabel("Miss Rate (%)")
    axes[0].set_title("Miss Rate (dusuk=iyi)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories)
    axes[0].legend()

    axes[1].bar(x - width/2, rgb_ap, width, label="RGB-only", color="#3498db")
    axes[1].bar(x + width/2, thermal_ap, width, label="Thermal-only", color="#e67e22")
    axes[1].set_ylabel("AP (%)")
    axes[1].set_title("Average Precision (yuksek=iyi)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories)
    axes[1].legend()

    fig.suptitle("Gun 11: RGB-only vs Thermal-only Karsilastirmasi (Day/Night)", fontsize=13)
    plt.tight_layout()
    summary_path = os.path.join(output_dir, "rgb_vs_thermal_summary.png")
    plt.savefig(summary_path, dpi=150)
    plt.close()

    return mr_path, ap_path, summary_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gun 11: RGB vs Thermal karsilastirma grafikleri")
    # Varsayilan degerler = Gun 9 ve Gun 10'da elde ettigin gercek sonuclar
    parser.add_argument("--rgb_day_mr", type=float, default=42.74)
    parser.add_argument("--rgb_night_mr", type=float, default=61.88)
    parser.add_argument("--thermal_day_mr", type=float, default=55.79)
    parser.add_argument("--thermal_night_mr", type=float, default=32.72)
    parser.add_argument("--rgb_day_ap", type=float, default=64.17)
    parser.add_argument("--rgb_night_ap", type=float, default=40.77)
    parser.add_argument("--thermal_day_ap", type=float, default=50.33)
    parser.add_argument("--thermal_night_ap", type=float, default=72.48)
    parser.add_argument("--output_dir", type=str, default="./comparison_results")
    args = parser.parse_args()

    results = dict(vars(args))
    del results["output_dir"]

    mr_path, ap_path, summary_path = plot_comparison(results, args.output_dir)

    print("Grafikler kaydedildi:")
    print(f"  - {mr_path}")
    print(f"  - {ap_path}")
    print(f"  - {summary_path}  <- rapor icin en kullanisli olan bu (MR+AP yan yana)")

    print("\n=== Ozet Bulgu ===")
    print(f"RGB     : Day MR={args.rgb_day_mr}% / Night MR={args.rgb_night_mr}%  "
          f"-> {'Gunduzde daha iyi' if args.rgb_day_mr < args.rgb_night_mr else 'Gecede daha iyi'}")
    print(f"Thermal : Day MR={args.thermal_day_mr}% / Night MR={args.thermal_night_mr}%  "
          f"-> {'Gunduzde daha iyi' if args.thermal_day_mr < args.thermal_night_mr else 'Gecede daha iyi'}")
    print("\nBu ters egilim (RGB gunduzde / Thermal gecede daha iyi), Fusion")
    print("yaklasiminin (Gun 13-18) neden gerekli oldugunun somut kanitidir.")