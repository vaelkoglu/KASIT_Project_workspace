import json
import matplotlib.pyplot as plt


def main():
    # alpha path
    log_file = r"C:\Users\acer5\OneDrive\Desktop\kaist-project\alpha_logs.json"

    print(f"Reading {log_file}...")
    with open(log_file, "r") as f:
        logs = json.load(f)

    alphas = [item["alpha"] for item in logs]
    # تم تصحيح الخطأ هنا: استخدام الـ index أو frame_id بدلاً من batch_id الغير موجود في ملف الـ JSON
    batches = range(len(logs))
    # أو إذا أردت استخدام الـ frame_id: frame_ids = [item["frame_id"] for item in logs]

    if not alphas:
        print("No data found in log file.")
        return

    # رسم المخطط البياني (Line Plot)
    plt.figure(figsize=(12, 6))
    plt.plot(batches, alphas, marker='.', linestyle='-', color='teal', alpha=0.7)

    # تحسين شكل المخطط
    plt.title("Adaptive Fusion: Trust Weight (Alpha) Distribution Over Test Samples", fontsize=14)
    plt.xlabel("Test Sample Index", fontsize=12)
    plt.ylabel("Alpha Value (RGB Trust vs Thermal Trust)", fontsize=12)
    plt.axhline(y=0.5, color='r', linestyle='--', label="50/50 Fusion Baseline")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # حفظ الصورة
    output_image = "alpha_distribution_plot.png"
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"تم رسم المخطط بنجاح! تم حفظ الصورة باسم: {output_image}")

    plt.show()


if __name__ == "__main__":
    main()