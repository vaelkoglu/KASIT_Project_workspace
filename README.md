# Adaptive Multispectral Pedestrian Detection (RGB-T) using KAIST Dataset

This repository contains the complete implementation, training scripts, evaluation protocols, and qualitative results for my Summer Internship Project. The project focuses on robust pedestrian detection by intelligently fusing **RGB (Visible)** and **Thermal (LWIR)** camera streams using deep learning.

---

## 🚀 Project Overview & Motivation
Single-modality pedestrian detection systems often fail under extreme environmental conditions:
- **RGB Cameras** lose visibility in pure darkness or nighttime scenarios.
- **Thermal Cameras** suffer from background sun glare and clutter during daytime.

To overcome these limitations, this project implements a **Multispectral Fusion** pipeline built on PyTorch and Faster R-CNN, featuring an innovative **Adaptive Gating Module** that dynamically weights modality trust based on environmental context (Day vs. Night).

---

## 📂 Project Architecture & Roadmap
The project was structured across a comprehensive 25-day development roadmap:

1. **Exploratory Data Analysis (EDA) & Pipeline Setup (Days 1–7):**
   - Parsed KAIST XML annotations and created custom PyTorch `Dataset` and `DataLoader` classes (`day6_dataset.py`).
   - Implemented standard Caltech/Reasonable Protocol evaluation metrics including Log-Average Miss Rate ($MR^{-2}$) and FPPI (`day7_evaluator.py`).

2. **Baseline Models (Days 8–12):**
   - Trained and evaluated individual RGB-only (`day8`) and Thermal-only (`day10`) Faster R-CNN baselines.
   - Analyzed single-modality failure modes under day/night variations.

3. **Fusion Strategies & Adaptive Gating (Days 13–20):**
   - **Early Fusion:** Pixel-level 6-channel concatenation (`day13`).
   - **Feature Fusion:** Dual-stream feature-level FPN integration (`day15`-`day16`).
   - **Adaptive Fusion:** Designed a dynamic $\alpha$-weighting gating module (`day18`) with a 50x Learning Rate Multiplier to successfully resolve gradient stagnation.

4. **Robustness & Qualitative Evaluation (Days 21–25):**
   - Integrated Modality Dropout for sensor failure resilience (`day21`).
   - Generated qualitative comparison matrices (`day24`) and alpha distribution analysis (`day20`).

---

## 📊 Key Results & Visualizations
- **Adaptive Gating:** Automatically shifts modality trust—favoring RGB during daylight and Thermal during nighttime.
- **Performance:** Significantly reduces Miss Rate ($MR^{-2}$) and False Positives compared to single-modality and static fusion baselines.

---

## 🛠️ Requirements & Installation
- Python 3.8+
- PyTorch & Torchvision
- NumPy, Matplotlib, Pillow, OpenCV

---
*Developed as part of the Summer Internship Program.* ا
