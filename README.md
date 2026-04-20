# 🖼️ Image Stitching using SIFT

## 📌 Overview
This project implements a **multi-image panorama stitching system** using concepts from the SIFT (Scale-Invariant Feature Transform) paper.

The system detects features, matches them across images, estimates transformations using RANSAC, and stitches images into a seamless panorama.

---

## 🚀 Features
- SIFT-based feature detection
- Feature matching using KNN + ratio test
- Robust homography estimation using RANSAC
- Image warping and stitching
- Basic blending for smoother output

---

## 🧩 Project Structure
image_stitching/
│
├── modules/
│ ├── feature.py
│ ├── matcher.py
│ ├── homography.py
│ └── stitcher.py
│
├── tests/
│ └── test_homography.py
│
├── main.py
├── requirements.txt
└── README.md

---

## ⚙️ Installation

1. Create virtual environment:
python -m venv venv

2. Activate environment:
venv\Scripts\activate # Windows

3. Install dependencies:
pip install -r requirements.txt

---

## ▶️ Running the Project

Run tests:
python -m tests.test_homography

Run main pipeline:
python -m main

---

## 👥 Team Responsibilities

- **Person A**: Feature Extraction (SIFT)
- **Person B**: Feature Matching
- **Person C**: Homography & RANSAC
- **Person D**: Warping & Blending

---

## 📅 Timeline
- Day 1–4: Feature detection & matching
- Day 5–8: Homography & warping
- Day 9–12: Blending & multi-image stitching

---

## 📚 References
- Lowe, D. G. (2004). Distinctive Image Features from Scale-Invariant Keypoints.

---

## 📸 Expected Output
- Stitched panorama from 2–5 overlapping images

---

## 🧠 Future Improvements
- Multi-band blending
- Cylindrical projection
- Real-time stitching

---