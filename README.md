# 🖼️ Image Stitching using SIFT

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-brightgreen)

---

## 📌 Overview

This project implements a **multi-image panorama stitching system** using concepts from the **SIFT (Scale-Invariant Feature Transform)** paper.

The pipeline detects keypoints, matches them across images, estimates geometric transformations using **RANSAC**, and stitches images into a seamless panorama.

---

## 🚀 Features

* 🔍 SIFT-based feature detection
* 🔗 Feature matching using KNN + ratio test
* 📐 Robust homography estimation using RANSAC
* 🖼️ Image warping and stitching
* 🎨 Basic blending for smoother output
* 📊 Visualization of keypoints and matches

---

## 🧩 Project Structure

```
image_stitching/
│
├── modules/
│   ├── feature.py        # Feature extraction (SIFT)
│   ├── matcher.py        # Feature matching
│   ├── homography.py     # Homography + RANSAC
│   └── stitcher.py       # Warping + blending
│
├── tests/
│   └── test_homography.py
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```
git clone <your-repo-link>
cd image_stitching
```

### 2. Create virtual environment

```
python -m venv venv
```

### 3. Activate environment

```
venv\Scripts\activate   # Windows
```

### 4. Install dependencies

```
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Run tests

```
python -m tests.test_homography
```

### Run main pipeline

```
python -m main
```

---

## 🧠 Pipeline

```
Images → SIFT → Feature Matching → Ratio Test
        → RANSAC → Homography → Warping
        → Blending → Panorama
```

---

## 👥 Team Responsibilities

* **Person A** → Feature Extraction (SIFT)
* **Person B** → Feature Matching
* **Person C** → Homography & RANSAC
* **Person D** → Warping & Blending

---

## 📅 Timeline

| Phase     | Work                             |
| --------- | -------------------------------- |
| Days 1–4  | Feature detection & matching     |
| Days 5–8  | Homography & warping             |
| Days 9–12 | Blending & multi-image stitching |

---

## 📸 Expected Output

* Panorama generated from **2–5 overlapping images**
* Visualization of:

  * Keypoints
  * Feature matches
  * Inliers after RANSAC

---

## 📚 References

* David G. Lowe, *Distinctive Image Features from Scale-Invariant Keypoints*, 2004

---

## 🧠 Future Improvements

* Multi-band blending
* Cylindrical projection
* Real-time stitching
* GPU acceleration

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

---

## 📜 License

This project is licensed under the MIT License.
