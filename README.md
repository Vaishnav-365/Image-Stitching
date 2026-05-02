# 🖼️ Advanced Image Stitching using SIFT

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-brightgreen)

---

## 📌 Overview

This project implements a **complete multi-image panorama stitching system** based on concepts from the **SIFT (Scale-Invariant Feature Transform)** paper.

It detects keypoints, matches features across multiple images, estimates transformations using **RANSAC**, aligns images into a common coordinate space, and produces a **clean, blended panoramic output**.

---

## 🚀 Features

* 🔍 SIFT-based feature detection with preprocessing
* 🔗 Feature matching using KNN + Lowe’s ratio test
* 📐 Robust homography estimation using RANSAC + refinement
* 🌍 Multi-image stitching (3+ images supported)
* 🔄 Global homography chaining (reference frame alignment)
* 🌐 Cylindrical projection for distortion reduction
* 🎨 Advanced blending (feather blending / weighted blending)
* ✂️ Automatic black border cropping
* ✨ Optional sharpening for final output
* 📊 Visualization of keypoints, matches, and inliers

---

## 🧩 Project Structure

```
image_stitching/
│
├── modules/
│ ├── feature.py # Feature extraction (SIFT + preprocessing)
│ ├── matcher.py # Feature matching (KNN + filtering)
│ ├── homography.py # Homography + RANSAC
│ ├── transform.py # Homography chaining & normalization
│ ├── projection.py # Cylindrical projection
│ ├── pipeline.py # Full stitching pipeline
│ ├── postprocess.py # Cropping & sharpening
│ └── stitch.py # (legacy / optional stitching logic)
│
├── tests/
│ ├── test_homography.py
│ └── test_matcher.py
│
├── images/ # Input images
├── outputs/ # Generated results
├── main.py # Entry point
├── requirements.txt
├── .gitignore
└── README.md

```

---

## ⚙️ Installation

### 1. Clone the repository

``` 
git clone https://github.com/Vaishnav-365/Image-Stitching.git
cd Image-Stitching

```

### 2. Create virtual environment

```
python -m venv venv

```
### 3. Activate environment

```
venv\Scripts\activate # Windows
source venv/bin/activate # Linux/Mac

```

### 4. Install dependencies

```
pip install -r requirements.txt

```

---

## ▶️ Running the Project

### Run main pipeline

```
python -m main

```

---

## 🧠 Pipeline

```
Images
→ Preprocessing (Grayscale + Histogram Equalization)
→ SIFT Feature Detection
→ Feature Matching (KNN + Ratio Test)
→ RANSAC Homography Estimation
→ Inlier Refinement
→ Global Homography Alignment
→ Cylindrical Projection
→ Image Warping
→ Blending (Feather / Weighted)
→ Cropping & Sharpening
→ Final Panorama

```

---

## 👥 Team Responsibilities

* **Person A** → Feature Extraction (SIFT + preprocessing)
* **Person B** → Feature Matching (robust matching + filtering)
* **Person C** → Pipeline, Homography, Multi-image alignment, Projection, Blending, Post-processing
* **Person D** → (Optional) Advanced blending, UI, optimization

---

## 📅 Timeline

| Phase     | Work                                      |
|----------|-------------------------------------------|
| Days 1–4  | Feature detection & matching              |
| Days 5–8  | Homography, validation & multi-image setup|
| Days 9–10 | Blending improvements & robustness        |
| Day 11    | Cylindrical projection                   |
| Day 12    | Cropping, sharpening & final polish       |

---

## 📸 Output

The system generates:

* 🖼️ Seamless panorama from **multiple overlapping images**
* 📊 Visualizations:
  * Keypoints detected
  * Feature matches
  * RANSAC inliers
* 📁 Saved outputs:
  * Final panorama
  * Intermediate debug images

---

## ⚠️ Known Limitations

* Minor ghosting may occur due to parallax
* Assumes moderate overlap between images
* Works best with consistent lighting conditions

---

## 📚 References

* David G. Lowe,  
  *Distinctive Image Features from Scale-Invariant Keypoints*, 2004

---

## 🧠 Future Improvements

* Multi-band blending (reduce ghosting further)
* Exposure compensation
* Seam finding algorithms
* Real-time stitching
* GPU acceleration (CUDA/OpenCL)
* Integration with video streams

---

## 🤝 Contributing

Contributions, improvements, and suggestions are welcome!

---

## 📜 License

This project is licensed under the MIT License.