# Face-Mask-Detection

# 😷 Face Mask Detection Using Deep Learning and OpenCV

This project is a deep learning-based system to automatically detect whether a person is wearing a face mask or not in real-time. Built using **TensorFlow**, **Keras**, **MobileNetV2**, and **OpenCV**, this project is optimized for both training performance and real-world deployment.

---

## 🧠 Project Overview

Face mask detection has become an essential tool in maintaining public health in spaces like airports, hospitals, offices, and other public areas. This project:
- Uses **transfer learning** with **MobileNetV2**
- Achieves **99.19% accuracy** on the test dataset
- Detects faces and classifies them as `With Mask` or `Without Mask`
- Includes **real-time webcam detection**
- Future-ready for deployment on **AWS / Streamlit / Flask**

---

## 📁 Dataset Structure

This project uses a dataset organized into three folders:
dataset/
├── train/
│ ├── WithMask/
│ └── WithoutMask/
├── validation/
│ ├── WithMask/
│ └── WithoutMask/
└── test/
├── WithMask/
└── WithoutMask/

> 🔢 Total images used: **12,000+**
>  
> Source: Custom or Kaggle-based datasets (split manually into train/validation/test)

---

## 🧱 Model Architecture

We used **MobileNetV2** as the base feature extractor with custom top layers added:

- `GlobalAveragePooling2D()`
- `Dropout(0.5)`
- `Dense(128, activation='relu')`
- `Dropout(0.5)`
- `Dense(1, activation='sigmoid')`  → Binary classifier

**Training Strategy:**
1. Freeze the base MobileNetV2 model and train the custom top layers (5 epochs)
2. Unfreeze the last 20 layers of MobileNetV2 and fine-tune the model (5 epochs)

---

## 📊 Performance Results

| Metric         | Value      |
|----------------|------------|
| Test Accuracy  | **99.19%** |
| Optimizer      | Adam       |
| Loss Function  | Binary Crossentropy |
| Input Shape    | (128, 128, 3)       |

**Confusion Matrix, Accuracy vs. Loss plots, and sample predictions are included in the notebook.**

---

## 📂 Project Files

| File / Folder                   | Description                              |
|--------------------------------|------------------------------------------|
| `face_mask_training.ipynb`     | Jupyter notebook with full training flow |
| `face_mask_model.h5`           | Saved trained Keras model                |
| `real_time_face_detection.py`  | Python script for real-time detection    |
| `haarcascade_frontalface_default.xml` | OpenCV Haar cascade for face detection |
| `requirements.txt`             | Python dependencies                      |
| `README.md`                    | Project documentation                    |

---

## 🖥️ How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/your-username/face-mask-detection.git
cd face-mask-detection
