# ♻️ Plastic Waste Classification using CNN

An end-to-end **Deep Learning application** that classifies plastic waste images into  
**Organic** and **Recyclable** categories using a **Convolutional Neural Network (CNN)**.

This project demonstrates the practical application of **Computer Vision for environmental sustainability** and is designed to be reusable as a **model, web app, and showcase project**.

---

## 🎯 Project Goals

- Automate waste classification using image-based AI  
- Reduce manual effort in waste segregation  
- Apply CNNs to a real-world environmental problem  
- Build a reusable and deployable ML model  
- Provide a clean web-based interface for predictions  

---

## 🧠 Model Overview

- **Model Type:** Convolutional Neural Network (CNN)  
- **Framework:** TensorFlow / Keras  
- **Input Shape:** `224 × 224 × 3` (RGB image)  
- **Output Classes:**  
  - Organic  
  - Recyclable  
- **Activation (Output):** Softmax  
- **Saved Format:** `.h5`  

### 🔹 Model Architecture (High Level)

```
Input Image (224×224×3)
        ↓
Conv2D + BatchNorm + MaxPooling
        ↓
Conv2D + BatchNorm + MaxPooling
        ↓
Conv2D + BatchNorm + MaxPooling
        ↓
Flatten
        ↓
Dense + Dropout
        ↓
Dense (Softmax Output)
```

Class order is **locked** as:
```python
["Organic", "Recyclable"]
```

---

## 📂 Dataset Information

- **Dataset Source:** Kaggle  
- **Dataset Name:** Waste Classification Dataset  
- **Kaggle Link:**  
  https://www.kaggle.com/datasets/techsash/waste-classification-data  

### 🔹 Dataset Structure

```
DATASET/
├── TRAIN/
│   ├── O/   (Organic)
│   └── R/   (Recyclable)
├── TEST/
│   ├── O/
│   └── R/
```

---

## ☁️ Google Colab Notebook

Model training and experimentation were performed using **Google Colab**.

📓 **Colab Notebook:**  
[Google Colab link here](https://colab.research.google.com/drive/17b-PB5u30vmC8nE3tbMAFxF1jxLIAVmL?usp=sharing)

The notebook covers:
- Data preprocessing  
- CNN training  
- Validation  
- Model saving (`.h5`)  
- Testing predictions  

---

## 🚀 How to Use the Model Locally

### 1️⃣ Install Dependencies

```bash
pip install tensorflow numpy opencv-python pillow
```

---

### 2️⃣ Load and Predict using the Saved Model

```python
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("PlasticWasteClassifier_v1.h5")

CLASS_NAMES = ["Organic", "Recyclable"]

img = cv2.imread("test.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))
img = img / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)
print("Prediction:", CLASS_NAMES[np.argmax(prediction)])
```

---

## 🌐 Streamlit Web Application

A **Streamlit-based web interface** is used to showcase the trained model.

### Features:
- Image upload  
- Real-time prediction  
- Confidence score display  
- Custom CSS-based UI  

### 🔗 Streamlit Deployment Link:
(Add your Streamlit Cloud link here)

Run locally:
```bash
streamlit run app.py
```

---

## 🛠️ Tech Stack

| Category | Technology |
|--------|------------|
| Language | Python |
| Deep Learning | TensorFlow, Keras |
| Image Processing | OpenCV, Pillow |
| Data Handling | NumPy |
| Web UI | Streamlit |
| Training Platform | Google Colab |
| Dataset Hosting | Kaggle |
| Version Control | Git, GitHub |

---

## 📦 Project Structure

```
PlasticWasteAI/
├── model/
│   └── PlasticWasteClassifier_v1.h5
├── app.py                  # Streamlit web app
├── README.md
├── LICENSE                 # Apache License 2.0
└── NOTICE
```

---

## 🔮 Future Enhancements

- Transfer learning (MobileNet / EfficientNet)  
- Multi-class waste classification  
- Real-time webcam detection  
- Mobile deployment using TensorFlow Lite  
- API-based deployment (FastAPI)  

---

## 👨‍💻 Author

**Soumodeep Das**  
GitHub: https://github.com/SoumodeepDas2004  

---

## 📜 License

This project is licensed under the **Apache License 2.0**.

You are free to use, modify, and distribute this project,  
provided that **original authorship and attribution are preserved**.

---

♻️ *Using AI to build a cleaner and smarter future.*
