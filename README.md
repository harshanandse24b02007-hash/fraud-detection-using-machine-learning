# 💳 Credit Card Fraud Detection System (Machine Learning)

## 🚀 Overview

This project implements a **production-oriented credit card fraud detection system** using machine learning techniques. It analyzes transaction data to identify fraudulent activities with high recall while minimizing false positives.

The system is designed with a modular pipeline including **data preprocessing, model training, evaluation, and deployment-ready prediction interface**.

---

## 🎯 Objectives

* Detect fraudulent transactions in highly imbalanced datasets
* Build a scalable and reusable ML pipeline
* Optimize performance using precision, recall, and ROC-AUC
* Provide a simple user interface for real-time predictions

---

## 🧠 Tech Stack

* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, Joblib
* **Visualization:** Matplotlib, Seaborn
* **Deployment UI:** Streamlit

---

## 📂 Project Structure

```
credit-card-fraud-detection-ml/
│
├── data/
│   └── creditcard.csv
│
├── models/
│   ├── fraud_model.pkl
│   └── scaler.pkl
│
├── train_model.py
├── predict.py
├── app.py
├── requirements.txt
├── .gitignore
├── notebook.ipynb
└── README.md
```

---

## 📊 Dataset

The dataset contains anonymized credit card transactions.

* Features: PCA-transformed variables (V1–V28), Time, Amount
* Target:

  * `0` → Legitimate
  * `1` → Fraud

⚠️ Dataset is highly imbalanced.

👉 Download dataset:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place it in:

```
data/creditcard.csv
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```
git clone https://github.com/your-username/credit-card-fraud-detection-ml.git
cd credit-card-fraud-detection-ml
```

---

### 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

### 3️⃣ Train Model

```
python train_model.py
```

This will:

* Train the model
* Save files in `models/` folder

---

### 4️⃣ Run Prediction Script (Optional)

```
python predict.py
```

---

### 5️⃣ Launch Web App (Recommended)

```
streamlit run app.py
```

Then open browser at:

```
http://localhost:8501
```

---

## 📈 Machine Learning Pipeline

1. Data Loading
2. Data Preprocessing
3. Feature Scaling
4. Handling Class Imbalance
5. Model Training (Random Forest)
6. Model Evaluation
7. Model Serialization (Joblib)

---

## 📊 Evaluation Metrics

* Precision
* Recall
* F1-Score
* Confusion Matrix
* ROC-AUC

Focus is on **high recall** to minimize undetected fraud.

---

## 🔍 Key Features

* Handles imbalanced dataset
* Modular ML pipeline
* Model persistence using Joblib
* Ready for deployment
* Interactive UI using Streamlit

---

## 🚀 Future Enhancements

* Real-time API deployment (Flask/FastAPI)
* Deep learning models (ANN)
* Advanced anomaly detection techniques
* Cloud deployment (AWS/GCP)

---

## 👨‍💻 Author

B.Tech CSE Student
Machine Learning & Software Engineering Enthusiast

---

## ⭐ Project Highlights (For Recruiters)

* End-to-end ML pipeline implementation
* Real-world financial fraud detection use-case
* Deployment-ready architecture
* Clean and scalable project structure

---
