# 📩 Spam Detection App

An AI-powered web application that classifies SMS messages as **Spam** or **Ham** using natural language processing (NLP) and machine learning. Built with a Streamlit front end and trained on real-world SMS data, this tool is designed for educational, enterprise prototyping, and cybersecurity demonstration purposes.

---

## 🚀 Features

- ✅ Real-time SMS spam classification
- ✅ Clean and interactive web UI (Streamlit)
- ✅ Pretrained model using NLP + supervised ML
- ✅ Handles punctuation, casing, and irrelevant words
- ✅ Lightweight and easy to deploy locally or on the cloud

---

## 🛠️ Tools & Technologies Used

| Category         | Tools / Libraries                        |
|------------------|------------------------------------------|
| Programming      | Python                                   |
| Web Framework    | Streamlit                                |
| NLP              | NLTK / Scikit-learn Text Vectorizers     |
| ML Algorithms    | SVM (or Logistic Regression/Naive Bayes/
|                         RandomForest)
| Data Handling    | Pandas, NumPy                            |
| Model Persistence| `pickle` / `joblib`                      |
| Deployment       | Streamlit Cloud / Localhost              |

---

## 📊 Dataset Used

- **Name**: [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Source**: UCI Machine Learning Repository
- **Size**: 5,574 SMS messages labeled as "spam" or "ham"
- **Format**: Plain CSV (text + label)

---

## 🔍 Use Cases

### 💼 Professional Applications
- **Email/SMS Filtering**: Can be extended to email clients or enterprise messaging platforms.
- **Customer Support Screening**: Filter malicious or irrelevant messages.
- **Telecom Compliance Tools**: Detect unwanted spam to comply with communication regulations.

### 🎓 Educational Purposes
- Teaches the basics of NLP, text vectorization, and model deployment.
- A beginner-friendly entry point into AI and web app integration.

---

## ⚙️ Setup Instructions

### 🔐 Prerequisites

- Python 3.7+
- pip
- Git (optional)

### 📦 Installation Steps

```bash
git clone https://github.com/devarenidhi25/spam-detection-app.git
cd spam-detection-app
pip install -r requirements.txt
