# 📰 Fake News Detection System

A Deep Learning based Fake News Detection System built using:

- BiLSTM
- DistilBERT
- FastAPI
- Streamlit
- TensorFlow

The system predicts whether a news article is **Real** or **Fake**.

---

# 🚀 Features

- Real-time fake news prediction
- BiLSTM v2 production model
- DistilBERT experimental model
- FastAPI backend
- Streamlit frontend
- Real-world testing pipeline
- Model versioning
- REST API endpoints

---

# 🧠 Models Used

## 1. BiLSTM v2
Production-ready lightweight model.

### Advantages
- Fast inference
- Low memory usage
- Easy deployment
- Good real-world performance

---

## 2. DistilBERT
Transformer-based experimental model.

### Advantages
- Strong contextual understanding
- Better semantic learning

### Limitation
Due to deployment resource constraints,
DistilBERT is kept for local experimentation only.

---

# 📂 Dataset

Training data includes:

- ISOT Fake News Dataset
- Reuters-style real news samples
- Blog clickbait samples
- Twitter noisy samples

---

# 🏗️ Project Architecture

```text
Streamlit Frontend
        ↓
FastAPI Backend
        ↓
BiLSTM v2 Model
```

---

# ⚙️ Tech Stack

- Python
- TensorFlow
- FastAPI
- Streamlit
- Scikit-learn
- Pandas
- NumPy

---

# 📊 Model Performance

| Model | Real-world Accuracy | Speed |
|---|---|---|
| BiLSTM v1 | 40% | Fast |
| BiLSTM v2 | 82% | Fast |
| DistilBERT | 70% | Slow |

---

# 📌 API Endpoints

## Health Check

```bash
GET /health
```

## Prediction

```bash
POST /predict
```

Example request:

```json
{
    "text": "News article text here"
}
```

---

# ▶️ Run Locally

## 1. Clone Repository

```bash
git clone <your-repo-url>
cd fake-news-detection-system
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Run FastAPI Backend

```bash
uvicorn api.main_deploy:app --reload
```

Backend runs at:

```text
http://127.0.0.1:8000
```

Swagger docs:

```text
http://127.0.0.1:8000/docs
```

---

## 4. Run Streamlit Frontend

```bash
streamlit run app/streamlit_app.py
```

---

# 📷 Screenshots

(Add screenshots here later)

---

# 🔮 Future Improvements

- Live news scraping
- User authentication
- Multilingual fake news detection
- Explainable AI predictions
- Docker deployment
- Cloud deployment

---

# 👨‍💻 Author

Dhanush Kumar# Fake-news-system
