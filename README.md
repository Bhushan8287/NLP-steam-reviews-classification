## 🎮 Steam Reviews Sentiment Analysis

A binary sentiment classification project using a real-world dataset of Steam game reviews. The goal is to classify user reviews as **positive** or **negative**, using classical NLP techniques and machine learning models.

---

### 📖 Overview
This project demonstrates a machine learning flow for binary text classification using NLP. It covers:
- Data cleaning and preprocessing
- Feature extraction using **BoW**, **TF-IDF**, and **Word2Vec**
- Model training and evaluation (Logistic Regression, etc.)
- Hyperparameter tuning
- Threshold optimization for classification
- Streamlit-based deployment for live inference

The dataset contains 25,000 Steam reviews (trimmed from ~50k for performance reasons). All metrics, models, and preprocessing artifacts are stored in a structured folder hierarchy.

---

### ❓ Problem Statement
The task is to predict whether a Steam user review expresses **positive** or **negative** sentiment based on its text content.

---

### 🧰 Tech Stack
- Python 
- scikit-learn
- pandas, NumPy
- nltk, gensim
- matplotlib, seaborn
- Streamlit (deployment)
- joblib / pickle

---

### 📂 Project Structure
```
.
├── app.py                            # Streamlit app for inference
├── working.ipynb                    # Main development notebook
├── prediction.ipynb                # Inference + threshold tuning
├── dataset/
│   ├── steam_reviews_raw.csv
│   └── steam_reviews_cleaned_25k.csv
├── hyper_tune_models/              # Pickled models after tuning
│   └── logreg_tfidf.pkl
├── metrics/
│   └── model_metrics.json
├── pre_processors/
│   └── tfidf_vectorizer.pkl
├── split_arrays/
│   ├── X_train.npy
│   └── X_test.npy
├── final_model_log_reg.pkl         # Best performing model
├── best_tfidf_vectorizer.pkl       # Used in Streamlit app
├── requirements.txt
└── README.md
```

---

### ⚙️ How It Works
1. **Preprocessing**:
   - Cleaned reviews, removed stopwords, lowercased text
2. **Vectorization**:
   - Used three techniques: BoW, TF-IDF, and Word2Vec
   - Evaluated all three using classical ML models
3. **Model Training & Tuning**:
   - Trained and evaluated models: [logistic regression, multi NB, linear svc, knn, random forest, gradboost, decision trees]
   - Tuned hyperparameters for TF-IDF
4. **Evaluation**:
   - ROC-AUC, F1-score, Precision-Recall, Confusion Matrix
   - Threshold optimization for class prediction
5. **Deployment**:
   - Final TF-IDF + Logistic Regression(best final model) pipeline integrated into Streamlit
   - Hosted on Streamlit Community Cloud

---

### 📊 Results
- Best performing vectorization: **TF-IDF**
- Final model: **Logistic Regression**
- AUC Score: **[0.9173]**
- Threshold optimized for better precision-recall balance
- Tested the model in `prediction.ipynb`

---

### 🚀 Run Locally
```
git clone the repo
pip install -r requirements.txt
streamlit run app.py
```

---

### 🔗 Live Demo
Access the deployed Streamlit app here:  
👉 [Streamlit Cloud URL]()

---

### 📉 Limitations
- No use of deep learning (LSTM, BERT) to keep it lightweight
- Models are classical, not neural
- Only binary classification (neutral/ambiguous excluded)

---

### 🔮 Future Improvements
- Use BERT or DistilBERT for embeddings
- Integrate MLflow or Weights & Biases for experiment tracking
- Add REST API for backend integration
- Implement real-time data ingestion
- Improve UI/UX of the Streamlit app

---

### 📜 License
[MIT License](LICENSE)
