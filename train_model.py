import os
import sys
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# import preprocess
sys.path.append(os.path.dirname(__file__))
from preprocess import clean_text

# paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# load datasets
fake = pd.read_csv(os.path.join(DATA_DIR, "Fake.csv"))
true = pd.read_csv(os.path.join(DATA_DIR, "True.csv"))

# labels
fake["label"] = 0  # FAKE
true["label"] = 1  # REAL

# combine title + text (IMPORTANT)
fake["content"] = (fake["title"].fillna("") + " " + fake["text"].fillna(""))
true["content"] = (true["title"].fillna("") + " " + true["text"].fillna(""))

# combine & shuffle
df = pd.concat([fake, true], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# clean text
df["clean"] = df["content"].apply(clean_text)

X = df["clean"]
y = df["label"]

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# vectorizer (NO stopwords here — already cleaned)
vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    min_df=5
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# STABLE MODEL
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train_vec, y_train)

# evaluation
pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

# save
joblib.dump(model, os.path.join(MODEL_DIR, "fake_news_model.pkl"))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))

print("✅ Model training complete and saved.")
