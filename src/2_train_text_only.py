import pandas as pd
import numpy as np
import os
import joblib
import re

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sentence_transformers import SentenceTransformer
from imblearn.over_sampling import SMOTE

# 1. CONFIG
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
# Use the Biased data (so we catch the 'He' vs 'She' bias)
DATA_PATH = os.path.join(project_root, 'data', 'recruitment_data_biased.csv')
MODEL_DIR = os.path.join(project_root, 'models')

print(f"📖 Loading Data from {DATA_PATH}...")
if not os.path.exists(DATA_PATH):
    print("❌ Error: Run 'src/0_generate_biased_data.py' first!")
    exit()

df = pd.read_csv(DATA_PATH)

# 2. CLEAN & PREPARE
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return " ".join(text.split())

df['clean_text'] = df['ResumeText'].apply(clean_text)
df['target'] = df['HiringDecision'].map({'Selected': 1, 'Rejected': 0})

# 3. FEATURE ENGINEERING (TEXT ONLY - NO EXPERIENCE COLUMN)
print("🧠 Converting Text to Numbers (MiniLM)...")
encoder = SentenceTransformer('all-MiniLM-L6-v2')
X = encoder.encode(df['clean_text'].tolist(), show_progress_bar=True)
y = df['target'].values

# 4. BALANCE (Crucial)
print("⚖️ Balancing the dataset...")
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# 5. TRAIN NEURAL NET
print("🐢 Training Text-Only Neural Network...")
# Smaller network forces it to generalize, not memorize
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)

# 6. EVALUATE
y_pred = mlp.predict(X_test)
print(f"\n🚀 TEXT-ONLY ACCURACY: {accuracy_score(y_test, y_pred)*100:.2f}%")

# Save
joblib.dump(mlp, os.path.join(MODEL_DIR, 'mlp_text_only.pkl'))
print("💾 Model saved as 'mlp_text_only.pkl'")