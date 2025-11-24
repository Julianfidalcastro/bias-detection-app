import pandas as pd
import numpy as np
import os
import joblib
import re

# Libraries for Neural Network
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sentence_transformers import SentenceTransformer
from imblearn.over_sampling import SMOTE

# ==========================================
# 1. CONFIG & LOAD
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
# Use your Biased dataset to show the "Favoritism" detection
DATA_PATH = os.path.join(project_root, 'data', 'recruitment_data_biased.csv')
MODEL_DIR = os.path.join(project_root, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"🧠 Starting Neural Network Training Pipeline...")
print(f"📂 Loading Data: {DATA_PATH}")

if not os.path.exists(DATA_PATH):
    print("❌ Error: Dataset not found. Run 'src/0_generate_biased_data.py' first.")
    exit()

df = pd.read_csv(DATA_PATH)

# Clean Text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return " ".join(text.split())

df['clean_text'] = df['ResumeText'].apply(clean_text)
df['target'] = df['HiringDecision'].map({'Selected': 1, 'Rejected': 0})
df = df.dropna(subset=['clean_text', 'target'])

# ==========================================
# 2. FEATURE ENGINEERING (With Scaling)
# ==========================================
print("⏳ Generating Embeddings (MiniLM)...")
encoder = SentenceTransformer('all-MiniLM-L6-v2')
X_text = encoder.encode(df['clean_text'].tolist(), show_progress_bar=True)

# Experience & Role
X_exp = df['Experience'].fillna(0).astype(int).values.reshape(-1, 1)
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_role = ohe.fit_transform(df[['JobRole_Applied']])

# Combine
X = np.hstack([X_text, X_exp, X_role])
y = df['target'].values

# SCALING (Crucial for Neural Networks)
print("⚖️ Scaling Features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 3. BALANCE & SPLIT
# ==========================================
print("⚖️ Applying SMOTE (Balancing)...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# ==========================================
# 4. TRAIN MLP (Neural Network)
# ==========================================
print("🐢 Training MLP Classifier (This might take a moment)...")
# A robust configuration for tabular + text data
mlp = MLPClassifier(
    hidden_layer_sizes=(32, 16), # Two layers of "neurons"
    activation='relu',
    solver='adam',
    alpha=0.0001,
    learning_rate='adaptive',
    max_iter=500,
    random_state=42
)

mlp.fit(X_train, y_train)

# ==========================================
# 5. EVALUATE & SAVE
# ==========================================
y_pred = mlp.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n🚀 NEURAL NET ACCURACY: {acc * 100:.2f}%")
print(classification_report(y_test, y_pred))

# Save with distinct names
joblib.dump(mlp, os.path.join(MODEL_DIR, 'mlp_model.pkl'))
joblib.dump(ohe, os.path.join(MODEL_DIR, 'mlp_ohe_encoder.pkl'))
joblib.dump(scaler, os.path.join(MODEL_DIR, 'mlp_scaler.pkl'))

print(f"💾 Saved 'mlp_model.pkl', 'mlp_scaler.pkl', 'mlp_ohe_encoder.pkl' to {MODEL_DIR}")