import pandas as pd
import numpy as np
import os
import joblib
import re

# AI & Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. SETUP PATHS (The Important Part)
# ==========================================
# This gets the folder where THIS script is running (src/)
current_dir = os.path.dirname(os.path.abspath(__file__))

# This moves one step up to the project folder, then into 'data'
# It looks for: Bias_Detection_Project/data/recruitment_data.csv
project_root = os.path.dirname(current_dir)
data_dir = os.path.join(project_root, 'data')` `

# Prefer the new dataset name, but keep previous fallbacks for compatibility
possible_names = ['recruitment_data_biased.csv']
DATA_PATH = None
for name in possible_names:
    candidate = os.path.join(data_dir, name)
    if os.path.exists(candidate):
        DATA_PATH = candidate
        found_name = name
        break
    
if DATA_PATH is None:
    # Helpful diagnostic listing files in the data folder (if any)
    available = []
    if os.path.exists(data_dir):
        try:
            available = os.listdir(data_dir)
        except Exception:
            available = []
    print(f"\n❌ ERROR: File not found!")
    print(f"Searched for: {', '.join(possible_names)}")
    if available:
        print(f"Files in `data/`: {', '.join(available)}")
    else:
        print(f"`data/` folder is missing or empty. Current working directory is: {os.getcwd()}")
    exit()
MODEL_DIR = os.path.join(project_root, 'models')

os.makedirs(MODEL_DIR, exist_ok=True)

print(f"📂 Using data file: {DATA_PATH}")

# ==========================================
# 2. LOAD DATA
# ==========================================
if not os.path.exists(DATA_PATH):
    print("\n❌ ERROR: File not found!")
    print(f"Please ensure your file is named 'recruitment_data.csv' and is inside the 'data' folder.")
    print(f"Current working directory is: {os.getcwd()}")
    exit()

df = pd.read_csv(DATA_PATH)
print(f"✅ Data Loaded Successfully. Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# ==========================================
# 3. DATA CLEANING
# ==========================================
def clean_text(text):
    if pd.isnull(text): return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text) # Remove special chars
    return " ".join(text.split())

print("⏳ Cleaning text...")
# We use the column names seen in your screenshot
df['clean_text'] = df['ResumeText'].apply(clean_text)

# Map Target labels to 1 (Selected) and 0 (Rejected)
df['target'] = df['HiringDecision'].map({'Selected': 1, 'Rejected': 0})
df['gender_numeric'] = df['Gender'].map({'Male': 0, 'Female': 1})

# Drop rows with missing values
df = df.dropna(subset=['clean_text', 'target', 'gender_numeric'])

# ==========================================
# 4. FEATURE ENGINEERING (The Brain)
# ==========================================
print("⏳ Converting text to AI numbers (Embeddings)... This might take 1-2 mins.")
encoder = SentenceTransformer('all-mpnet-base-v2')
X_text = encoder.encode(df['clean_text'].tolist(), show_progress_bar=True)

# Process 'Experience'
X_exp = df['Experience'].fillna(0).astype(int).values.reshape(-1, 1)

# Process 'JobRole_Applied'
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_role = ohe.fit_transform(df[['JobRole_Applied']])

# Combine all features into one big matrix
X = np.hstack([X_text, X_exp, X_role])
y = df['target'].values
gender = df['gender_numeric'].values 

# ==========================================
# 5. TRAIN/TEST SPLIT & TRAINING
# ==========================================
X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
    X, y, gender, test_size=0.2, stratify=y, random_state=42
)

print("⏳ Training Random Forest Model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# ==========================================
# 6. EVALUATION & FAIRNESS
# ==========================================
y_pred = rf_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n🎯 Model Accuracy: {acc * 100:.2f}%")
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

print("\n--- ⚖️ Fairness Analysis ---")
test_df = pd.DataFrame({'y_pred': y_pred, 'gender': g_test})
selection_rates = test_df.groupby('gender')['y_pred'].mean()

male_rate = selection_rates.get(0, 0)
female_rate = selection_rates.get(1, 0)

print(f"Selection Rate (Male):   {male_rate:.2%}")
print(f"Selection Rate (Female): {female_rate:.2%}")

if male_rate > 0:
    impact_ratio = female_rate / male_rate
    print(f"Disparate Impact Ratio: {impact_ratio:.2f}")
    if impact_ratio < 0.8:
        print("⚠️ WARNING: Bias Detected! (Female selection is < 80% of Male selection)")
    else:
        print("✅ Model looks fair.")

# ==========================================
# 7. SAVE ARTIFACTS
# ==========================================
print("\n💾 Saving model to 'models/' folder...")
joblib.dump(rf_model, os.path.join(MODEL_DIR, 'rf_model.pkl'))
joblib.dump(ohe, os.path.join(MODEL_DIR, 'ohe_encoder.pkl'))
print("🎉 DONE! Model saved successfully.") 