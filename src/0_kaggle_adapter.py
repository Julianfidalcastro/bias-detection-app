import pandas as pd
import random
import re
import os

# SETUP PATHS
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
kaggle_file = os.path.join(project_root, 'data', 'UpdatedResumeDataSet.csv') 
output_file = os.path.join(project_root, 'data', 'recruitment_data_v2.csv')

print(f"📂 Reading Kaggle file: {kaggle_file}...")
try:
    df = pd.read_csv(kaggle_file)
except FileNotFoundError:
    # Fallback to standard name if not found
    df = pd.read_csv(os.path.join(project_root, 'data', 'recruitment_dataset.csv'))

# RENAME COLUMNS
if 'Resume' in df.columns:
    df = df.rename(columns={'Resume': 'ResumeText', 'Category': 'JobRole_Applied'})

# INFER GENDER (Randomized for demo)
df['Gender'] = [random.choice(['Male', 'Female']) for _ in range(len(df))]

# EXTRACT EXPERIENCE
def extract_exp(text):
    match = re.search(r'(\d+)\+?\s?years?', str(text).lower())
    return int(match.group(1)) if match else random.randint(1, 10)

df['Experience'] = df['ResumeText'].apply(extract_exp)

# =====================================================
# THE "FORCE BALANCE" LOGIC
# =====================================================
def calculate_score(text):
    text = str(text).lower()
    score = 0
    # 1. Length Reward
    if len(text.split()) > 50: score += 50
    # 2. Keyword Reward
    keywords = ['data', 'java', 'python', 'management', 'sales', 'communication', 'sql', 'aws', 'analysis']
    for word in keywords:
        if word in text: score += 10
    # 3. Bias Penalty
    if 'aggressive' in text or 'ninja' in text: score -= 50
    
    # Add Noise
    score += random.randint(0, 20)
    return score

# 1. Calculate scores for EVERYONE
df['Score'] = df['ResumeText'].apply(calculate_score)

# 2. Find the Median Score (The middle point)
median_score = df['Score'].median()

# 3. Force Split: Anyone above median is Selected, below is Rejected
# This GUARANTEES a 50/50 split (or very close to it)
df['HiringDecision'] = df['Score'].apply(lambda x: 'Selected' if x >= median_score else 'Rejected')

# Save
df.drop(columns=['Score'], inplace=True) # Clean up temp column
df.to_csv(output_file, index=False)

print(f"✅ FORCED BALANCE APPLIED!")
print(df['HiringDecision'].value_counts())
print(f"💾 Saved to {output_file}")