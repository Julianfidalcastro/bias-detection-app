import pandas as pd
import os

# Check the V2 file (the one you are training on)
path = 'data/recruitment_data_v2.csv'

if os.path.exists(path):
    df = pd.read_csv(path)
    print("--- DATA DIAGNOSIS ---")
    print(f"Total Rows: {len(df)}")
    print("Label Distribution:")
    print(df['HiringDecision'].value_counts())
    print("----------------------")
else:
    print("❌ File not found!")