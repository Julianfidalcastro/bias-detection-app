import streamlit as st
import joblib
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components

# 1. LOAD ASSETS
@st.cache_resource
def load_assets():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = joblib.load(os.path.join(root, 'models', 'mlp_text_only.pkl'))
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    return model, encoder

try:
    model, encoder = load_assets()
except:
    st.error("Model not found! Run 'python src/2_train_text_only.py' first.")
    st.stop()

# Wrapper
def prediction_wrapper(text_list):
    cleaned_texts = [str(t).lower() for t in text_list]
    vectors = encoder.encode(cleaned_texts)
    return model.predict_proba(vectors)

# Text Cleaning for Audit
def clean_text_func(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return " ".join(text.split())

# ==========================================
# SMART BIAS REMOVAL
# ==========================================
def smart_bias_removal(text, explainer_obj):
    identity_terms = [
        'he', 'she', 'him', 'her', 'his', 'hers', 'himself', 'herself',
        'man', 'woman', 'men', 'women', 'male', 'female',
        'mr', 'mrs', 'miss', 'ms', 'guy', 'guys', 'lady', 'ladies',
        'brother', 'sister', 'father', 'mother',
        'white', 'black', 'asian', 'hispanic', 'indian', 'american'
    ]
    
    map_exp = explainer_obj.as_list()
    
    cleaned_text = text
    removed_words = []
    kept_weaknesses = []
    
    for word, weight in map_exp:
        clean_word = re.sub(r'[^a-zA-Z]', '', word).lower()
        
        if clean_word in identity_terms:
            removed_words.append(clean_word)
            pattern = re.compile(r'\b' + re.escape(clean_word) + r'\b', re.IGNORECASE)
            cleaned_text = pattern.sub("████", cleaned_text)
        elif weight < 0:
            kept_weaknesses.append(clean_word)

    return cleaned_text, removed_words, kept_weaknesses

# UI CONFIG
st.set_page_config(page_title="Fair Hiring AI", layout="wide", page_icon="⚖️")
st.title("⚖️ AI Bias Detection")

# TABS
tab1, tab2 = st.tabs(["📝 Resume Check", "📉 Global Audit"])

# ==========================================
# TAB 1: SIMPLIFIED CHECK
# ==========================================
with tab1:
    st.subheader("Paste Resume Text")
    user_input = st.text_area("Input", height=150, label_visibility="collapsed",
                             value="He struggled with Python and has limited knowledge.")
    
    if st.button("Analyze Resume"):
        # 1. INTERNAL: Run Original Analysis (Hidden) to find bias
        with st.spinner("Processing..."):
            explainer = LimeTextExplainer(class_names=['Rejected', 'Selected'])
            # Fast mode (20 samples)
            exp_orig = explainer.explain_instance(user_input, prediction_wrapper, num_features=10, num_samples=20)
            
            # 2. INTERNAL: Remove Bias
            clean_text, removed_list, kept_list = smart_bias_removal(user_input, exp_orig)
            
            # 3. FINAL: Analyze Clean Text
            probs_new = prediction_wrapper([clean_text])
            score_new = probs_new[0][1]
            
            # Fast mode (20 samples) for final visual
            exp_new = explainer.explain_instance(clean_text, prediction_wrapper, num_features=10, num_samples=20)
            
        st.divider()
        
        # --- OUTPUT 1: FINAL DECISION ---
        st.subheader("Final Decision")
        if score_new > 0.5:
            st.success(f"✅ **SELECTED** (Confidence: {score_new*100:.1f}%)")
        else:
            st.error(f"❌ **REJECTED** (Confidence: {score_new*100:.1f}%)")
            
        # --- OUTPUT 2: HIGHLIGHTED TEXT ---
        st.subheader("Highlighted Factors (Why?)")
        st.caption("Orange = Positive Impact, Blue = Negative Impact")
        components.html(exp_new.as_html(), height=300, scrolling=True)
        
        # Small note on what was removed (Optional, can remove if you want it 100% clean)
        if removed_list:
            st.info(f"Note: Identity bias terms removed before evaluation: {', '.join(set(removed_list))}")

# ==========================================
# TAB 2: GLOBAL AUDIT
# ==========================================
with tab2:
    st.subheader("📉 Global Hiring Pattern Analysis")
    
    col_controls, col_metrics = st.columns([1, 3])
    
    with col_controls:
        bias_target = st.radio("Select Bias Factor:", ["Gender", "Race", "Age Group"])
        
        if st.button("🚀 Run Audit"):
            run_audit = True
        else:
            run_audit = False

    if run_audit:
        with st.spinner(f"Scanning dataset for {bias_target} bias..."):
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_path = os.path.join(root, 'data', 'recruitment_data_biased.csv')
            
            if os.path.exists(data_path):
                df = pd.read_csv(data_path)
                
                if bias_target == "Age Group":
                    df['Age Group'] = pd.cut(df['Experience'], bins=[-1, 5, 15, 100], labels=['Junior', 'Mid-Level', 'Senior'])
                
                df['clean_text'] = df['ResumeText'].apply(clean_text_func)
                vectors = encoder.encode(df['clean_text'].tolist())
                predictions = model.predict(vectors)
                df['AI_Decision'] = predictions
                
                stats = df.groupby(bias_target)['AI_Decision'].mean().reset_index()
                stats['Selection Rate %'] = stats['AI_Decision'] * 100
                stats = stats.sort_values('Selection Rate %', ascending=False)
                
                with col_metrics:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.barplot(data=stats, x=bias_target, y='Selection Rate %', palette='viridis')
                    plt.title(f"Hiring Probability by {bias_target}")
                    plt.ylabel("Selection Rate (%)")
                    st.pyplot(fig)

                    best_group = stats.iloc[0][bias_target]
                    worst_group = stats.iloc[-1][bias_target]
                    gap = stats.iloc[0]['Selection Rate %'] - stats.iloc[-1]['Selection Rate %']
                        
                                    
                                    # See streamlit-related files inside env\Scripts
                                  
                    st.error(f"🚨 **Favoritism Detected:** {best_group} are {gap:.1f}% more likely to be hired than {worst_group}.")
            else:
                st.error("Dataset not found. Run 'src/0_generate_biased_data.py' first.")