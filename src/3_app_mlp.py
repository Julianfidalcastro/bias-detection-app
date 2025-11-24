import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import os
from sentence_transformers import SentenceTransformer
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components

# ==========================================
# 1. LOAD NEURAL NET ASSETS
# ==========================================
@st.cache_resource
def load_assets():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # Load MLP specific files
    model_path = os.path.join(project_root, 'models', 'mlp_model.pkl')
    encoder_path = os.path.join(project_root, 'models', 'mlp_ohe_encoder.pkl')
    scaler_path = os.path.join(project_root, 'models', 'mlp_scaler.pkl')
    
    model = joblib.load(model_path)
    ohe_encoder = joblib.load(encoder_path)
    scaler = joblib.load(scaler_path)
    text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    return model, ohe_encoder, scaler, text_encoder

# Load them
model, ohe_encoder, scaler, text_encoder = load_assets()

# ==========================================
# 2. PREDICTION PIPELINE (With Scaling)
# ==========================================
def prediction_wrapper(text_list):
    # 1. Text to Vector
    cleaned_texts = [str(t).lower() for t in text_list]
    text_embeds = text_encoder.encode(cleaned_texts)
    
    # 2. Tabular Features
    num_samples = len(text_list)
    exp_matrix = np.full((num_samples, 1), current_exp)
    role_matrix = np.tile(current_role_vec, (num_samples, 1))
    
    # 3. Combine
    final_features = np.hstack([text_embeds, exp_matrix, role_matrix])
    
    # 4. SCALE (Unique to MLP)
    final_features_scaled = scaler.transform(final_features)
    
    return model.predict_proba(final_features_scaled)

# ==========================================
# 3. UI
# ==========================================
st.set_page_config(page_title="Bias Detector (Neural Net)", layout="wide", page_icon="🧠")

st.title("🧠 Neural Network Bias Detection")
st.caption("Powered by MLP Classifier + MiniLM Embeddings")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Analyze Candidate")
    user_input = st.text_area("Paste Resume Text:", height=150, 
                             value="She is a collaborative HR Manager with 5 years of experience.")
    
    c1, c2 = st.columns(2)
    exp_input = c1.slider("Experience (Years)", 0, 20, 5)
    role_input = c2.selectbox("Role", ["HR Specialist", "Data Scientist", "Software Engineer", "Sales Manager"])

    if st.button("Run Neural Analysis", type="primary"):
        with st.spinner("🧠 Neural Net is thinking..."):
            # Setup Context
            global current_exp, current_role_vec
            current_exp = exp_input
            role_df = pd.DataFrame([[role_input]], columns=['JobRole_Applied'])
            current_role_vec = ohe_encoder.transform(role_df)

            # Predict
            probs = prediction_wrapper([user_input])
            hiring_prob = probs[0][1]
            
            # Explanation
            explainer = LimeTextExplainer(class_names=['Rejected', 'Selected'])
            exp = explainer.explain_instance(user_input, prediction_wrapper, num_features=6, num_samples=50)

            # Display
            st.divider()
            st.metric("Hiring Probability", f"{hiring_prob*100:.1f}%", 
                     delta="High Chance" if hiring_prob > 0.5 else "Low Chance")
            
            st.subheader("🔍 LIME Explanation (Neural Context)")
            components.html(exp.as_html(), height=400, scrolling=True)

with col2:
    st.info("ℹ️ **Model Architecture**")
    st.markdown("""
    This app uses a **Multi-Layer Perceptron (MLP)**.
    
    Unlike Random Forest, this model learns **non-linear relationships** between words. 
    
    It can understand that:
    * `Python` + `Expert` = ✅
    * `Python` + `Beginner` = ❌
    
    It is generally more accurate at detecting subtle bias.
    """)