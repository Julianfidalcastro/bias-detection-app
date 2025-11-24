import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import os
from sentence_transformers import SentenceTransformer

# LIME for Explainability
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components

# ==========================================
# 1. LOAD ASSETS
# ==========================================
@st.cache_resource
def load_assets():
    # Path setup to find models
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    model_path = os.path.join(project_root, 'models', 'rf_model.pkl')
    encoder_path = os.path.join(project_root, 'models', 'ohe_encoder.pkl')
    
    # Load Artifacts
    model = joblib.load(model_path)
    ohe_encoder = joblib.load(encoder_path) # We need this for roles
    text_encoder = SentenceTransformer('all-mpnet-base-v2')
    
    return model, ohe_encoder, text_encoder

model, ohe_encoder, text_encoder = load_assets()

# ==========================================
# 2. HELPER: PREDICTION PIPELINE FOR LIME
# ==========================================
# LIME sends a list of texts. We must turn them into the exact format the RF model expects.
# We 'freeze' the experience and role inputs so LIME only tests the text impact.
def prediction_wrapper(text_list):
    # 1. Encode Texts (List of strings -> Embeddings)
    cleaned_texts = [str(t).lower() for t in text_list]
    text_embeds = text_encoder.encode(cleaned_texts)
    
    # 2. Create Features for Experience & Role (Using the user's current selection)
    # We use the global variables 'current_exp' and 'current_role_vec' defined in the main block
    # This replicates the user's inputs for every text variation LIME generates.
    
    num_samples = len(text_list)
    
    # Repeat the experience value for all samples
    exp_matrix = np.full((num_samples, 1), current_exp)
    
    # Repeat the role vector for all samples
    role_matrix = np.tile(current_role_vec, (num_samples, 1))
    
    # 3. Combine into final matrix
    final_features = np.hstack([text_embeds, exp_matrix, role_matrix])
    
    # 4. Return Probabilities
    return model.predict_proba(final_features)

# ==========================================
# 3. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="Bias Detector (XAI)", layout="wide")

st.title("🧠 XAI-Powered Bias Detection")
st.markdown("Using **LIME** to explain why a resume was accepted or rejected.")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📄 Input Analysis")
    user_input = st.text_area("Paste Resume / JD Text:", height=150, 
                             value="He is a dominant leader looking for a young team.")
    
    c1, c2 = st.columns(2)
    exp_input = c1.slider("Experience (Years)", 0, 20, 5)
    role_input = c2.selectbox("Role", ["Data Scientist", "Software Engineer", "Sales Manager", "HR Specialist"])

    if st.button("Analyze with AI", type="primary"):
        if user_input:
            with st.spinner("🤖 Running LIME Explanation (This takes 5-10 seconds)..."):
                
                # --- PREPARE DATA FOR WRAPPER ---
                # We need to make these available to the wrapper function above
                global current_exp, current_role_vec
                current_exp = exp_input
                
                # Transform Role using the saved OneHotEncoder
                # (Must be a dataframe to match training format)
                role_df = pd.DataFrame([[role_input]], columns=['JobRole_Applied'])
                current_role_vec = ohe_encoder.transform(role_df)

                # --- GET PREDICTION ---
                # We pass a list of 1 item (the user input) to get the main prediction
                probs = prediction_wrapper([user_input])
                hiring_prob = probs[0][1] # Probability of Class 1 (Selected)
                
                # --- RUN LIME EXPLAINER ---
                explainer = LimeTextExplainer(class_names=['Rejected', 'Selected'])
                
                # Ask LIME to create 100 variations of the text to test the model
                exp = explainer.explain_instance(
                    user_input, 
                    prediction_wrapper, 
                    num_features=6,  # Highlight top 6 most important words
                    num_samples=100  # Lower this number if it's too slow (e.g., 50)
                )

                # --- DISPLAY RESULTS ---
                st.divider()
                st.subheader("📊 Decision Report")
                
                m1, m2 = st.columns(2)
                m1.metric("Hiring Probability", f"{hiring_prob*100:.1f}%", 
                         delta="Strong Candidate" if hiring_prob > 0.7 else "Risk of Rejection")
                
                # Interpret Bias
                if hiring_prob < 0.5:
                    st.error("🚨 Prediction: REJECTED")
                else:
                    st.success("✅ Prediction: SELECTED")

                st.markdown("### 🔍 Explainable AI (LIME) Output")
                st.info("The AI highlighted words that pushed the score UP (Green) or DOWN (Red).")
                
                # Render LIME HTML in Streamlit
                # We use a scrollable container for the visualization
                components.html(exp.as_html(), height=400, scrolling=True)

        else:
            st.warning("Please enter text.")

with col2:
    st.subheader("ℹ️ How LIME Works")
    st.markdown("""
    **LIME** (Local Interpretable Model-agnostic Explanations) doesn't use a dictionary.
    
    1. It takes your text.
    2. It generates **100 variations** by removing random words.
    3. It checks how the **Hiring Probability** changes.
    4. If removing **"Young"** makes the score go UP, then "Young" is marked as a **Negative Factor (Red)**.
    
    

[Image of LIME text explanation diagram]

    
    **Legend:**
    - 🟢 **Green:** Helps the candidate.
    - 🔴 **Red:** Hurts the candidate (Bias/Risk).
    """)