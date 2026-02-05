import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import importlib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import torch

# --- PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '../data/processed/train.parquet')
sys.path.append(BASE_DIR)

# --- ENGINE RELOAD ---
import inference
importlib.reload(inference)
from inference import RiskInferenceEngine

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="FraudGuardian",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="collapsed"
)

# --- CSS STYLING ---
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Metrics Styling */
    div[data-testid="stMetric"] {
        background-color: #f8f9fa05;
        border: 1px solid #e6e6e620;
        padding: 15px;
        border-radius: 8px;
    }
    
    /* Buttons */
    .stButton button {
        height: 3rem;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Review Box Highlight */
    .review-box {
        border: 1px solid #ffc107;
        padding: 20px;
        border-radius: 10px;
        background-color: #fff3cd05;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. INITIALIZE RESOURCES ---
@st.cache_resource
def get_resources():
    engine = RiskInferenceEngine()
    if os.path.exists(DATA_PATH):
        dummy_df = pd.read_parquet(DATA_PATH).iloc[:1]
        dummy_cols = dummy_df.drop(['isFraud', 'TransactionID', 'TransactionDT'], axis=1).shape[1]
        engine.load_models(feature_count=dummy_cols)
    else:
        return None, None
    explainer = shap.TreeExplainer(engine.lgbm)
    return engine, explainer

try:
    engine, explainer = get_resources()
except Exception:
    st.error("System Core Failed to Load")
    st.stop()

# --- 2. GAUGE CHART ---
def plot_gauge(prob):
    color = "#28a745" # Green
    if prob > 0.5: color = "#ffc107" # Yellow
    if prob > 0.8: color = "#dc3545" # Red
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob * 100,
        number = {'suffix': "%", 'font': {'size': 24}},
        title = {'text': "Risk Probability", 'font': {'size': 14, 'color': "gray"}},
        gauge = {
            'axis': {'range': [0, 100], 'visible': False}, 
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 0,
            'steps': [{'range': [0, 100], 'color': "rgba(230, 230, 230, 0.1)"}],
        }))
    fig.update_layout(height=180, margin=dict(l=20, r=20, t=30, b=0), paper_bgcolor="rgba(0,0,0,0)")
    return fig

# --- MAIN UI ---
st.title("FraudGuardian Intelligence")
st.markdown("##### Enterprise Batch Processing & Forensics")
st.divider()

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload Transaction Log (CSV/Parquet)", type=["csv", "parquet"])

if 'batch_results' not in st.session_state: st.session_state.batch_results = None
if 'batch_data' not in st.session_state: st.session_state.batch_data = None

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'): batch_df = pd.read_csv(uploaded_file)
        else: batch_df = pd.read_parquet(uploaded_file)
        
        # Sanitization
        if 'isFraud' in batch_df.columns: batch_df = batch_df.drop('isFraud', axis=1)
        if 'TransactionID' in batch_df.columns:
            ids_batch = batch_df['TransactionID'].values
            X_batch = batch_df.drop(['TransactionID', 'TransactionDT'], axis=1, errors='ignore')
        else:
            ids_batch = np.arange(len(batch_df))
            X_batch = batch_df

        if st.button(f"Analyze {len(X_batch)} Transactions", type="primary"):
            progress_bar = st.progress(0)
            chunk_size = 100
            all_probs = []
            num_chunks = int(np.ceil(len(X_batch) / chunk_size))
            
            for i in range(num_chunks):
                start = i * chunk_size
                end = start + chunk_size
                probs = engine.predict(X_batch.iloc[start:end])
                all_probs.extend(probs)
                progress_bar.progress((i + 1) / num_chunks)
            
            # Store results
            st.session_state.batch_results = pd.DataFrame({
                'TransactionID': ids_batch,
                'Risk_Score': all_probs,
                'Verdict': ['BLOCK' if p > 0.8 else 'REVIEW' if p > 0.5 else 'APPROVE' for p in all_probs]
            })
            X_batch['TransactionID'] = ids_batch
            st.session_state.batch_data = X_batch
            st.rerun()

    except Exception as e:
        st.error(f"Error reading file: {e}")

# --- RESULTS DASHBOARD ---
if st.session_state.batch_results is not None:
    res_df = st.session_state.batch_results
    
    # 1. High Level Metrics & Download
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Scanned", len(res_df))
    c2.metric("Blocked (>80%)", len(res_df[res_df['Verdict'] == 'BLOCK']))
    
    review_count = len(res_df[res_df['Verdict'] == 'REVIEW'])
    c3.metric("Review Queue (50-80%)", review_count, delta="Action Required" if review_count > 0 else None, delta_color="inverse")
    c4.metric("Approved (<50%)", len(res_df[res_df['Verdict'] == 'APPROVE']))
    
    # --- DOWNLOAD BUTTON ---
    col_dl_1, col_dl_2 = st.columns([6, 1])
    with col_dl_2:
        csv = res_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Report",
            data=csv,
            file_name="fraud_risk_report.csv",
            mime="text/csv",
            type="secondary"
        )
    
    st.write("")

    # --- ENHANCED REVIEW QUEUE ---
    if review_count > 0:
        with st.container():
            st.markdown("### ⚠️ Manual Review Queue")
            st.info(f"There are {review_count} transactions in the gray zone (50-80% risk). Please adjudicate.")
            
            review_df = res_df[res_df['Verdict'] == 'REVIEW']
            
            # Split into List and Decision Box
            col_list, col_decision = st.columns([3, 2])
            
            with col_list:
                st.dataframe(
                    review_df[['TransactionID', 'Risk_Score']].style.background_gradient(subset=['Risk_Score'], cmap='Oranges'),
                    use_container_width=True, 
                    height=250
                )
            
            with col_decision:
                # Decision Box Logic
                st.markdown('<div class="review-box">', unsafe_allow_html=True)
                st.subheader("Review Console")
                
                # Dynamic Select Box (Auto-selects top risk)
                top_risk_id = review_df.sort_values('Risk_Score', ascending=False).iloc[0]['TransactionID']
                review_id = st.selectbox("Select Transaction ID", review_df['TransactionID'].unique(), index=0)
                
                # Show details for selected ID
                curr_score = review_df[review_df['TransactionID'] == review_id]['Risk_Score'].values[0]
                st.metric("Risk Score", f"{curr_score:.2%}")
                
                # Action Buttons
                btn_col1, btn_col2 = st.columns(2)
                if btn_col1.button("✅ Mark Safe", use_container_width=True):
                    idx = st.session_state.batch_results[st.session_state.batch_results['TransactionID'] == review_id].index
                    st.session_state.batch_results.loc[idx, 'Verdict'] = 'APPROVE'
                    st.toast(f"Transaction {review_id} Approved!", icon="✅")
                    st.rerun()
                    
                if btn_col2.button("⛔ Mark Fraud", type="primary", use_container_width=True):
                    idx = st.session_state.batch_results[st.session_state.batch_results['TransactionID'] == review_id].index
                    st.session_state.batch_results.loc[idx, 'Verdict'] = 'BLOCK'
                    st.toast(f"Transaction {review_id} Blocked!", icon="⛔")
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
                
        st.divider()

    # --- MAIN TABLE ---
    st.subheader("Full Transaction Register")
    st.dataframe(
        res_df.head(100).style.background_gradient(subset=['Risk_Score'], cmap='RdYlGn_r'),
        use_container_width=True,
        height=300
    )
    
    st.divider()

    # --- FORENSICS DEEP DIVE ---
    st.subheader("🔎 Forensics Deep Dive")
    
    col_search, col_act = st.columns([3, 1])
    with col_search:
        search_id = st.selectbox("Search Transaction ID", res_df['TransactionID'].unique())
    with col_act:
        investigate = st.button("Generate Risk Profile", type="secondary", use_container_width=True)
        
    if investigate:
        row_data = st.session_state.batch_data[st.session_state.batch_data['TransactionID'] == search_id]
        selected_row = row_data.drop(['TransactionID'], axis=1) 
        prob = res_df[res_df['TransactionID'] == search_id]['Risk_Score'].values[0]
        
        # --- NEW: RAW DATA DISPLAY ---
        with st.expander("📄 View Full Raw Data Row", expanded=True):
            st.dataframe(row_data, use_container_width=True)
        # -----------------------------
        
        st.write("")
        left, right = st.columns([1, 2], gap="large")
        
        with left:
            st.markdown("#### Verdict")
            st.plotly_chart(plot_gauge(prob), use_container_width=True)
            
            verdict = res_df[res_df['TransactionID'] == search_id]['Verdict'].values[0]
            if verdict == 'BLOCK': st.error("**Verdict: BLOCKED**")
            elif verdict == 'REVIEW': st.warning("**Verdict: REVIEW NEEDED**")
            else: st.success("**Verdict: APPROVED**")
            
            with st.expander("Model Consensus", expanded=True):
                lgbm_score = engine.lgbm.predict(selected_row.fillna(-999))[0]
                x_neural = selected_row.fillna(0).values.astype(np.float32)
                x_neural[x_neural < 0] = 0
                if hasattr(engine.tabnet_std, 'cat_idxs'):
                    for i, d in zip(engine.tabnet_std.cat_idxs, engine.tabnet_std.cat_dims):
                        x_neural[:, i] = np.clip(x_neural[:, i], 0, d - 1)
                
                tabnet_score = engine.tabnet_std.predict_proba(x_neural)[0, 1]
                x_tensor = torch.tensor(x_neural, dtype=torch.float32)
                with torch.no_grad():
                    _, mtl_raw = engine.tabnet_mtl(x_tensor[:, :50], x_tensor[:, 50:])
                mtl_score = mtl_raw.numpy().flatten()[0]
                
                st.markdown(f"**Gradient Boost:** `{lgbm_score:.2%}`")
                st.markdown(f"**Std TabNet:** `{tabnet_score:.2%}`")
                st.markdown(f"**Asymmetric MTL:** `{mtl_score:.2%}`")
                st.divider()
                st.markdown(f"**🏆 Ensemble:** `{prob:.2%}`")

        with right:
            st.markdown("#### Context & Logic")
            m1, m2 = st.columns(2)
            amt = selected_row['TransactionAmt'].values[0] if 'TransactionAmt' in selected_row.columns else 0
            card = selected_row['card4'].values[0] if 'card4' in selected_row.columns else "N/A"
            m1.metric("Amount", f"${amt:,.2f}")
            m2.metric("Card Type", str(card))
            
            st.write("")
            st.caption("SHAP Feature Impact")
            X_shap = selected_row.fillna(-999)
            shap_values = explainer.shap_values(X_shap)
            if isinstance(shap_values, list): sv = shap_values[1][0]; ev = explainer.expected_value[1]
            else: sv = shap_values[0]; ev = explainer.expected_value
            
            with plt.style.context("dark_background"):
                fig, ax = plt.subplots(figsize=(10, 3.5))
                fig.patch.set_alpha(0.0)
                ax.patch.set_alpha(0.0)
                shap.plots.waterfall(shap.Explanation(values=sv, base_values=ev, data=X_shap.iloc[0], feature_names=X_shap.columns), show=False, max_display=8)
                for text in ax.texts: text.set_color("white")
                ax.tick_params(colors='white', which='both')
                st.pyplot(fig, clear_figure=True)