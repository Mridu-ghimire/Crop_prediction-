"""
admin.py  —  Professional Admin Dashboard
Crop Prediction System | Mridu Ghimire | 77466817
Run: streamlit run admin.py --server.port 8502
"""
import streamlit as st
import pandas as pd
import numpy as np
import os, json, hashlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from sklearn.base import clone
import warnings
warnings.filterwarnings("ignore")

from data import load_data, load_uploaded_data, dataset_summary, get_feature_target, FEATURE_NAMES
from model import (
    MODELS, get_train_test, evaluate_model, cross_validate_model,
    train_and_compare_all, get_feature_importance,
    get_confusion_matrix, get_classification_report,
    get_error_analysis, save_model,
)

# ── Constants ──────────────────────────────────────────────────────
ADMIN_USER      = "admin"
ADMIN_HASH      = hashlib.sha256("admin123".encode()).hexdigest()
LOGS_FILE       = "prediction_logs.json"
DATASET_PATH    = "Crop_recommendation.csv"
CUSTOM_DATASET  = "admin_dataset.csv"

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Admin Dashboard — Crop Prediction",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
# DESIGN SYSTEM  (exact colours from spec)
# ══════════════════════════════════════════════════════════════════
COLORS = {
    "bg":       "#0F172A",
    "sidebar":  "#1E293B",
    "card":     "#1E293B",
    "card2":    "#334155",
    "border":   "#334155",
    "accent":   "#38BDF8",
    "accent2":  "#0EA5E9",
    "green":    "#4ADE80",
    "yellow":   "#FBBF24",
    "red":      "#F87171",
    "text":     "#F1F5F9",
    "muted":    "#94A3B8",
    "chart_bg": "#1E293B",
}

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
  * {{ font-family: 'Inter', sans-serif !important; }}

  /* ── Base ── */
  .stApp {{ background: {COLORS['bg']} !important; color: {COLORS['text']} !important; }}
  .main .block-container {{ padding: 0 2rem 2rem 2rem !important; max-width: 1400px; }}

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {{
    background: {COLORS['sidebar']} !important;
    border-right: 1px solid {COLORS['border']} !important;
    width: 240px !important;
  }}
  section[data-testid="stSidebar"] * {{ color: {COLORS['muted']} !important; }}
  section[data-testid="stSidebar"] .stRadio > div > label {{
    padding: 10px 14px !important;
    border-radius: 8px !important;
    margin: 2px 0 !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    transition: all 0.15s !important;
    display: block !important;
  }}
  section[data-testid="stSidebar"] .stRadio > div > label:hover {{
    background: {COLORS['card2']} !important;
    color: {COLORS['accent']} !important;
  }}

  /* ── Metric cards ── */
  div[data-testid="metric-container"] {{
    background: {COLORS['card']} !important;
    border: 1px solid {COLORS['border']} !important;
    border-radius: 16px !important;
    padding: 20px 24px !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3) !important;
  }}
  div[data-testid="metric-container"] label {{
    color: {COLORS['muted']} !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
  }}
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] {{
    color: {COLORS['accent']} !important;
    font-size: 1.9rem !important;
    font-weight: 800 !important;
  }}

  /* ── Buttons ── */
  .stButton > button {{
    background: linear-gradient(135deg, {COLORS['accent2']}, #0369A1) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 11px 26px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 14px rgba(14,165,233,0.3) !important;
  }}
  .stButton > button:hover {{
    background: linear-gradient(135deg, {COLORS['accent']}, {COLORS['accent2']}) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(56,189,248,0.4) !important;
  }}

  /* ── File uploader fix ── */
  [data-testid="stFileUploaderDropzone"] {{
    background: {COLORS['card2']} !important;
    border: 1px dashed {COLORS['border']} !important;
    border-radius: 12px !important;
    padding: 20px !important;
  }}
  [data-testid="stFileUploaderDropzone"] button {{
    background: {COLORS['card']} !important;
    border: 1px solid {COLORS['accent']} !important;
    border-radius: 8px !important;
    color: {COLORS['accent']} !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 7px 18px !important;
    cursor: pointer !important;
    font-family: 'Inter', sans-serif !important;
    box-shadow: none !important;
    transform: none !important;
    transition: background 0.2s !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    position: relative !important;
    overflow: hidden !important;
    white-space: nowrap !important;
  }}
  [data-testid="stFileUploaderDropzone"] button:hover {{
    background: {COLORS['card2']} !important;
    transform: none !important;
    box-shadow: none !important;
  }}
  [data-testid="stFileUploaderDropzone"] button > span:nth-child(2),
  [data-testid="stFileUploaderDropzone"] button > div:nth-child(2) {{
    display: none !important;
  }}
  [data-testid="stFileUploaderDropzone"] small,
  [data-testid="stFileUploaderDropzone"] span {{
    color: {COLORS['muted']} !important;
    font-size: 0.8rem !important;
  }}

  /* ── Expander fix ── */
  [data-testid="stExpander"] {{
    background: {COLORS['card']} !important;
    border: 1px solid {COLORS['border']} !important;
    border-radius: 12px !important;
    overflow: hidden !important;
  }}
  [data-testid="stExpander"] > div > div > div > button {{
    background: {COLORS['card']} !important;
    border: none !important;
    box-shadow: none !important;
    transform: none !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    width: 100% !important;
    padding: 14px 18px !important;
    color: {COLORS['accent']} !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    font-family: 'Inter', sans-serif !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
  }}
  [data-testid="stExpander"] > div > div > div > button:hover {{
    background: {COLORS['card2']} !important;
    transform: none !important;
    box-shadow: none !important;
  }}
  [data-testid="stExpander"] > div > div > div > button > span:nth-child(2),
  [data-testid="stExpander"] > div > div > div > button > div:nth-child(2) {{
    display: none !important;
  }}
  [data-testid="stExpander"] > div > div > div > button svg {{
    color: {COLORS['accent']} !important;
    flex-shrink: 0 !important;
    min-width: 16px !important;
  }}

  /* ── Inputs ── */
  .stTextInput > div > div > input,
  .stSelectbox > div > div {{
    background: {COLORS['card2']} !important;
    border: 1px solid {COLORS['border']} !important;
    border-radius: 10px !important;
    color: {COLORS['text']} !important;
  }}
  .stTextInput > div > div > input:focus {{
    border-color: {COLORS['accent']} !important;
    box-shadow: 0 0 0 2px rgba(56,189,248,0.2) !important;
  }}

  /* ── DataFrames ── */
  .stDataFrame {{ border-radius: 12px !important; overflow: hidden !important; }}

  /* ── Headings ── */
  h1 {{ color: {COLORS['text']} !important; font-weight: 800 !important; font-size: 1.7rem !important; }}
  h2 {{ color: {COLORS['text']} !important; font-weight: 700 !important; }}
  h3 {{ color: {COLORS['accent']} !important; font-weight: 600 !important; }}
  hr {{ border-color: {COLORS['border']} !important; margin: 1.2rem 0 !important; }}

  /* ── Alerts ── */
  div[data-testid="stSuccessMessage"] {{ background: #052e16 !important; border-color: #16a34a !important; }}
  div[data-testid="stInfoMessage"]    {{ background: #0c1a2e !important; border-color: {COLORS['accent']} !important; }}
  div[data-testid="stErrorMessage"]   {{ background: #2a0a0a !important; border-color: #dc2626 !important; }}
  div[data-testid="stWarningMessage"] {{ background: #2a1a00 !important; border-color: #d97706 !important; }}

  /* ── Scrollbar ── */
  ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
  ::-webkit-scrollbar-track {{ background: {COLORS['bg']}; }}
  ::-webkit-scrollbar-thumb {{ background: {COLORS['border']}; border-radius: 3px; }}
  ::-webkit-scrollbar-thumb:hover {{ background: {COLORS['accent']}; }}

  /* ── Custom components ── */
  .stat-card {{
    background: {COLORS['card']};
    border: 1px solid {COLORS['border']};
    border-radius: 16px;
    padding: 22px 24px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.25);
    position: relative;
    overflow: hidden;
  }}
  .stat-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, {COLORS['accent']}, {COLORS['accent2']});
  }}
  .stat-val  {{ color: {COLORS['accent']}; font-size: 2rem; font-weight: 800; line-height: 1.1; }}
  .stat-lbl  {{ color: {COLORS['muted']}; font-size: 0.72rem; font-weight: 600;
                text-transform: uppercase; letter-spacing: 0.08em; margin-top: 6px; }}
  .stat-icon {{ font-size: 1.8rem; margin-bottom: 8px; }}

  .chart-card {{
    background: {COLORS['card']};
    border: 1px solid {COLORS['border']};
    border-radius: 16px;
    padding: 20px 24px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.2);
    margin: 0 0 20px 0;
  }}
  .chart-title {{
    color: {COLORS['text']};
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
  }}

  .section-label {{
    color: {COLORS['muted']};
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 24px 0 12px 0;
  }}

  .badge-blue   {{ background: rgba(56,189,248,0.15); color: {COLORS['accent']};
                   border: 1px solid rgba(56,189,248,0.3); border-radius: 20px;
                   padding: 3px 12px; font-size: 0.78rem; font-weight: 600; }}
  .badge-green  {{ background: rgba(74,222,128,0.15); color: {COLORS['green']};
                   border: 1px solid rgba(74,222,128,0.3); border-radius: 20px;
                   padding: 3px 12px; font-size: 0.78rem; font-weight: 600; }}
  .badge-yellow {{ background: rgba(251,191,36,0.15); color: {COLORS['yellow']};
                   border: 1px solid rgba(251,191,36,0.3); border-radius: 20px;
                   padding: 3px 12px; font-size: 0.78rem; font-weight: 600; }}
  .badge-red    {{ background: rgba(248,113,113,0.15); color: {COLORS['red']};
                   border: 1px solid rgba(248,113,113,0.3); border-radius: 20px;
                   padding: 3px 12px; font-size: 0.78rem; font-weight: 600; }}

  .log-row {{
    background: {COLORS['card2']};
    border: 1px solid {COLORS['border']};
    border-radius: 10px;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 0.85rem;
    color: {COLORS['muted']};
    display: flex;
    align-items: center;
    gap: 12px;
  }}
  .log-crop {{ color: {COLORS['accent']}; font-weight: 700; }}
  .log-time {{ color: {COLORS['muted']}; font-size: 0.78rem; margin-left: auto; }}

  .page-header {{
    padding: 24px 0 16px 0;
    border-bottom: 1px solid {COLORS['border']};
    margin-bottom: 24px;
  }}
  .page-title {{ color: {COLORS['text']}; font-size: 1.6rem; font-weight: 800; margin: 0; }}
  .page-sub   {{ color: {COLORS['muted']}; font-size: 0.88rem; margin: 4px 0 0 0; }}

  .val-check {{
    background: {COLORS['card2']};
    border: 1px solid {COLORS['border']};
    border-radius: 10px;
    padding: 12px 16px;
    margin: 5px 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.88rem;
    color: {COLORS['text']};
  }}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════
def load_logs():
    if not os.path.exists(LOGS_FILE): return []
    try:
        with open(LOGS_FILE) as f: return json.load(f)
    except: return []

def save_log(e):
    logs = load_logs(); logs.append(e)
    with open(LOGS_FILE, "w") as f: json.dump(logs, f, indent=2)

def clear_logs():
    if os.path.exists(LOGS_FILE): os.remove(LOGS_FILE)

def active_dataset():
    return CUSTOM_DATASET if os.path.exists(CUSTOM_DATASET) else DATASET_PATH

def plotly_cfg():
    return dict(
        paper_bgcolor=COLORS["chart_bg"],
        plot_bgcolor=COLORS["chart_bg"],
        font=dict(color=COLORS["text"], family="Inter"),
        margin=dict(t=30, b=40, l=40, r=20),
        xaxis=dict(gridcolor=COLORS["border"], linecolor=COLORS["border"]),
        yaxis=dict(gridcolor=COLORS["border"], linecolor=COLORS["border"]),
    )

# ══════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════
for k, v in {
    "admin_in": False,
    "trained":  None,
    "t_name":   None,
    "t_metrics":None,
    "t_fi":     None,
}.items():
    if k not in st.session_state: st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════
# LOGIN
# ══════════════════════════════════════════════════════════════════
def show_login():
    st.markdown("""
    <div style='text-align:center; padding:60px 0 32px;'>
      <div style='font-size:4rem; margin-bottom:14px;'>🛡️</div>
      <div style='color:#F1F5F9; font-size:1.8rem; font-weight:800; margin-bottom:6px;'>Admin Dashboard</div>
      <div style='color:#94A3B8; font-size:0.92rem;'>Crop Prediction System — Administrator Access</div>
    </div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1, 1])
    with col:
        st.markdown(f"""
        <div style='background:{COLORS["card"]};border:1px solid {COLORS["border"]};
                    border-radius:20px;padding:36px;box-shadow:0 20px 60px rgba(0,0,0,0.5);'>
        """, unsafe_allow_html=True)
        st.markdown(f"<div style='color:{COLORS['accent']};font-weight:700;font-size:1rem;margin-bottom:16px;'>🔐 Sign In</div>", unsafe_allow_html=True)
        u = st.text_input("Username", key="au", placeholder="admin")
        p = st.text_input("Password", type="password", key="ap", placeholder="••••••••")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Login to Dashboard", use_container_width=True):
            if u == ADMIN_USER and hashlib.sha256(p.encode()).hexdigest() == ADMIN_HASH:
                st.session_state.admin_in = True; st.rerun()
            else:
                st.error("Invalid credentials.")
        st.markdown("</div>", unsafe_allow_html=True)

if not st.session_state.admin_in:
    show_login(); st.stop()

# ══════════════════════════════════════════════════════════════════
# CACHED DATA
# ══════════════════════════════════════════════════════════════════
@st.cache_data
def _load_default(): return load_data(DATASET_PATH)

@st.cache_data
def _compare_all():
    df = load_data(active_dataset())
    return train_and_compare_all(df)

def load_active(): return load_data(active_dataset())

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style='padding:24px 16px 16px; text-align:center;'>
      <div style='font-size:2.4rem; margin-bottom:8px;'>🛡️</div>
      <div style='color:{COLORS["text"]}; font-weight:800; font-size:0.95rem; letter-spacing:0.02em;'>Admin Dashboard</div>
      <div style='color:{COLORS["muted"]}; font-size:0.7rem; margin-top:3px; letter-spacing:0.06em;'>CROP PREDICTION SYSTEM</div>
    </div>
    <hr style='border-color:{COLORS["border"]}; margin:0 0 8px;'>
    """, unsafe_allow_html=True)

    nav = st.radio("", [
        "📊  Dashboard",
        "📂  Dataset",
        "🤖  Models",
        "📋  Logs",
        "📈  Analytics",
        "🧪  Testing",
    ], label_visibility="collapsed")

    st.markdown(f"<hr style='border-color:{COLORS['border']}; margin:12px 0;'>", unsafe_allow_html=True)

    # Active dataset badge
    is_custom = os.path.exists(CUSTOM_DATASET)
    badge_cls = "badge-green" if is_custom else "badge-blue"
    badge_txt = "Custom Dataset" if is_custom else "Default Dataset"
    st.markdown(f"""
    <div style='padding:0 8px;'>
      <div style='color:{COLORS["muted"]};font-size:0.7rem;font-weight:700;
                  text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;'>Active Dataset</div>
      <span class='{badge_cls}'>{badge_txt}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<hr style='border-color:{COLORS['border']}; margin:12px 0;'>", unsafe_allow_html=True)
    st.markdown(f"<div style='padding:0 8px;color:{COLORS['muted']};font-size:0.82rem;'>👤 <strong style='color:{COLORS['text']};'>Administrator</strong></div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚪  Logout", use_container_width=True):
        st.session_state.admin_in = False; st.rerun()


# ══════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════
if nav == "📊  Dashboard":

    st.markdown(f"""
    <div class='page-header'>
      <div class='page-title'>📊 Dashboard</div>
      <div class='page-sub'>System overview — predictions, models, and dataset insights</div>
    </div>
    """, unsafe_allow_html=True)

    df   = load_active()
    logs = load_logs()
    logs_df = pd.DataFrame(logs) if logs else pd.DataFrame()

    # ── STAT CARDS ────────────────────────────────────────────────
    total_preds = len(logs)
    most_crop   = logs_df["Top Crop"].value_counts().index[0] if not logs_df.empty and "Top Crop" in logs_df else "—"
    avg_conf    = logs_df["Confidence"].str.replace("%","").astype(float).mean() if not logs_df.empty and "Confidence" in logs_df else 0

    # Get best model accuracy from cached comparison
    try:
        res_df, _, best_name, best_model, X_test, y_test = _compare_all()
        best_acc = f"{res_df['Accuracy'].max():.1f}%"
        active_model = best_name.split()[0] + " " + (best_name.split()[1] if len(best_name.split()) > 1 else "")
    except:
        best_acc = "—"
        active_model = "Random Forest"

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        st.markdown(f"""
        <div class='stat-card'>
          <div class='stat-icon'>🔮</div>
          <div class='stat-val'>{total_preds}</div>
          <div class='stat-lbl'>Total Predictions</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='stat-card'>
          <div class='stat-icon'>🎯</div>
          <div class='stat-val'>{best_acc}</div>
          <div class='stat-lbl'>Best Model Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='stat-card'>
          <div class='stat-icon'>🤖</div>
          <div class='stat-val' style='font-size:1.3rem;'>{active_model}</div>
          <div class='stat-lbl'>Active Model</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class='stat-card'>
          <div class='stat-icon'>🌾</div>
          <div class='stat-val' style='font-size:1.3rem;'>{most_crop}</div>
          <div class='stat-lbl'>Most Predicted Crop</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── MAIN CHART: Crop Prediction Frequency ─────────────────────
    col_main, col_side = st.columns([2, 1], gap="large")

    with col_main:
        st.markdown(f"<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='chart-title'>📊 Crop Prediction Frequency</div>", unsafe_allow_html=True)

        if not logs_df.empty and "Top Crop" in logs_df.columns:
            freq = logs_df["Top Crop"].value_counts().reset_index()
            freq.columns = ["Crop", "Count"]
        else:
            # Show dataset distribution as fallback
            freq = df["label"].value_counts().reset_index()
            freq.columns = ["Crop", "Count"]
            st.caption("Showing dataset distribution (no predictions yet)")

        fig = px.bar(
            freq.head(10), x="Crop", y="Count",
            color="Count",
            color_continuous_scale=[[0, "#1E3A5F"], [1, "#38BDF8"]],
            template="plotly_dark",
            text="Count",
        )
        fig.update_traces(textposition="outside", textfont_color=COLORS["text"])
        fig.update_layout(**plotly_cfg(), coloraxis_showscale=False,
                          xaxis_tickangle=-30, height=320,
                          yaxis_range=[0, freq["Count"].max() * 1.2])
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_side:
        st.markdown(f"<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='chart-title'>🥧 Crop Share</div>", unsafe_allow_html=True)
        top5 = freq.head(5)
        fig_pie = px.pie(
            top5, names="Crop", values="Count",
            color_discrete_sequence=["#38BDF8","#0EA5E9","#0284C7","#0369A1","#075985"],
            template="plotly_dark",
            hole=0.5,
        )
        fig_pie.update_layout(
            paper_bgcolor=COLORS["chart_bg"],
            font=dict(color=COLORS["text"], family="Inter"),
            margin=dict(t=10, b=10, l=10, r=10),
            legend=dict(bgcolor=COLORS["chart_bg"], font_size=11),
            height=320,
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── BOTTOM: Input trends + Model comparison ────────────────────
    col_b1, col_b2 = st.columns(2, gap="large")

    with col_b1:
        st.markdown(f"<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='chart-title'>🌧️ Rainfall & Temperature by Crop (Top 8)</div>", unsafe_allow_html=True)
        top8 = df["label"].value_counts().head(8).index.tolist()
        df8  = df[df["label"].isin(top8)]
        avg8 = df8.groupby("label")[["rainfall","temperature"]].mean().reset_index()
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Bar(name="Rainfall (mm)", x=avg8["label"], y=avg8["rainfall"],
                                   marker_color="#38BDF8", opacity=0.85))
        fig_trend.add_trace(go.Bar(name="Temperature (°C)", x=avg8["label"], y=avg8["temperature"],
                                   marker_color="#FBBF24", opacity=0.85))
        fig_trend.update_layout(**plotly_cfg(), barmode="group", height=280,
                                 legend=dict(bgcolor=COLORS["chart_bg"], font_size=11),
                                 xaxis_tickangle=-30)
        st.plotly_chart(fig_trend, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b2:
        st.markdown(f"<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='chart-title'>🤖 Model Accuracy Comparison</div>", unsafe_allow_html=True)
        try:
            acc_df = res_df["Accuracy"].reset_index()
            acc_df.columns = ["Model", "Accuracy"]
            colors_bar = [COLORS["accent"] if i == 0 else COLORS["card2"]
                          for i in range(len(acc_df))]
            fig_acc = go.Figure(go.Bar(
                x=acc_df["Model"], y=acc_df["Accuracy"],
                marker_color=colors_bar,
                text=acc_df["Accuracy"].apply(lambda x: f"{x:.1f}%"),
                textposition="outside",
                textfont_color=COLORS["text"],
            ))
            fig_acc.update_layout(**plotly_cfg(), height=280,
                                   xaxis_tickangle=-20,
                                   yaxis_range=[0, 105])
            st.plotly_chart(fig_acc, use_container_width=True)
        except:
            st.info("Train models to see comparison.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Dataset summary strip ──────────────────────────────────────
    st.markdown(f"<div class='section-label'>Dataset Overview</div>", unsafe_allow_html=True)
    s = dataset_summary(df)
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Samples",       s["total_samples"])
    d2.metric("Features",      s["num_features"])
    d3.metric("Crop Classes",  s["num_classes"])
    d4.metric("Missing Values",s["missing_values"])
    d5.metric("Samples/Class", int(s["total_samples"]/s["num_classes"]))


# ══════════════════════════════════════════════════════════════════
# PAGE: DATASET
# ══════════════════════════════════════════════════════════════════
elif nav == "📂  Dataset":

    st.markdown(f"""
    <div class='page-header'>
      <div class='page-title'>📂 Dataset Management</div>
      <div class='page-sub'>Upload, validate, and manage training datasets</div>
    </div>
    """, unsafe_allow_html=True)

    df = load_active()

    # ── Active dataset card ────────────────────────────────────────
    st.markdown(f"<div class='section-label'>Active Dataset</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='chart-card'>
      <div style='display:flex; gap:40px; align-items:center; flex-wrap:wrap;'>
        <div>
          <div style='color:{COLORS["muted"]};font-size:0.7rem;font-weight:700;
                      text-transform:uppercase;letter-spacing:0.08em;'>File</div>
          <div style='color:{COLORS["text"]};font-size:1.1rem;font-weight:700;margin-top:4px;'>
            {os.path.basename(active_dataset())}
          </div>
        </div>
        <div>
          <div style='color:{COLORS["muted"]};font-size:0.7rem;font-weight:700;
                      text-transform:uppercase;letter-spacing:0.08em;'>Rows</div>
          <div style='color:{COLORS["accent"]};font-size:1.1rem;font-weight:700;margin-top:4px;'>{len(df)}</div>
        </div>
        <div>
          <div style='color:{COLORS["muted"]};font-size:0.7rem;font-weight:700;
                      text-transform:uppercase;letter-spacing:0.08em;'>Columns</div>
          <div style='color:{COLORS["accent"]};font-size:1.1rem;font-weight:700;margin-top:4px;'>{len(df.columns)}</div>
        </div>
        <div>
          <div style='color:{COLORS["muted"]};font-size:0.7rem;font-weight:700;
                      text-transform:uppercase;letter-spacing:0.08em;'>Classes</div>
          <div style='color:{COLORS["accent"]};font-size:1.1rem;font-weight:700;margin-top:4px;'>{df["label"].nunique()}</div>
        </div>
        <div>
          <div style='color:{COLORS["muted"]};font-size:0.7rem;font-weight:700;
                      text-transform:uppercase;letter-spacing:0.08em;'>Missing</div>
          <div style='color:{COLORS["green"]};font-size:1.1rem;font-weight:700;margin-top:4px;'>0</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("👁️  Preview Dataset (first 20 rows)"):
        st.dataframe(df.head(20), use_container_width=True)

    # ── Upload new dataset ─────────────────────────────────────────
    st.markdown(f"<div class='section-label'>Upload New Dataset</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='chart-card'>
      <div style='color:{COLORS["muted"]};font-size:0.85rem;margin-bottom:14px;'>
        <strong style='color:{COLORS["accent"]};'>Required columns:</strong>
        n, p, k, temperature, humidity, ph, rainfall, label &nbsp;|&nbsp;
        <strong style='color:{COLORS["muted"]};'>Format:</strong> CSV, max 200 MB
      </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="ds_upload",
                                 label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded:
        df_new, err = load_uploaded_data(uploaded)
        if err:
            st.error(f"❌ {err}")
        else:
            st.markdown(f"<div class='section-label'>Validation Report</div>", unsafe_allow_html=True)

            checks = [
                ("Required columns present",  True),
                ("No missing values",         df_new.isnull().sum().sum() == 0),
                ("Label column present",      "label" in df_new.columns),
                ("Sufficient rows (≥ 50)",    len(df_new) >= 50),
                ("All features are numeric",  all(pd.api.types.is_numeric_dtype(df_new[f])
                                                  for f in FEATURE_NAMES if f in df_new.columns)),
            ]
            all_ok = all(v for _, v in checks)

            for label, passed in checks:
                icon  = "✅" if passed else "❌"
                badge = "badge-green" if passed else "badge-red"
                txt   = "PASS" if passed else "FAIL"
                st.markdown(
                    f"<div class='val-check'><span>{icon} {label}</span>"
                    f"<span class='{badge}'>{txt}</span></div>",
                    unsafe_allow_html=True,
                )

            if all_ok:
                st.success(f"✅ Dataset valid — {len(df_new)} rows, {df_new['label'].nunique()} classes.")
                with st.expander("Preview uploaded data"):
                    st.dataframe(df_new.head(15), use_container_width=True)
                if st.button("💾  Save as Active Dataset", use_container_width=True):
                    df_new.to_csv(CUSTOM_DATASET, index=False)
                    st.cache_data.clear()
                    st.success("Saved! Restart the app to use the new dataset.")
            else:
                st.error("Dataset failed validation. Fix the issues above and re-upload.")

    # ── Delete custom dataset ──────────────────────────────────────
    if os.path.exists(CUSTOM_DATASET):
        st.markdown(f"<div class='section-label'>Danger Zone</div>", unsafe_allow_html=True)
        st.warning("Deleting the custom dataset will revert to the default dataset.")
        if st.button("🗑️  Delete Custom Dataset", use_container_width=True):
            os.remove(CUSTOM_DATASET)
            st.cache_data.clear()
            st.success("Custom dataset deleted.")
            st.rerun()


# ══════════════════════════════════════════════════════════════════
# PAGE: MODELS
# ══════════════════════════════════════════════════════════════════
elif nav == "🤖  Models":

    st.markdown(f"""
    <div class='page-header'>
      <div class='page-title'>🤖 Model Management</div>
      <div class='page-sub'>Train, evaluate, and compare machine learning models</div>
    </div>
    """, unsafe_allow_html=True)

    df = load_active()

    # ── Train single model ─────────────────────────────────────────
    st.markdown(f"<div class='section-label'>Train a Model</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='chart-card'>", unsafe_allow_html=True)

    c1, c2 = st.columns([3, 1])
    with c1:
        model_sel = st.selectbox("Select Model", list(MODELS.keys()), key="m_sel",
                                  label_visibility="visible")
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        train_btn = st.button("🚀  Train Model", use_container_width=True)

    if train_btn:
        with st.spinner(f"Training {model_sel}…"):
            model = clone(MODELS[model_sel])
            X_train, X_test, y_train, y_test = get_train_test(df)
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test)
            cv_acc  = cross_validate_model(model, df[FEATURE_NAMES], df["label"])
            fi      = get_feature_importance(model)

        st.session_state.trained   = model
        st.session_state.t_name    = model_sel
        st.session_state.t_metrics = metrics
        st.session_state.t_fi      = fi
        st.success(f"✅ {model_sel} trained successfully!")

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Show metrics if trained ────────────────────────────────────
    if st.session_state.t_metrics:
        m = st.session_state.t_metrics
        st.markdown(f"<div class='section-label'>Performance — {st.session_state.t_name}</div>", unsafe_allow_html=True)

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Accuracy",  f"{m['Accuracy']:.2f}%")
        r2.metric("Precision", f"{m['Precision']:.2f}%")
        r3.metric("Recall",    f"{m['Recall']:.2f}%")
        r4.metric("F1 Score",  f"{m['F1 Score']:.2f}%")

        # Feature importance
        fi = st.session_state.t_fi
        if fi is not None and not fi.empty:
            st.markdown(f"<div class='section-label'>Feature Importance</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='chart-card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='chart-title'>🔑 Which inputs influence prediction most?</div>", unsafe_allow_html=True)

            fi_df = fi.reset_index()
            fi_df.columns = ["Feature", "Importance"]
            fi_df["Feature"] = fi_df["Feature"].str.upper().replace(
                {"TEMPERATURE":"Temp","HUMIDITY":"Humidity","RAINFALL":"Rainfall","PH":"pH"})

            fig_fi = go.Figure(go.Bar(
                x=fi_df["Importance"], y=fi_df["Feature"],
                orientation="h",
                marker=dict(
                    color=fi_df["Importance"],
                    colorscale=[[0,"#1E3A5F"],[1,"#38BDF8"]],
                ),
                text=fi_df["Importance"].apply(lambda x: f"{x:.3f}"),
                textposition="outside",
                textfont_color=COLORS["text"],
            ))
            fig_fi.update_layout(**plotly_cfg(), height=280,
                                  xaxis_range=[0, fi_df["Importance"].max()*1.2])
            fig_fi.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(fig_fi, use_container_width=True)
            st.markdown(f"""
            <div style='color:{COLORS["muted"]};font-size:0.82rem;margin-top:8px;'>
              💡 Higher values = stronger influence on crop prediction.
              Rainfall and humidity typically dominate in this dataset.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        if st.button("💾  Save Model to Disk", use_container_width=True):
            path = save_model(st.session_state.trained, st.session_state.t_name)
            st.success(f"Saved to `{path}`")

    # ── Compare all models ─────────────────────────────────────────
    st.markdown(f"<div class='section-label'>Compare All Models</div>", unsafe_allow_html=True)
    if st.button("🔄  Train & Compare All 6 Models", use_container_width=True):
        with st.spinner("Training all 6 models…"):
            res_df, t_models, best_n, best_m, X_test, y_test = train_and_compare_all(df)

        st.success(f"🏆 Best: **{best_n}** — Accuracy: **{res_df.loc[best_n,'Accuracy']:.2f}%**")

        # Styled table
        st.markdown(f"<div class='chart-card'>", unsafe_allow_html=True)
        styled = res_df.style.background_gradient(cmap="Blues", axis=0).format("{:.2f}")
        st.dataframe(styled, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Grouped bar chart
        st.markdown(f"<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='chart-title'>📊 All Metrics Comparison</div>", unsafe_allow_html=True)
        melted = res_df[["Accuracy","Precision","Recall","F1 Score"]].reset_index().melt(
            id_vars="Model", var_name="Metric", value_name="Score (%)")
        fig_cmp = px.bar(
            melted, x="Model", y="Score (%)", color="Metric", barmode="group",
            color_discrete_sequence=["#38BDF8","#818CF8","#FBBF24","#F87171"],
            template="plotly_dark",
        )
        fig_cmp.update_layout(**plotly_cfg(), xaxis_tickangle=-20, height=320,
                               legend=dict(bgcolor=COLORS["chart_bg"], font_size=11))
        st.plotly_chart(fig_cmp, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # DT vs RF highlight
        st.markdown(f"<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='chart-title'>🌲 Decision Tree vs Random Forest</div>", unsafe_allow_html=True)
        dt_rf = res_df.loc[res_df.index.isin(["Decision Tree","Random Forest"]),
                           ["Accuracy","Precision","Recall","F1 Score"]]
        if len(dt_rf) >= 1:
            fig_dtrf = go.Figure()
            metrics_list = ["Accuracy","Precision","Recall","F1 Score"]
            colors_dtrf  = ["#38BDF8","#FBBF24"]
            for idx, (model_n, row) in enumerate(dt_rf.iterrows()):
                fig_dtrf.add_trace(go.Bar(
                    name=model_n, x=metrics_list,
                    y=[row[m] for m in metrics_list],
                    marker_color=colors_dtrf[idx % 2],
                    text=[f"{row[m]:.1f}%" for m in metrics_list],
                    textposition="outside",
                    textfont_color=COLORS["text"],
                ))
            fig_dtrf.update_layout(**plotly_cfg(), barmode="group", height=300,
                                    yaxis_range=[0,105],
                                    legend=dict(bgcolor=COLORS["chart_bg"]))
            st.plotly_chart(fig_dtrf, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("📄  Full Classification Report — Best Model"):
            st.code(get_classification_report(best_m, X_test, y_test), language="text")


# ══════════════════════════════════════════════════════════════════
# PAGE: LOGS
# ══════════════════════════════════════════════════════════════════
elif nav == "📋  Logs":

    st.markdown(f"""
    <div class='page-header'>
      <div class='page-title'>📋 Prediction Logs</div>
      <div class='page-sub'>All user predictions with inputs, outputs, and timestamps</div>
    </div>
    """, unsafe_allow_html=True)

    logs = load_logs()

    if not logs:
        st.markdown(f"""
        <div style='text-align:center; padding:60px; background:{COLORS["card"]};
                    border-radius:16px; border:1px dashed {COLORS["border"]}; color:{COLORS["muted"]};'>
          <div style='font-size:3rem; margin-bottom:12px;'>📋</div>
          <p>No prediction logs yet.<br>Logs are recorded when users make predictions in the main app.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        logs_df = pd.DataFrame(logs)

        # Summary metrics
        l1, l2, l3, l4 = st.columns(4)
        l1.metric("Total Logs",    len(logs_df))
        l2.metric("Unique Crops",  logs_df["Top Crop"].nunique() if "Top Crop" in logs_df else "—")
        l3.metric("Models Used",   logs_df["Model"].nunique() if "Model" in logs_df else "—")
        l4.metric("Latest",        logs_df["Timestamp"].iloc[-1][:10] if "Timestamp" in logs_df else "—")

        st.markdown("<br>", unsafe_allow_html=True)

        # Filter bar
        st.markdown(f"<div class='chart-card'>", unsafe_allow_html=True)
        f1, f2, f3 = st.columns([2, 2, 1])
        with f1:
            crops = ["All"] + sorted(logs_df["Top Crop"].unique().tolist()) if "Top Crop" in logs_df else ["All"]
            sel_crop = st.selectbox("Filter by Crop", crops, key="lf_crop")
        with f2:
            models_l = ["All"] + sorted(logs_df["Model"].unique().tolist()) if "Model" in logs_df else ["All"]
            sel_model = st.selectbox("Filter by Model", models_l, key="lf_model")
        with f3:
            st.markdown("<br>", unsafe_allow_html=True)
            show_n = st.number_input("Show last N", min_value=5, max_value=500, value=50, step=5)

        filtered = logs_df.copy()
        if "Top Crop" in logs_df.columns and sel_crop != "All":
            filtered = filtered[filtered["Top Crop"] == sel_crop]
        if "Model" in logs_df.columns and sel_model != "All":
            filtered = filtered[filtered["Model"] == sel_model]
        filtered = filtered.tail(int(show_n))

        st.markdown(f"<div style='color:{COLORS['muted']};font-size:0.8rem;margin:8px 0;'>Showing {len(filtered)} records</div>", unsafe_allow_html=True)
        st.dataframe(filtered, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Actions
        c1, c2, _ = st.columns([1, 1, 3])
        with c1:
            csv = filtered.to_csv(index=False).encode()
            st.download_button("⬇️  Export CSV", data=csv,
                               file_name=f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                               mime="text/csv", use_container_width=True)
        with c2:
            if st.button("🗑️  Clear All Logs", use_container_width=True):
                clear_logs(); st.success("Logs cleared."); st.rerun()

    # ── Registered Users ──────────────────────────────────────────
    st.markdown(f"<div class='section-label'>Registered Users</div>", unsafe_allow_html=True)
    try:
        import sqlite3 as _sq
        _db = "users.db"
        if not os.path.exists(_db):
            _db = os.path.join("database", "users.db")
        if os.path.exists(_db):
            _conn = _sq.connect(_db, check_same_thread=False)
            try:
                users_df = pd.read_sql_query(
                    "SELECT id, username, full_name, phone, email FROM users ORDER BY id DESC",
                    _conn
                )
            except Exception:
                users_df = pd.read_sql_query(
                    "SELECT id, username FROM users ORDER BY id DESC", _conn
                )
            _conn.close()
            st.markdown(f"<div class='chart-card'>", unsafe_allow_html=True)
            st.markdown(f"<div style='color:{COLORS['muted']};font-size:0.8rem;margin-bottom:10px;'>{len(users_df)} registered user(s)</div>", unsafe_allow_html=True)
            users_df.columns = [c.replace("_"," ").title() for c in users_df.columns]
            st.dataframe(users_df, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No user database found.")
    except Exception as e:
        st.warning(f"Could not load users: {e}")


# ══════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ══════════════════════════════════════════════════════════════════
elif nav == "📈  Analytics":

    st.markdown(f"""
    <div class='page-header'>
      <div class='page-title'>📈 Analytics</div>
      <div class='page-sub'>Visual insights into dataset patterns and prediction trends</div>
    </div>
    """, unsafe_allow_html=True)

    df   = load_active()
    logs = load_logs()

    # ── Prediction analytics ───────────────────────────────────────
    if logs:
        logs_df = pd.DataFrame(logs)
        st.markdown(f"<div class='section-label'>Prediction Analytics</div>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2, gap="large")
        with col_a:
            st.markdown(f"<div class='chart-card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='chart-title'>📊 Top Predicted Crops</div>", unsafe_allow_html=True)
            if "Top Crop" in logs_df.columns:
                freq = logs_df["Top Crop"].value_counts().head(8).reset_index()
                freq.columns = ["Crop","Count"]
                fig = px.bar(freq, x="Count", y="Crop", orientation="h",
                             color="Count",
                             color_continuous_scale=[[0,"#1E3A5F"],[1,"#38BDF8"]],
                             template="plotly_dark", text="Count")
                fig.update_traces(textposition="outside", textfont_color=COLORS["text"])
                fig.update_layout(**plotly_cfg(), coloraxis_showscale=False, height=300)
                fig.update_yaxes(categoryorder="total ascending")
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_b:
            st.markdown(f"<div class='chart-card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='chart-title'>🤖 Model Usage Distribution</div>", unsafe_allow_html=True)
            if "Model" in logs_df.columns:
                mu = logs_df["Model"].value_counts().reset_index()
                mu.columns = ["Model","Count"]
                fig_mu = px.pie(mu, names="Model", values="Count", hole=0.45,
                                color_discrete_sequence=["#38BDF8","#0EA5E9","#0284C7","#0369A1","#075985","#0C4A6E"],
                                template="plotly_dark")
                fig_mu.update_layout(paper_bgcolor=COLORS["chart_bg"],
                                      font=dict(color=COLORS["text"],family="Inter"),
                                      margin=dict(t=10,b=10,l=10,r=10),
                                      legend=dict(bgcolor=COLORS["chart_bg"],font_size=11),
                                      height=300)
                st.plotly_chart(fig_mu, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # ── Dataset analytics ──────────────────────────────────────────
    st.markdown(f"<div class='section-label'>Dataset Analytics</div>", unsafe_allow_html=True)

    col_c, col_d = st.columns(2, gap="large")
    with col_c:
        st.markdown(f"<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='chart-title'>🌧️ Rainfall vs Crop</div>", unsafe_allow_html=True)
        fig_r = px.box(df, x="label", y="rainfall", color="label",
                       template="plotly_dark",
                       color_discrete_sequence=px.colors.qualitative.Pastel,
                       labels={"label":"Crop","rainfall":"Rainfall (mm)"})
        fig_r.update_layout(**plotly_cfg(), xaxis_tickangle=-45, showlegend=False, height=300)
        st.plotly_chart(fig_r, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_d:
        st.markdown(f"<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='chart-title'>🌡️ Temperature vs Crop</div>", unsafe_allow_html=True)
        fig_t = px.box(df, x="label", y="temperature", color="label",
                       template="plotly_dark",
                       color_discrete_sequence=px.colors.qualitative.Set2,
                       labels={"label":"Crop","temperature":"Temperature (°C)"})
        fig_t.update_layout(**plotly_cfg(), xaxis_tickangle=-45, showlegend=False, height=300)
        st.plotly_chart(fig_t, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Correlation heatmap
    st.markdown(f"<div class='chart-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='chart-title'>🔗 Feature Correlation Heatmap</div>", unsafe_allow_html=True)
    fig_corr, ax = plt.subplots(figsize=(9, 4))
    fig_corr.patch.set_facecolor(COLORS["chart_bg"])
    ax.set_facecolor(COLORS["card2"])
    sns.heatmap(df[FEATURE_NAMES].corr(), annot=True, fmt=".2f",
                cmap="Blues", linewidths=0.5, linecolor=COLORS["border"], ax=ax)
    ax.tick_params(colors=COLORS["muted"])
    plt.tight_layout()
    st.pyplot(fig_corr)
    plt.close(fig_corr)
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: TESTING
# ══════════════════════════════════════════════════════════════════
elif nav == "🧪  Testing":

    st.markdown(f"""
    <div class='page-header'>
      <div class='page-title'>🧪 Testing Panel</div>
      <div class='page-sub'>Evaluate model performance on test data</div>
    </div>
    """, unsafe_allow_html=True)

    df_train = load_active()

    st.markdown(f"<div class='chart-card'>", unsafe_allow_html=True)
    c1, c2 = st.columns([3, 1])
    with c1:
        test_model = st.selectbox("Select Model", list(MODELS.keys()), key="t_model")
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        use_default = st.checkbox("Use 20% holdout", value=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if use_default:
        if st.button("▶️  Run Evaluation", use_container_width=True):
            with st.spinner(f"Training {test_model} and evaluating…"):
                model = clone(MODELS[test_model])
                X_train, X_test, y_train, y_test = get_train_test(df_train)
                model.fit(X_train, y_train)
                metrics = evaluate_model(model, X_test, y_test)
                cv_acc  = cross_validate_model(model, df_train[FEATURE_NAMES], df_train["label"])

            st.success(f"✅ {test_model} evaluated on {len(y_test)} test samples (20% holdout)")

            r1, r2, r3, r4, r5 = st.columns(5)
            r1.metric("Accuracy",    f"{metrics['Accuracy']:.2f}%")
            r2.metric("Precision",   f"{metrics['Precision']:.2f}%")
            r3.metric("Recall",      f"{metrics['Recall']:.2f}%")
            r4.metric("F1 Score",    f"{metrics['F1 Score']:.2f}%")
            r5.metric("CV Accuracy", f"{cv_acc:.2f}%")

            # Results table
            y_pred = model.predict(X_test)
            probs  = model.predict_proba(X_test)
            confs  = [round(probs[i].max()*100, 2) for i in range(len(probs))]
            res_tbl = X_test.copy()
            res_tbl["Actual"]     = y_test.values
            res_tbl["Predicted"]  = y_pred
            res_tbl["Confidence"] = confs
            res_tbl["Correct"]    = res_tbl["Actual"] == res_tbl["Predicted"]
            res_tbl = res_tbl.reset_index(drop=True)

            st.markdown(f"<div class='section-label'>Test Results (first 50 rows)</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='chart-card'>", unsafe_allow_html=True)
            st.dataframe(res_tbl.head(50), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            csv_t = res_tbl.to_csv(index=False).encode()
            st.download_button("⬇️  Download Full Results", data=csv_t,
                               file_name=f"test_{test_model.replace(' ','_').lower()}.csv",
                               mime="text/csv", use_container_width=True)

            with st.expander("📄  Full Classification Report"):
                st.code(get_classification_report(model, X_test, y_test), language="text")

    else:
        st.markdown(f"<div class='section-label'>Upload Custom Test CSV</div>", unsafe_allow_html=True)
        test_file = st.file_uploader("Upload test CSV (must have label column)",
                                      type=["csv"], key="t_upload")
        if test_file:
            df_test, err = load_uploaded_data(test_file)
            if err:
                st.error(f"❌ {err}")
            elif "label" not in df_test.columns:
                st.error("❌ Test CSV must have a 'label' column.")
            else:
                with st.spinner("Training and evaluating…"):
                    model = clone(MODELS[test_model])
                    X_all, y_all = get_feature_target(df_train)
                    model.fit(X_all, y_all)
                    X_t, y_t = get_feature_target(df_test)
                    metrics = evaluate_model(model, X_t, y_t)

                st.success(f"✅ Evaluated on {len(df_test)} uploaded samples.")
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Accuracy",  f"{metrics['Accuracy']:.2f}%")
                r2.metric("Precision", f"{metrics['Precision']:.2f}%")
                r3.metric("Recall",    f"{metrics['Recall']:.2f}%")
                r4.metric("F1 Score",  f"{metrics['F1 Score']:.2f}%")

                y_pred = model.predict(X_t)
                out_df = df_test.copy()
                out_df["Predicted"] = y_pred
                out_df["Correct"]   = out_df["label"] == out_df["Predicted"]
                st.dataframe(out_df.head(30), use_container_width=True)

                with st.expander("📄  Classification Report"):
                    st.code(get_classification_report(model, X_t, y_t), language="text")
