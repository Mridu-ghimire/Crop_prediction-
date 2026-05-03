"""
app.py  -  Crop Prediction System
Mridu Ghimire | 77466817 | BSc (Hons) Computing Level 6
"""
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import hashlib
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
import json
import os
warnings.filterwarnings("ignore")

LOGS_FILE = "prediction_logs.json"

def _append_log(entry: dict):
    logs = []
    if os.path.exists(LOGS_FILE):
        try:
            with open(LOGS_FILE, "r") as f:
                logs = json.load(f)
        except Exception:
            logs = []
    logs.append(entry)
    with open(LOGS_FILE, "w") as f:
        json.dump(logs, f, indent=2)

from data import (
    load_data, load_uploaded_data, validate_inputs,
    get_feature_target, dataset_summary, get_crop_info,
    get_fertilizer_recommendations, explain_prediction,
    FEATURE_NAMES, FEATURE_RANGES, FEATURE_DEFAULTS,
)
from model import (
    MODELS, get_train_test, train_single, evaluate_model,
    train_and_compare_all, predict_top3, batch_predict,
    get_feature_importance, get_confusion_matrix,
    get_classification_report, get_error_analysis, error_summary,
    save_model,
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Crop Prediction System",
    page_icon="🌾",
    layout="wide",
)

# ─────────────────────────────────────────────
# PREMIUM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  * { font-family: 'Inter', sans-serif !important; }

  /* ── Base ── */
  .stApp { background: #0a0f0a !important; color: #e2e8e2 !important; }
  .main .block-container { padding: 1.5rem 2rem 2rem 2rem !important; max-width: 1400px; }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background: #0d1a0d !important;
    border-right: 1px solid #1a3a1a !important;
    width: 260px !important;
  }
  section[data-testid="stSidebar"] * { color: #b8d4b8 !important; }
  section[data-testid="stSidebar"] .stRadio label {
    padding: 8px 12px !important;
    border-radius: 8px !important;
    margin: 2px 0 !important;
    transition: all 0.2s !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
  }
  section[data-testid="stSidebar"] .stRadio label:hover {
    background: #1a3a1a !important;
    color: #4ade80 !important;
  }

  /* ── Metric cards ── */
  div[data-testid="metric-container"] {
    background: #0d1a0d !important;
    border: 1px solid #1a3a1a !important;
    border-radius: 14px !important;
    padding: 18px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4) !important;
  }
  div[data-testid="metric-container"] label {
    color: #6b9e6b !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
  }
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #4ade80 !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
  }

  /* ── Buttons — scoped to avoid leaking into file uploader & expander ── */
  .stButton > button {
    background: linear-gradient(135deg, #16a34a, #15803d) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 28px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.25s !important;
    box-shadow: 0 4px 15px rgba(22,163,74,0.3) !important;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #22c55e, #16a34a) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(34,197,94,0.4) !important;
  }

  /* ── File Uploader — definitive fix ── */
  [data-testid="stFileUploaderDropzone"] {
    background: #0d1a0d !important;
    border: 1px dashed #1a3a1a !important;
    border-radius: 10px !important;
    padding: 20px !important;
  }
  /* Hide the duplicate label span that causes "uploadUpload" */
  [data-testid="stFileUploaderDropzone"] button span[data-testid="stMarkdownContainer"],
  [data-testid="stFileUploaderDropzone"] button > div > span:last-child,
  [data-testid="stFileUploaderDropzone"] button > span + span {
    display: none !important;
  }
  [data-testid="stFileUploaderDropzone"] button {
    background: #0d1a0d !important;
    border: 1px solid #1a3a1a !important;
    border-radius: 8px !important;
    color: #4ade80 !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 7px 18px !important;
    cursor: pointer !important;
    font-family: 'Inter', sans-serif !important;
    box-shadow: none !important;
    transform: none !important;
    transition: background 0.2s !important;
    /* Prevent text overlap */
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    position: relative !important;
    overflow: hidden !important;
    white-space: nowrap !important;
  }
  [data-testid="stFileUploaderDropzone"] button:hover {
    background: #1a3a1a !important;
    border-color: #4ade80 !important;
    transform: none !important;
    box-shadow: none !important;
  }
  [data-testid="stFileUploaderDropzone"] small,
  [data-testid="stFileUploaderDropzone"] span {
    color: #6b9e6b !important;
    font-size: 0.82rem !important;
  }

  /* ── Expander — definitive fix ── */
  [data-testid="stExpander"] {
    border: 1px solid #1a3a1a !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    background: #0d1a0d !important;
  }
  /* The expander toggle button */
  [data-testid="stExpander"] > div > div > div > button,
  [data-testid="stExpander"] summary {
    background: #0d1a0d !important;
    border: none !important;
    box-shadow: none !important;
    transform: none !important;
    padding: 12px 16px !important;
    width: 100% !important;
    text-align: left !important;
    color: #4ade80 !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    font-family: 'Inter', sans-serif !important;
    cursor: pointer !important;
    /* Key fix: use flex to prevent text overlap */
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
  }
  [data-testid="stExpander"] > div > div > div > button:hover,
  [data-testid="stExpander"] summary:hover {
    background: #1a3a1a !important;
    transform: none !important;
    box-shadow: none !important;
  }
  /* Hide duplicate text spans inside expander button */
  [data-testid="stExpander"] > div > div > div > button > span:nth-child(2),
  [data-testid="stExpander"] > div > div > div > button > div:nth-child(2) {
    display: none !important;
  }
  [data-testid="stExpander"] > div > div > div > button svg {
    color: #4ade80 !important;
    flex-shrink: 0 !important;
    min-width: 16px !important;
  }

  /* ── Sliders ── */
  .stSlider > div > div > div > div {
    background: #4ade80 !important;
  }
  .stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #4ade80 !important;
    border: 2px solid #16a34a !important;
    box-shadow: 0 0 8px rgba(74,222,128,0.5) !important;
  }

  /* ── Selectbox ── */
  .stSelectbox > div > div {
    background: #0d1a0d !important;
    border: 1px solid #1a3a1a !important;
    border-radius: 10px !important;
    color: #e2e8e2 !important;
  }

  /* ── Text inputs ── */
  .stTextInput > div > div > input {
    background: #0d1a0d !important;
    border: 1px solid #1a3a1a !important;
    border-radius: 10px !important;
    color: #e2e8e2 !important;
    padding: 10px 14px !important;
  }
  .stTextInput > div > div > input:focus {
    border-color: #4ade80 !important;
    box-shadow: 0 0 0 2px rgba(74,222,128,0.2) !important;
  }

  /* ── Headings ── */
  h1 { color: #4ade80 !important; font-weight: 700 !important; font-size: 1.8rem !important; }
  h2 { color: #4ade80 !important; font-weight: 600 !important; }
  h3 { color: #86efac !important; font-weight: 600 !important; }

  /* ── Divider ── */
  hr { border-color: #1a3a1a !important; margin: 1rem 0 !important; }

  /* ── DataFrames ── */
  .stDataFrame { border-radius: 12px !important; overflow: hidden !important; }
  .stDataFrame thead th {
    background: #0d1a0d !important;
    color: #4ade80 !important;
    font-weight: 600 !important;
  }

  /* ── Alerts ── */
  .stAlert { border-radius: 10px !important; }
  div[data-testid="stSuccessMessage"] { background: #052e16 !important; border-color: #16a34a !important; }
  div[data-testid="stInfoMessage"]    { background: #0c1a2e !important; border-color: #1d4ed8 !important; }
  div[data-testid="stErrorMessage"]   { background: #2a0a0a !important; border-color: #dc2626 !important; }

  /* ── Custom cards ── */
  .crop-card {
    background: linear-gradient(135deg, #052e16, #064e3b);
    border: 1px solid #16a34a;
    border-radius: 16px;
    padding: 22px;
    margin: 10px 0;
    box-shadow: 0 4px 20px rgba(22,163,74,0.15);
    transition: transform 0.2s;
  }
  .crop-card:hover { transform: translateY(-2px); }
  .crop-card h3 { color: #4ade80 !important; font-size: 1.3rem !important; margin: 0 0 8px 0; }
  .crop-card p  { color: #bbf7d0; margin: 4px 0; font-size: 0.9rem; }

  .conf-badge {
    display: inline-block;
    background: linear-gradient(135deg, #16a34a, #15803d);
    color: #fff;
    border-radius: 20px;
    padding: 5px 16px;
    font-weight: 700;
    font-size: 0.9rem;
    box-shadow: 0 2px 8px rgba(22,163,74,0.3);
  }

  .explain-box {
    background: #052e16;
    border-left: 4px solid #4ade80;
    border-radius: 0 10px 10px 0;
    padding: 16px 20px;
    margin: 12px 0;
    color: #bbf7d0;
    font-size: 0.92rem;
    line-height: 1.6;
  }

  .fert-box {
    background: #0d1a0d;
    border: 1px solid #1a3a1a;
    border-radius: 10px;
    padding: 12px 16px;
    margin: 6px 0;
    color: #d1fae5;
    font-size: 0.87rem;
    line-height: 1.5;
  }

  .section-hdr {
    background: linear-gradient(90deg, #052e16, #0a0f0a);
    color: #4ade80;
    padding: 10px 18px;
    border-radius: 10px;
    font-weight: 700;
    font-size: 1rem;
    margin: 20px 0 14px;
    border-left: 4px solid #4ade80;
    letter-spacing: 0.02em;
  }

  .stat-card {
    background: #0d1a0d;
    border: 1px solid #1a3a1a;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
  }
  .stat-card .val { color: #4ade80; font-size: 2rem; font-weight: 700; }
  .stat-card .lbl { color: #6b9e6b; font-size: 0.75rem; font-weight: 600;
                    text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px; }

  .info-row {
    background: #0d1a0d;
    border: 1px solid #1a3a1a;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 6px 0;
    display: flex;
    align-items: center;
    gap: 10px;
    color: #d1fae5;
    font-size: 0.9rem;
  }

  /* ── Login page ── */
  .login-wrap {
    max-width: 420px;
    margin: 60px auto;
    background: #0d1a0d;
    border: 1px solid #1a3a1a;
    border-radius: 20px;
    padding: 40px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.5);
  }

  /* ── Expander ── */
  .streamlit-expanderHeader {
    background: #0d1a0d !important;
    border-radius: 10px !important;
    color: #4ade80 !important;
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    background: #0d1a0d !important;
    border-radius: 12px !important;
    padding: 6px !important;
    gap: 4px !important;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: #6b9e6b !important;
    font-weight: 600 !important;
  }
  .stTabs [aria-selected="true"] {
    background: #16a34a !important;
    color: #fff !important;
  }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: #0a0f0a; }
  ::-webkit-scrollbar-thumb { background: #1a3a1a; border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: #16a34a; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────
DB_PATH = "users.db"

def _get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def _init_db():
    c = _get_conn()
    c.execute("""CREATE TABLE IF NOT EXISTS users(
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        username  TEXT UNIQUE NOT NULL,
        password  TEXT NOT NULL,
        full_name TEXT DEFAULT '',
        phone     TEXT DEFAULT '',
        email     TEXT DEFAULT '')""")
    # Add new columns if upgrading from old schema
    try:
        c.execute("ALTER TABLE users ADD COLUMN full_name TEXT DEFAULT ''")
    except Exception:
        pass
    try:
        c.execute("ALTER TABLE users ADD COLUMN phone TEXT DEFAULT ''")
    except Exception:
        pass
    try:
        c.execute("ALTER TABLE users ADD COLUMN email TEXT DEFAULT ''")
    except Exception:
        pass
    c.commit(); c.close()

_init_db()

def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def add_user(u, p, full_name="", phone="", email=""):
    try:
        c = _get_conn()
        c.execute(
            "INSERT INTO users(username,password,full_name,phone,email) VALUES(?,?,?,?,?)",
            (u.strip(), hash_pw(p), full_name.strip(), phone.strip(), email.strip()),
        )
        c.commit(); c.close(); return True
    except sqlite3.IntegrityError:
        return False

def login_user(u, p):
    c = _get_conn()
    row = c.execute("SELECT 1 FROM users WHERE username=? AND password=?",
                    (u.strip(), hash_pw(p))).fetchone()
    c.close(); return row is not None

def get_user_profile(u):
    c = _get_conn()
    row = c.execute(
        "SELECT full_name, phone, email FROM users WHERE username=?", (u.strip(),)
    ).fetchone()
    c.close()
    return {"full_name": row[0], "phone": row[1], "email": row[2]} if row else {}

# ─────────────────────────────────────────────
# OTP HELPERS
# ─────────────────────────────────────────────
import random

def generate_otp() -> str:
    """Generate a 6-digit OTP."""
    return str(random.randint(100000, 999999))

def send_otp_sms(phone: str, otp: str) -> bool:
    """
    Simulate sending OTP via SMS.
    In production, replace this with a real SMS API (e.g. Twilio).
    Returns True to simulate successful delivery.
    """
    # Simulate network delay
    import time; time.sleep(0.5)
    return True

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
for k, v in {"logged_in": False, "page": "login", "user": "", "history": [],
              "otp_code": None, "otp_phone": None, "otp_data": None, "otp_verified": False}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# AUTH PAGES
# ─────────────────────────────────────────────
def show_login():
    st.markdown("""
    <div style='text-align:center; padding: 50px 0 30px;'>
      <div style='font-size:4rem; margin-bottom:12px;'>🌾</div>
      <h1 style='color:#4ade80; font-size:2rem; margin:0 0 6px;'>Crop Prediction System</h1>
      <p style='color:#6b9e6b; font-size:0.95rem; margin:0;'>ML-Based Intelligent Crop Advisor</p>
    </div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.1, 1])
    with col:
        st.markdown("<div style='background:#0d1a0d;border:1px solid #1a3a1a;border-radius:16px;padding:32px;'>", unsafe_allow_html=True)
        st.markdown("#### 🔐 Sign In")
        u = st.text_input("Username", key="lu", placeholder="Enter your username")
        p = st.text_input("Password", type="password", key="lp", placeholder="Enter your password")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Login", use_container_width=True):
            if not u or not p:
                st.error("Please fill in all fields.")
            elif login_user(u, p):
                st.session_state.logged_in = True
                st.session_state.user = u
                st.rerun()
            else:
                st.error("Invalid username or password.")
        st.markdown("<div style='text-align:center;margin-top:16px;color:#6b9e6b;font-size:0.85rem;'>Don't have an account?</div>", unsafe_allow_html=True)
        if st.button("Create Account →", use_container_width=True):
            st.session_state.page = "signup"; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

def show_signup():
    st.markdown("""
    <div style='text-align:center; padding: 50px 0 30px;'>
      <div style='font-size:4rem; margin-bottom:12px;'>🌾</div>
      <h1 style='color:#4ade80; font-size:2rem; margin:0 0 6px;'>Create Account</h1>
      <p style='color:#6b9e6b; font-size:0.95rem; margin:0;'>Join the Crop Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown("<div style='background:#0d1a0d;border:1px solid #1a3a1a;border-radius:16px;padding:32px;'>", unsafe_allow_html=True)

        # ── STEP 1: Registration form ──────────────────────────────
        if not st.session_state.otp_verified and st.session_state.otp_code is None:
            st.markdown("#### 📝 Step 1 of 2 — Your Details")

            full_name = st.text_input("Full Name *",        key="su_name",  placeholder="Enter your full name")
            phone     = st.text_input("Phone Number *",     key="su_phone", placeholder="e.g. 9800000000")
            email     = st.text_input("Email (optional)",   key="su_email", placeholder="e.g. name@email.com")

            st.markdown("<hr style='border-color:#1a3a1a; margin:12px 0;'>", unsafe_allow_html=True)

            u  = st.text_input("Username *",          key="su",   placeholder="Choose a username")
            p  = st.text_input("Password *",          type="password", key="sp",  placeholder="Min 6 characters")
            p2 = st.text_input("Confirm Password *",  type="password", key="sp2", placeholder="Repeat password")

            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("📱 Send OTP to Phone", use_container_width=True):
                # Validate all fields first
                if not full_name:
                    st.error("Full name is required.")
                elif not phone:
                    st.error("Phone number is required.")
                elif not phone.replace("+","").replace("-","").replace(" ","").isdigit():
                    st.error("Phone number must contain only digits.")
                elif len(phone.replace("+","").replace("-","").replace(" ","")) < 7:
                    st.error("Enter a valid phone number (min 7 digits).")
                elif email and "@" not in email:
                    st.error("Enter a valid email address.")
                elif not u:
                    st.error("Username is required.")
                elif not p or not p2:
                    st.error("Please fill in password fields.")
                elif p != p2:
                    st.error("Passwords do not match.")
                elif len(p) < 6:
                    st.warning("Password must be at least 6 characters.")
                else:
                    # Check username not taken before sending OTP
                    c = _get_conn()
                    exists = c.execute("SELECT 1 FROM users WHERE username=?", (u.strip(),)).fetchone()
                    c.close()
                    if exists:
                        st.error("Username already taken. Choose another.")
                    else:
                        # Generate and "send" OTP
                        otp = generate_otp()
                        with st.spinner("Sending OTP…"):
                            send_otp_sms(phone, otp)
                        # Store OTP and form data in session
                        st.session_state.otp_code  = otp
                        st.session_state.otp_phone = phone
                        st.session_state.otp_data  = {
                            "full_name": full_name, "phone": phone,
                            "email": email, "username": u, "password": p,
                        }
                        st.rerun()

        # ── STEP 2: OTP Verification ───────────────────────────────
        elif st.session_state.otp_code is not None and not st.session_state.otp_verified:
            phone_display = st.session_state.otp_phone

            st.markdown("#### 📱 Step 2 of 2 — Verify Phone Number")

            # Show OTP (simulated SMS delivery)
            st.markdown(f"""
            <div style='background:#052e16; border:1px solid #16a34a; border-radius:12px;
                        padding:16px 20px; margin:12px 0; text-align:center;'>
              <div style='color:#6b9e6b; font-size:0.8rem; margin-bottom:6px;'>
                📱 OTP sent to <strong style='color:#4ade80;'>{phone_display}</strong>
              </div>
              <div style='color:#4ade80; font-size:2rem; font-weight:800; letter-spacing:0.3em;'>
                {st.session_state.otp_code}
              </div>
              <div style='color:#6b9e6b; font-size:0.75rem; margin-top:6px;'>
                (In production this would be sent via SMS)
              </div>
            </div>
            """, unsafe_allow_html=True)

            entered_otp = st.text_input(
                "Enter 6-digit OTP *",
                key="otp_input",
                placeholder="Enter the OTP shown above",
                max_chars=6,
            )

            st.markdown("<br>", unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ Verify & Create Account", use_container_width=True):
                    if not entered_otp:
                        st.error("Please enter the OTP.")
                    elif entered_otp.strip() != st.session_state.otp_code:
                        st.error("❌ Incorrect OTP. Please try again.")
                    else:
                        # OTP correct — create account
                        d = st.session_state.otp_data
                        if add_user(d["username"], d["password"],
                                    d["full_name"], d["phone"], d["email"]):
                            st.success(f"✅ Account created for **{d['full_name']}**! Please log in.")
                            # Clear OTP state
                            st.session_state.otp_code     = None
                            st.session_state.otp_phone    = None
                            st.session_state.otp_data     = None
                            st.session_state.otp_verified = False
                            st.session_state.page = "login"
                            st.rerun()
                        else:
                            st.error("Username already taken.")

            with c2:
                if st.button("🔄 Resend OTP", use_container_width=True):
                    new_otp = generate_otp()
                    with st.spinner("Resending OTP…"):
                        send_otp_sms(st.session_state.otp_phone, new_otp)
                    st.session_state.otp_code = new_otp
                    st.success("New OTP sent!")
                    st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("← Back to Registration", use_container_width=True):
                st.session_state.otp_code  = None
                st.session_state.otp_phone = None
                st.session_state.otp_data  = None
                st.rerun()

        st.markdown("""
        <div style='text-align:center; margin-top:14px; color:#6b9e6b; font-size:0.78rem;'>
          * Required fields
        </div>
        """, unsafe_allow_html=True)

        if st.button("← Back to Login", use_container_width=True, key="back_login_btn"):
            st.session_state.otp_code  = None
            st.session_state.otp_phone = None
            st.session_state.otp_data  = None
            st.session_state.page = "login"
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

if not st.session_state.logged_in:
    if st.session_state.page == "signup":
        show_signup()
    else:
        show_login()
    st.stop()

# ─────────────────────────────────────────────
# CACHED HELPERS
# ─────────────────────────────────────────────
@st.cache_data
def _load():
    return load_data()

@st.cache_data
def _compare():
    df = _load()
    return train_and_compare_all(df)

def fetch_weather(city):
    try:
        r = requests.get(f"https://wttr.in/{city}?format=j1", timeout=5)
        d = r.json()["current_condition"][0]
        return {"temp": float(d["temp_C"]), "hum": float(d["humidity"]), "ok": True}
    except Exception:
        return {"temp": 25.0, "hum": 70.0, "ok": False}

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:20px 0 16px;'>
      <div style='font-size:3rem; margin-bottom:8px;'>🌾</div>
      <div style='color:#4ade80; font-weight:700; font-size:1rem; letter-spacing:0.02em;'>Crop Prediction System</div>
      <div style='color:#6b9e6b; font-size:0.72rem; margin-top:4px; letter-spacing:0.05em;'>ML-Based Intelligence</div>
    </div>
    <hr style='border-color:#1a3a1a; margin:0 0 12px;'>
    """, unsafe_allow_html=True)

    nav = st.radio("Nav", [
        "🌱 Predict Crop",
        "📤 Upload & Predict",
        "📊 Model Performance",
        "📈 Data Visualisation",
        "🔍 Error Analysis",
        "📋 History",
        "ℹ️ About",
    ], label_visibility="collapsed")

    st.markdown("<hr style='border-color:#1a3a1a; margin:12px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='color:#4ade80; font-weight:600; font-size:0.85rem; margin-bottom:8px;'>🌤️ Live Weather</div>", unsafe_allow_html=True)
    city = st.text_input("City", "Kathmandu", key="wc", label_visibility="collapsed")
    wx = fetch_weather(city)
    if wx["ok"]:
        st.success(f"🌡️ {wx['temp']}°C  |  💧 {wx['hum']}%")
    else:
        st.info(f"🌡️ {wx['temp']}°C  |  💧 {wx['hum']}%  *(default)*")

    st.markdown("<hr style='border-color:#1a3a1a; margin:12px 0;'>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:#6b9e6b; font-size:0.82rem; margin-bottom:10px;'>👤 <strong style='color:#b8d4b8;'>{st.session_state.user}</strong></div>", unsafe_allow_html=True)
    if st.button("🚪 Logout", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# ═══════════════════════════════════════════════════════════════════
# PAGE 1 – PREDICT CROP
# ═══════════════════════════════════════════════════════════════════
if nav == "🌱 Predict Crop":
    st.title("🌱 Predict Crop")
    st.markdown("<p style='color:#6b9e6b; margin-top:-10px; margin-bottom:20px;'>Enter soil and climate parameters to get AI-powered crop recommendations.</p>", unsafe_allow_html=True)
    st.markdown("---")

    left, right = st.columns([1, 1.4], gap="large")

    with left:
        st.markdown("<div class='section-hdr'>⚙️ Input Parameters</div>", unsafe_allow_html=True)

        n_val   = st.slider("🌿 Nitrogen (N) – kg/ha",   0,   140,  int(FEATURE_DEFAULTS["n"]))
        p_val   = st.slider("🌿 Phosphorus (P) – kg/ha", 5,   145,  int(FEATURE_DEFAULTS["p"]))
        k_val   = st.slider("🌿 Potassium (K) – kg/ha",  5,   205,  int(FEATURE_DEFAULTS["k"]))

        st.markdown(f"""
        <div class='info-row'>
          🌡️ <span><strong>Temperature</strong> — auto from weather: <strong style='color:#4ade80'>{wx['temp']}°C</strong></span>
        </div>
        <div class='info-row'>
          💧 <span><strong>Humidity</strong> — auto from weather: <strong style='color:#4ade80'>{wx['hum']}%</strong></span>
        </div>
        """, unsafe_allow_html=True)
        temp_val = wx["temp"]
        hum_val  = wx["hum"]

        ph_val   = st.slider("🧪 Soil pH",        3.5, 9.5,  float(FEATURE_DEFAULTS["ph"]),       step=0.1)
        rain_val = st.slider("🌧️ Rainfall – mm",  20,  300,  int(FEATURE_DEFAULTS["rainfall"]))

        model_choice = st.selectbox("🤖 Select ML Model", list(MODELS.keys()))
        predict_btn  = st.button("🔍 Predict Crop", use_container_width=True)

        # ── Live Radar Chart ──
        st.markdown("<div class='section-hdr'>📡 Input Radar (Live)</div>", unsafe_allow_html=True)
        df_bg      = _load()
        feat_maxes = df_bg[FEATURE_NAMES].max()
        feat_means = df_bg[FEATURE_NAMES].mean()
        user_vals  = [n_val, p_val, k_val, temp_val, hum_val, ph_val, rain_val]
        user_norm  = [user_vals[i] / feat_maxes.iloc[i] for i in range(len(FEATURE_NAMES))]
        avg_norm   = [feat_means.iloc[i] / feat_maxes.iloc[i] for i in range(len(FEATURE_NAMES))]
        labels_r   = ["N", "P", "K", "Temp", "Humidity", "pH", "Rainfall"]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=user_norm + [user_norm[0]], theta=labels_r + [labels_r[0]],
            fill="toself", name="Your Input",
            line=dict(color="#4ade80", width=2),
            fillcolor="rgba(74,222,128,0.15)",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=avg_norm + [avg_norm[0]], theta=labels_r + [labels_r[0]],
            fill="toself", name="Dataset Avg",
            line=dict(color="#6b7280", width=1.5, dash="dot"),
            fillcolor="rgba(107,114,128,0.1)",
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="#0d1a0d",
                radialaxis=dict(visible=True, range=[0, 1], color="#6b9e6b",
                                gridcolor="#1a3a1a", linecolor="#1a3a1a"),
                angularaxis=dict(color="#6b9e6b", gridcolor="#1a3a1a"),
            ),
            paper_bgcolor="#0a0f0a", plot_bgcolor="#0a0f0a",
            legend=dict(bgcolor="#0d1a0d", font=dict(color="#b8d4b8", size=11)),
            margin=dict(t=20, b=20, l=20, r=20), height=280,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with right:
        st.markdown("<div class='section-hdr'>🏆 Prediction Results</div>", unsafe_allow_html=True)

        if predict_btn:
            user_inputs = {
                "n": n_val, "p": p_val, "k": k_val,
                "temperature": temp_val, "humidity": hum_val,
                "ph": ph_val, "rainfall": rain_val,
            }
            errors = validate_inputs(user_inputs)
            if errors:
                for e in errors:
                    st.error(e)
                st.stop()

            df = _load()
            with st.spinner("Training model and predicting…"):
                model, X_test, y_test = train_single(model_choice, df)

            features = [n_val, p_val, k_val, temp_val, hum_val, ph_val, rain_val]
            top3    = predict_top3(model, features)
            metrics = evaluate_model(model, X_test, y_test)

            # ── Accuracy metrics ──
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Accuracy",  f"{metrics['Accuracy']:.1f}%")
            a2.metric("Precision", f"{metrics['Precision']:.1f}%")
            a3.metric("Recall",    f"{metrics['Recall']:.1f}%")
            a4.metric("F1 Score",  f"{metrics['F1 Score']:.1f}%")

            st.markdown("---")

            # ── Top-3 crop cards ──
            medals = ["🥇", "🥈", "🥉"]
            for i, (crop, conf) in enumerate(top3):
                st.markdown(
                    f"<div class='crop-card'>"
                    f"<h3>{medals[i]} {crop.title()}</h3>"
                    f"<p><span class='conf-badge'>Confidence: {conf:.1f}%</span></p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # ── Confidence gauge ──
            top_crop, top_conf = top3[0]
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=top_conf,
                title={"text": f"Model Confidence", "font": {"color": "#4ade80", "size": 14}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#6b9e6b",
                             "tickfont": {"color": "#6b9e6b"}},
                    "bar":  {"color": "#16a34a", "thickness": 0.25},
                    "bgcolor": "#0d1a0d",
                    "bordercolor": "#1a3a1a",
                    "steps": [
                        {"range": [0,  40], "color": "#1a0a0a"},
                        {"range": [40, 70], "color": "#1a1a0a"},
                        {"range": [70,100], "color": "#0a1a0a"},
                    ],
                    "threshold": {"line": {"color": "#4ade80", "width": 3}, "value": 75},
                },
                number={"suffix": "%", "font": {"color": "#4ade80", "size": 40}},
            ))
            fig_gauge.update_layout(
                paper_bgcolor="#0a0f0a", height=200,
                margin=dict(t=30, b=10, l=30, r=30),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # ── Explainability ──
            explanation = explain_prediction(
                top_crop, n_val, p_val, k_val, temp_val, hum_val, ph_val, rain_val
            )
            st.markdown(
                f"<div class='explain-box'>💡 <strong>Why {top_crop.title()}?</strong><br><br>{explanation}</div>",
                unsafe_allow_html=True,
            )

            # ── Crop info ──
            info = get_crop_info(top_crop)
            st.markdown("---")
            st.markdown(f"#### 📖 About *{top_crop.title()}*")
            ci1, ci2 = st.columns(2)
            with ci1:
                st.markdown(f"**🗓️ Season:** {info['season']}")
                st.markdown(f"**📍 Regions:** {info['regions']}")
            with ci2:
                st.markdown(f"**📝** {info['description']}")
                st.markdown(f"**💡 Tips:** {info['tips']}")

            # ── Fertilizer tips ──
            st.markdown("---")
            st.markdown("<div class='section-hdr'>🧪 Fertilizer & Soil Recommendations</div>", unsafe_allow_html=True)
            for tip in get_fertilizer_recommendations(n_val, p_val, k_val, ph_val, rain_val):
                st.markdown(f"<div class='fert-box'>{tip}</div>", unsafe_allow_html=True)

            # ── Input vs avg chart ──
            st.markdown("---")
            st.markdown("#### 📊 Your Inputs vs Dataset Averages")
            avg = df[FEATURE_NAMES].mean()
            chart_df = pd.DataFrame({
                "Feature":     ["N", "P", "K", "Temp", "Humidity", "pH", "Rainfall"],
                "Your Input":  features,
                "Dataset Avg": avg.values,
            })
            fig_bar = px.bar(
                chart_df.melt(id_vars="Feature", var_name="Source", value_name="Value"),
                x="Feature", y="Value", color="Source", barmode="group",
                color_discrete_map={"Your Input": "#4ade80", "Dataset Avg": "#374151"},
                template="plotly_dark",
            )
            fig_bar.update_layout(
                paper_bgcolor="#0a0f0a", plot_bgcolor="#0d1a0d",
                legend=dict(bgcolor="#0d1a0d", font=dict(color="#b8d4b8")),
                margin=dict(t=20, b=20),
                xaxis=dict(gridcolor="#1a3a1a"),
                yaxis=dict(gridcolor="#1a3a1a"),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # ── Feature importance ──
            fi = get_feature_importance(model)
            if not fi.empty:
                st.markdown("#### 🔑 Feature Importance")
                fi_df = fi.reset_index()
                fi_df.columns = ["Feature", "Importance"]
                fi_df["Feature"] = fi_df["Feature"].str.upper().replace(
                    {"TEMPERATURE": "Temp", "HUMIDITY": "Humidity",
                     "RAINFALL": "Rainfall", "PH": "pH"})
                fig_fi = px.bar(
                    fi_df, x="Importance", y="Feature", orientation="h",
                    color="Importance", color_continuous_scale=["#1a3a1a", "#4ade80"],
                    template="plotly_dark",
                )
                fig_fi.update_layout(
                    paper_bgcolor="#0a0f0a", plot_bgcolor="#0d1a0d",
                    coloraxis_showscale=False, margin=dict(t=10, b=10),
                    yaxis={"categoryorder": "total ascending"},
                    xaxis=dict(gridcolor="#1a3a1a"),
                )
                st.plotly_chart(fig_fi, use_container_width=True)

            # ── Save to history & log ──
            log_entry = {
                "Timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "N": n_val, "P": p_val, "K": k_val,
                "Temperature": temp_val, "Humidity": hum_val,
                "pH": ph_val, "Rainfall": rain_val,
                "Top Crop":    top_crop.title(),
                "Confidence":  f"{top_conf:.1f}%",
                "Model":       model_choice,
                "Accuracy":    f"{metrics['Accuracy']:.1f}%",
            }
            st.session_state.history.append(log_entry)
            _append_log(log_entry)
            st.success("✅ Prediction saved to history.")

        else:
            st.markdown("""
            <div style='text-align:center; padding:80px 20px; color:#1a3a1a;
                        background:#0d1a0d; border-radius:16px; border:1px dashed #1a3a1a;'>
              <div style='font-size:4rem; margin-bottom:16px;'>🌿</div>
              <p style='color:#6b9e6b; font-size:1rem; margin:0;'>
                Set your soil parameters on the left<br>and click
                <strong style='color:#4ade80;'>Predict Crop</strong> to get results.
              </p>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE 2 – UPLOAD & PREDICT
# ═══════════════════════════════════════════════════════════════════
elif nav == "📤 Upload & Predict":
    st.title("📤 Upload & Batch Predict")
    st.markdown("<p style='color:#6b9e6b; margin-top:-10px;'>Upload a CSV with soil/climate data and get predictions for every row.</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("""
    <div style='background:#0d1a0d; border:1px solid #1a3a1a; border-radius:12px; padding:16px 20px; margin-bottom:20px;'>
      <strong style='color:#4ade80;'>Required columns:</strong>
      <span style='color:#b8d4b8;'> n, p, k, temperature, humidity, ph, rainfall</span><br>
      <strong style='color:#6b9e6b;'>Optional:</strong>
      <span style='color:#b8d4b8;'> label (for accuracy comparison)</span>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("📂 Upload CSV File", type=["csv"])

    if uploaded:
        df_up, err = load_uploaded_data(uploaded)
        if err:
            st.error(f"❌ Upload error: {err}")
        else:
            st.success(f"✅ Loaded **{len(df_up)}** rows successfully.")
            with st.expander("Preview uploaded data"):
                st.dataframe(df_up.head(10), use_container_width=True)

            model_up = st.selectbox("Select Model", list(MODELS.keys()), key="up_model")

            if st.button("🔍 Run Batch Prediction", use_container_width=True):
                df_base = _load()
                with st.spinner("Training model and running predictions…"):
                    model_trained, _, _ = train_single(model_up, df_base)
                result_df = batch_predict(model_trained, df_up)

                st.markdown("### 📋 Results")
                st.dataframe(result_df, use_container_width=True)

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("### 📊 Confidence Distribution")
                    fig_conf = px.histogram(
                        result_df, x="Confidence_%", nbins=20,
                        color_discrete_sequence=["#4ade80"],
                        template="plotly_dark",
                    )
                    fig_conf.update_layout(
                        paper_bgcolor="#0a0f0a", plot_bgcolor="#0d1a0d",
                        margin=dict(t=20, b=20),
                        xaxis=dict(gridcolor="#1a3a1a"),
                        yaxis=dict(gridcolor="#1a3a1a"),
                    )
                    st.plotly_chart(fig_conf, use_container_width=True)

                with c2:
                    st.markdown("### 🌿 Predicted Crops")
                    crop_counts = result_df["Predicted_Crop"].value_counts().reset_index()
                    crop_counts.columns = ["Crop", "Count"]
                    fig_crops = px.pie(
                        crop_counts, names="Crop", values="Count",
                        color_discrete_sequence=px.colors.qualitative.Pastel,
                        template="plotly_dark",
                    )
                    fig_crops.update_layout(
                        paper_bgcolor="#0a0f0a",
                        legend=dict(bgcolor="#0d1a0d", font=dict(color="#b8d4b8")),
                        margin=dict(t=20, b=20),
                    )
                    st.plotly_chart(fig_crops, use_container_width=True)

                csv_out = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Results CSV", data=csv_out,
                    file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv", use_container_width=True,
                )
    else:
        st.markdown("""
        <div style='text-align:center; padding:60px 20px; background:#0d1a0d;
                    border-radius:16px; border:1px dashed #1a3a1a;'>
          <div style='font-size:3rem; margin-bottom:12px;'>📂</div>
          <p style='color:#6b9e6b;'>Upload a CSV file above to get started.</p>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE 3 – MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════
elif nav == "📊 Model Performance":
    st.title("📊 Model Performance Dashboard")
    st.markdown("<p style='color:#6b9e6b; margin-top:-10px;'>6 classifiers trained on 80/20 stratified split with 5-fold cross-validation.</p>", unsafe_allow_html=True)
    st.markdown("---")

    with st.spinner("Training all 6 models…"):
        results_df, trained_models, best_name, best_model, X_test, y_test = _compare()

    best_acc = results_df.loc[best_name, "Accuracy"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best Model",    best_name.split()[0])
    m2.metric("Best Accuracy", f"{best_acc:.2f}%")
    m3.metric("Models Tested", len(results_df))
    m4.metric("Test Samples",  "440")

    st.success(f"🏆 Best Model: **{best_name}** — Accuracy: **{best_acc:.2f}%**")

    st.markdown("### 📋 Performance Metrics")
    styled = results_df.style.background_gradient(cmap="Greens", axis=0).format("{:.2f}")
    st.dataframe(styled, use_container_width=True)

    st.markdown("### 📈 All Metrics Comparison")
    melted = results_df[["Accuracy","Precision","Recall","F1 Score"]].reset_index().melt(
        id_vars="Model", var_name="Metric", value_name="Score (%)")
    fig_cmp = px.bar(
        melted, x="Model", y="Score (%)", color="Metric", barmode="group",
        color_discrete_sequence=["#4ade80", "#60a5fa", "#f59e0b", "#f87171"],
        template="plotly_dark",
    )
    fig_cmp.update_layout(
        paper_bgcolor="#0a0f0a", plot_bgcolor="#0d1a0d",
        xaxis_tickangle=-20, legend=dict(bgcolor="#0d1a0d", font=dict(color="#b8d4b8")),
        margin=dict(t=20, b=60),
        xaxis=dict(gridcolor="#1a3a1a"), yaxis=dict(gridcolor="#1a3a1a"),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🎯 Accuracy")
        acc_df = results_df["Accuracy"].reset_index()
        acc_df.columns = ["Model", "Accuracy (%)"]
        fig_acc = px.bar(
            acc_df, x="Model", y="Accuracy (%)",
            color="Accuracy (%)", color_continuous_scale=["#1a3a1a", "#4ade80"],
            template="plotly_dark", text="Accuracy (%)",
        )
        fig_acc.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_acc.update_layout(
            paper_bgcolor="#0a0f0a", plot_bgcolor="#0d1a0d",
            coloraxis_showscale=False, xaxis_tickangle=-20,
            margin=dict(t=40, b=60), yaxis_range=[0, 105],
            xaxis=dict(gridcolor="#1a3a1a"), yaxis=dict(gridcolor="#1a3a1a"),
        )
        st.plotly_chart(fig_acc, use_container_width=True)

    with c2:
        st.markdown("### 🔄 Cross-Validation Accuracy")
        cv_df = results_df["CV Accuracy"].reset_index()
        cv_df.columns = ["Model", "CV Accuracy (%)"]
        fig_cv = px.bar(
            cv_df, x="Model", y="CV Accuracy (%)",
            color="CV Accuracy (%)", color_continuous_scale=["#1a2a3a", "#60a5fa"],
            template="plotly_dark", text="CV Accuracy (%)",
        )
        fig_cv.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_cv.update_layout(
            paper_bgcolor="#0a0f0a", plot_bgcolor="#0d1a0d",
            coloraxis_showscale=False, xaxis_tickangle=-20,
            margin=dict(t=40, b=60), yaxis_range=[0, 105],
            xaxis=dict(gridcolor="#1a3a1a"), yaxis=dict(gridcolor="#1a3a1a"),
        )
        st.plotly_chart(fig_cv, use_container_width=True)

    fi = get_feature_importance(best_model)
    if not fi.empty:
        st.markdown(f"### 🔑 Feature Importance – {best_name}")
        fi_df = fi.reset_index()
        fi_df.columns = ["Feature", "Importance"]
        fi_df["Feature"] = fi_df["Feature"].str.upper().replace(
            {"TEMPERATURE": "Temp", "HUMIDITY": "Humidity",
             "RAINFALL": "Rainfall", "PH": "pH"})
        fig_fi = px.bar(
            fi_df, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale=["#1a3a1a", "#4ade80"],
            template="plotly_dark", text="Importance",
        )
        fig_fi.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_fi.update_layout(
            paper_bgcolor="#0a0f0a", plot_bgcolor="#0d1a0d",
            coloraxis_showscale=False, margin=dict(t=20, b=20, r=60),
            yaxis={"categoryorder": "total ascending"},
            xaxis=dict(gridcolor="#1a3a1a"),
        )
        st.plotly_chart(fig_fi, use_container_width=True)
        st.markdown(
            "<div class='explain-box'>💡 <strong>Reading Feature Importance:</strong> "
            "Higher values = more influence on prediction. "
            "Rainfall and humidity typically dominate in crop classification.</div>",
            unsafe_allow_html=True,
        )

    st.markdown(f"### 🔢 Confusion Matrix – {best_name}")
    cm, labels = get_confusion_matrix(best_model, X_test, y_test)
    fig_cm, ax = plt.subplots(figsize=(14, 10))
    fig_cm.patch.set_facecolor("#0a0f0a")
    ax.set_facecolor("#0d1a0d")
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Greens",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, linecolor="#1a3a1a", ax=ax,
    )
    ax.set_xlabel("Predicted", color="#6b9e6b")
    ax.set_ylabel("Actual",    color="#6b9e6b")
    ax.tick_params(colors="#6b9e6b", labelsize=8)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig_cm)
    plt.close(fig_cm)

    with st.expander("📄 Full Classification Report"):
        st.code(get_classification_report(best_model, X_test, y_test), language="text")

    if st.button("💾 Save Best Model to Disk", use_container_width=True):
        path = save_model(best_model, best_name)
        st.success(f"Model saved to `{path}`")


# ═══════════════════════════════════════════════════════════════════
# PAGE 4 – DATA VISUALISATION
# ═══════════════════════════════════════════════════════════════════
elif nav == "📈 Data Visualisation":
    st.title("📈 Data Visualisation")
    st.markdown("<p style='color:#6b9e6b; margin-top:-10px;'>Explore the dataset with interactive charts.</p>", unsafe_allow_html=True)
    st.markdown("---")

    df = _load()
    summary = dataset_summary(df)

    st.markdown("### 📊 Dataset Overview")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Samples",  summary["total_samples"])
    m2.metric("Features",       summary["num_features"])
    m3.metric("Crop Classes",   summary["num_classes"])
    m4.metric("Missing Values", summary["missing_values"])
    m5.metric("Samples/Class",  int(summary["total_samples"] / summary["num_classes"]))

    with st.expander("🗂️ Dataset Preview"):
        st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### 🌿 Crop Class Distribution")
    class_dist = summary["class_distribution"].reset_index()
    class_dist.columns = ["Crop", "Count"]
    fig_dist = px.bar(
        class_dist, x="Crop", y="Count",
        color="Count", color_continuous_scale=["#1a3a1a", "#4ade80"],
        template="plotly_dark",
    )
    fig_dist.update_layout(
        paper_bgcolor="#0a0f0a", plot_bgcolor="#0d1a0d",
        xaxis_tickangle=-45, coloraxis_showscale=False,
        margin=dict(t=20, b=80),
        xaxis=dict(gridcolor="#1a3a1a"), yaxis=dict(gridcolor="#1a3a1a"),
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🌧️ Rainfall vs Crop")
        fig_rain = px.box(
            df, x="label", y="rainfall", color="label",
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            labels={"label": "Crop", "rainfall": "Rainfall (mm)"},
        )
        fig_rain.update_layout(
            paper_bgcolor="#0a0f0a", plot_bgcolor="#0d1a0d",
            xaxis_tickangle=-45, showlegend=False,
            margin=dict(t=20, b=80),
            xaxis=dict(gridcolor="#1a3a1a"), yaxis=dict(gridcolor="#1a3a1a"),
        )
        st.plotly_chart(fig_rain, use_container_width=True)

    with c2:
        st.markdown("### 🌡️ Temperature vs Crop")
        fig_temp = px.box(
            df, x="label", y="temperature", color="label",
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={"label": "Crop", "temperature": "Temperature (°C)"},
        )
        fig_temp.update_layout(
            paper_bgcolor="#0a0f0a", plot_bgcolor="#0d1a0d",
            xaxis_tickangle=-45, showlegend=False,
            margin=dict(t=20, b=80),
            xaxis=dict(gridcolor="#1a3a1a"), yaxis=dict(gridcolor="#1a3a1a"),
        )
        st.plotly_chart(fig_temp, use_container_width=True)

    st.markdown("### 📦 Feature Distribution by Crop")
    display_names = {"n": "N", "p": "P", "k": "K", "temperature": "Temperature",
                     "humidity": "Humidity", "ph": "pH", "rainfall": "Rainfall"}
    sel_feat = st.selectbox("Select Feature", FEATURE_NAMES,
                            format_func=lambda x: display_names.get(x, x),
                            key="vis_feat")
    fig_box = px.violin(
        df, x="label", y=sel_feat, color="label", box=True,
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        labels={"label": "Crop"},
    )
    fig_box.update_layout(
        paper_bgcolor="#0a0f0a", plot_bgcolor="#0d1a0d",
        xaxis_tickangle=-45, showlegend=False,
        margin=dict(t=20, b=80),
        xaxis=dict(gridcolor="#1a3a1a"), yaxis=dict(gridcolor="#1a3a1a"),
    )
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("### 🔗 Feature Correlation Heatmap")
    fig_corr, ax_corr = plt.subplots(figsize=(8, 5))
    fig_corr.patch.set_facecolor("#0a0f0a")
    ax_corr.set_facecolor("#0d1a0d")
    corr = df[FEATURE_NAMES].corr()
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="Greens",
        linewidths=0.5, linecolor="#1a3a1a", ax=ax_corr,
    )
    ax_corr.tick_params(colors="#6b9e6b")
    plt.tight_layout()
    st.pyplot(fig_corr)
    plt.close(fig_corr)

    st.markdown("### 📐 Feature Statistics")
    st.dataframe(summary["describe"].T.style.format("{:.2f}"), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE 5 – ERROR ANALYSIS
# ═══════════════════════════════════════════════════════════════════
elif nav == "🔍 Error Analysis":
    st.title("🔍 Error Analysis")
    st.markdown("<p style='color:#6b9e6b; margin-top:-10px;'>Understand where and why the model makes mistakes.</p>", unsafe_allow_html=True)
    st.markdown("---")

    with st.spinner("Training models…"):
        results_df, trained_models, best_name, best_model, X_test, y_test = _compare()

    err_model_name = st.selectbox(
        "Select Model", list(trained_models.keys()),
        index=list(trained_models.keys()).index(best_name),
    )
    err_model = trained_models[err_model_name]
    summ_e    = error_summary(err_model, X_test, y_test)

    e1, e2, e3 = st.columns(3)
    e1.metric("Total Errors",  summ_e["total_errors"])
    e2.metric("Error Rate",    f"{summ_e['error_rate']:.2f}%")
    e3.metric("Correct",       summ_e["total_samples"] - summ_e["total_errors"])

    if summ_e["total_errors"] == 0:
        st.success("🎉 Perfect classification — no errors on the test set!")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 🔀 Most Confused Pairs")
            confused_df = summ_e["most_confused_pairs"].reset_index()
            confused_df.columns = ["Actual → Predicted", "Count"]
            fig_conf = px.bar(
                confused_df, x="Count", y="Actual → Predicted",
                orientation="h", color="Count",
                color_continuous_scale=["#1a0a0a", "#ef4444"],
                template="plotly_dark",
            )
            fig_conf.update_layout(
                paper_bgcolor="#0a0f0a", plot_bgcolor="#0d1a0d",
                coloraxis_showscale=False, margin=dict(t=20, b=20),
                yaxis={"categoryorder": "total ascending"},
                xaxis=dict(gridcolor="#1a3a1a"),
            )
            st.plotly_chart(fig_conf, use_container_width=True)

        with c2:
            st.markdown("### 📉 Errors per Crop")
            pce = summ_e["per_class_errors"].reset_index()
            pce.columns = ["Crop", "Errors"]
            fig_pce = px.bar(
                pce, x="Crop", y="Errors",
                color="Errors", color_continuous_scale=["#1a0a0a", "#ef4444"],
                template="plotly_dark",
            )
            fig_pce.update_layout(
                paper_bgcolor="#0a0f0a", plot_bgcolor="#0d1a0d",
                coloraxis_showscale=False, xaxis_tickangle=-30,
                margin=dict(t=20, b=60),
                xaxis=dict(gridcolor="#1a3a1a"), yaxis=dict(gridcolor="#1a3a1a"),
            )
            st.plotly_chart(fig_pce, use_container_width=True)

        st.markdown("### 🗂️ Misclassified Samples")
        err_df = get_error_analysis(err_model, X_test, y_test)
        st.dataframe(err_df.head(30), use_container_width=True)

        csv_err = err_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Errors CSV", data=csv_err,
                           file_name=f"errors_{err_model_name.replace(' ','_').lower()}.csv",
                           mime="text/csv")

        st.markdown(
            "<div class='explain-box'>"
            "💡 <strong>Why does the model make mistakes?</strong><br><br>"
            "1. <strong>Similar profiles:</strong> Crops like mungbean and blackgram have nearly identical N/P/K and climate requirements.<br>"
            "2. <strong>Boundary overlap:</strong> With 100 samples per class, decision boundaries between similar crops are thin.<br>"
            "3. <strong>Model choice:</strong> Decision Tree overfits; Random Forest and Gradient Boosting generalise better.<br>"
            "4. <strong>Improvement:</strong> More diverse samples or additional features (soil type, altitude) would reduce errors."
            "</div>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════
# PAGE 6 – HISTORY
# ═══════════════════════════════════════════════════════════════════
elif nav == "📋 History":
    st.title("📋 Prediction History")
    st.markdown("<p style='color:#6b9e6b; margin-top:-10px;'>All predictions made in this session.</p>", unsafe_allow_html=True)
    st.markdown("---")

    history = st.session_state.history
    if not history:
        st.markdown("""
        <div style='text-align:center; padding:60px 20px; background:#0d1a0d;
                    border-radius:16px; border:1px dashed #1a3a1a;'>
          <div style='font-size:3rem; margin-bottom:12px;'>📋</div>
          <p style='color:#6b9e6b;'>No predictions yet. Go to <strong style='color:#4ade80;'>🌱 Predict Crop</strong> to get started.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        hist_df = pd.DataFrame(history)
        h1, h2, h3 = st.columns(3)
        h1.metric("Total Predictions", len(hist_df))
        h2.metric("Unique Crops",       hist_df["Top Crop"].nunique())
        h3.metric("Models Used",        hist_df["Model"].nunique())

        st.dataframe(hist_df, use_container_width=True)

        crop_freq = hist_df["Top Crop"].value_counts().reset_index()
        crop_freq.columns = ["Crop", "Count"]
        fig_hf = px.pie(
            crop_freq, names="Crop", values="Count",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            template="plotly_dark",
        )
        fig_hf.update_layout(
            paper_bgcolor="#0a0f0a",
            legend=dict(bgcolor="#0d1a0d", font=dict(color="#b8d4b8")),
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_hf, use_container_width=True)

        c1, c2, _ = st.columns([1, 1, 3])
        with c1:
            csv_h = hist_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download CSV", data=csv_h,
                               file_name=f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                               mime="text/csv", use_container_width=True)
        with c2:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.history = []; st.rerun()


# ═══════════════════════════════════════════════════════════════════
# PAGE 7 – ABOUT
# ═══════════════════════════════════════════════════════════════════
elif nav == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("---")

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown("""
### 🎓 Project Details
| Field | Details |
|---|---|
| **Title** | Crop Prediction System |
| **Student** | Mridu Ghimire |
| **ID** | 77466817 |
| **Course** | BSc (Hons) Computing – Level 6 |
| **Institution** | Softwarica College of IT & E-Commerce |
""")
        st.markdown("""
### 🎯 Aim
Develop an ML-powered web application that assists Nepali farmers in making
data-driven crop selection decisions based on soil nutrients and climate conditions.
""")
        st.markdown("""
### 📌 Objectives
1. Load and preprocess the Crop Recommendation dataset.
2. Train and compare 6 supervised ML classifiers.
3. Auto-select the best model based on accuracy.
4. Build an interactive Streamlit dashboard.
5. Integrate live weather API for auto temperature/humidity.
6. Provide "Why this crop?" explainability.
7. Implement fertilizer and soil improvement recommendations.
8. Support CSV batch upload and prediction export.
9. Track prediction history with download capability.
10. Implement secure user authentication with SQLite.
""")

    with col_b:
        st.markdown("""
### 🛠️ Technology Stack
| Layer | Technology |
|---|---|
| **UI** | Streamlit |
| **Data** | Pandas, NumPy |
| **ML** | scikit-learn |
| **Visualisation** | Plotly, Seaborn, Matplotlib |
| **Database** | SQLite 3 |
| **Weather** | wttr.in JSON API |
| **Persistence** | joblib |
| **Language** | Python 3.10+ |
""")
        st.markdown("""
### 📂 Dataset
- **Name:** Crop Recommendation Dataset
- **Source:** UCI ML Repository / Kaggle
- **Samples:** 2,200 (100 per class, balanced)
- **Features:** N, P, K, Temperature, Humidity, pH, Rainfall
- **Target:** 22 crop classes
""")
        st.markdown("""
### 🔬 System Flow
```
User Input / CSV Upload
        ↓
Input Validation
        ↓
ML Model Training (6 algorithms)
        ↓
Best Model Auto-Selected
        ↓
Prediction + Confidence Score
        ↓
Explainability + Fertilizer Tips
        ↓
Visualisations + History Export
```
""")

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#1a3a1a; font-size:.85rem;'>"
        "© 2024 Mridu Ghimire · Crop Prediction System · BSc (Hons) Computing Level 6"
        "</p>",
        unsafe_allow_html=True,
    )
