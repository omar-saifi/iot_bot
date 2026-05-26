import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings, os, json, joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score, roc_curve
)
import category_encoders as ce
import io

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="IDS — BoT-IoT",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f0c29, #1a1a3e, #24243e) !important;
}
[data-testid="stSidebar"] * { color: #e0e0ff !important; }
[data-testid="stSidebar"] .stSlider > div > div > div { background: #4f46e5 !important; }

/* ── Main background ── */
[data-testid="stAppViewContainer"] {
    background: #0d0d1a;
    color: #e0e0ff;
}

/* ── Cards ── */
.card {
    background: linear-gradient(135deg, #1a1a3e 0%, #16213e 100%);
    border: 1px solid #2d2d6e;
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 16px;
    box-shadow: 0 4px 24px rgba(79,70,229,0.15);
}
.card-title {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #a0a0cc;
    margin-bottom: 6px;
}
.card-value {
    font-size: 2rem;
    font-weight: 700;
    color: #e0e0ff;
}

/* ── Result banner ── */
.result-safe {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 1px solid #10b981;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
}
.result-attack {
    background: linear-gradient(135deg, #450a0a, #7f1d1d);
    border: 1px solid #ef4444;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
}
.result-title { font-size: 2.2rem; font-weight: 700; margin: 0; }
.result-sub   { font-size: 1rem; color: #cbd5e1; margin-top: 8px; }

/* ── Probability bar ── */
.prob-bar-wrap { background: #1e1e3f; border-radius: 8px; height: 22px; width: 100%; overflow: hidden; }
.prob-bar-fill { height: 22px; border-radius: 8px; transition: width 0.6s ease; }

/* ── Section headers ── */
.section-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #818cf8;
    border-left: 4px solid #4f46e5;
    padding-left: 10px;
    margin: 24px 0 12px 0;
}

/* ── Metric row ── */
.metric-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }
.metric-chip {
    background: #1e1e3f;
    border: 1px solid #3730a3;
    border-radius: 10px;
    padding: 10px 18px;
    min-width: 130px;
    text-align: center;
}
.metric-chip .label { font-size: 0.72rem; color: #a0a0cc; text-transform: uppercase; letter-spacing: 0.08em; }
.metric-chip .value { font-size: 1.4rem; font-weight: 700; color: #818cf8; }

/* ── Streamlit overrides ── */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 28px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.88; }
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"]  input,
div[data-testid="stSelectbox"] > div { 
    background: #1a1a3e !important; 
    color: #e0e0ff !important; 
    border: 1px solid #3730a3 !important;
    border-radius: 8px !important;
}
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 { color: #c7d2fe; }
.stTabs [data-baseweb="tab"] { color: #a0a0cc !important; }
.stTabs [aria-selected="true"] { color: #818cf8 !important; border-bottom: 2px solid #4f46e5 !important; }

/* ── Batch results table ── */
.batch-safe   { color: #10b981; font-weight: 600; }
.batch-attack { color: #ef4444; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  CONSTANTS — BoT-IoT known columns
# ─────────────────────────────────────────────
BINARY_TARGET = "label"
CATEGORY_TARGET = "category"
SUBCATEGORY_TARGET = "subcategory"

KNOWN_CATEGORIES = {
    "Normal":          {"Normal": "No attack detected — traffic is benign"},
    "DDoS":            {
        "HTTP":   "HTTP Flood — overwhelms web server with requests",
        "TCP":    "TCP SYN Flood — exhausts server connection table",
        "UDP":    "UDP Flood — saturates bandwidth with UDP packets",
    },
    "DoS":             {
        "HTTP":   "HTTP DoS — single-source HTTP request flood",
        "TCP":    "TCP DoS — single-source SYN exhaustion",
        "UDP":    "UDP DoS — single-source UDP flood",
    },
    "Reconnaissance":  {
        "OS Fingerprint": "OS scanning to identify target system",
        "Service Scan":   "Port/service enumeration on target",
    },
    "Theft":           {
        "Data Exfiltration": "Unauthorized data transfer to external host",
        "Keylogging":        "Capturing keystrokes from infected device",
    },
}

ATTACK_COLORS = {
    "Normal":         "#10b981",
    "DDoS":           "#ef4444",
    "DoS":            "#f97316",
    "Reconnaissance": "#eab308",
    "Theft":          "#a855f7",
}

SEVERITY = {
    "Normal":         ("🟢", "SAFE",     "#10b981"),
    "DDoS":           ("🔴", "CRITICAL", "#ef4444"),
    "DoS":            ("🟠", "HIGH",     "#f97316"),
    "Reconnaissance": ("🟡", "MEDIUM",   "#eab308"),
    "Theft":          ("🟣", "HIGH",     "#a855f7"),
}


# ─────────────────────────────────────────────
#  TRAIN / LOAD MODEL
# ─────────────────────────────────────────────
MODEL_DIR = "ids_models"

@st.cache_resource(show_spinner="🔧 Training IDS models…")
def load_or_train_models(csv_path: str):
    """Load saved models or train fresh from CSV."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(csv_path)

    # ── Strip whitespace from ALL column names ──
    df.columns = df.columns.str.strip()

    cols_lower = {c: c.lower().strip() for c in df.columns}

    binary_col = next(
        (c for c in df.columns if cols_lower[c] in ["attack", "label", "labels", "class", "attack_label"]),
        None
    )

    if binary_col and df[binary_col].nunique() > 2:
        binary_col = None

    cat_col = next(
        (c for c in df.columns if cols_lower[c] in ["category", "attack_type", "type", "attack_category"]),
        None
    )
    if not cat_col:
        cat_col = next(
            (c for c in df.columns if cols_lower[c] == "attack" and c != binary_col),
            None
        )

    subcat_col = next(
        (c for c in df.columns if cols_lower[c] in ["subcategory", "sub_category", "subcategory_name", "subcat"]),
        None
    )

    if not binary_col:
        detected = list(df.columns)
        raise ValueError(
            f"Cannot find binary label column.\n"
            f"Columns detected: {detected}\n"
            f"Expected one of: attack, label, class, attack_label"
        )
    if not cat_col:
        raise ValueError(
            f"Cannot find category column.\n"
            f"Expected one of: category, attack_type, type"
        )

    target_cols = [c for c in [binary_col, cat_col, subcat_col] if c]

    DROP_ALWAYS = [
        "pkseqid", "stime", "ltime",
        "smac", "dmac", "soui", "doui",
        "sco", "dco", "saddr", "daddr",
    ]
    drop_cols  = [c for c in df.columns if c.lower() in DROP_ALWAYS]
    feature_df = df.drop(columns=target_cols + drop_cols, errors="ignore")

    const_mask = feature_df.nunique() > 1
    feature_df = feature_df.loc[:, const_mask]

    num_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()

    for c in num_cols: feature_df[c] = feature_df[c].fillna(feature_df[c].median())
    for c in cat_cols: feature_df[c] = feature_df[c].fillna(feature_df[c].mode()[0])

    if len(num_cols) > 1:
        corr = feature_df[num_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop_corr = [c for c in upper.columns if any(upper[c] > 0.98)]
        feature_df.drop(columns=drop_corr, inplace=True)
        num_cols = [c for c in num_cols if c not in drop_corr]

    all_feat_cols = num_cols + cat_cols

    le_binary = LabelEncoder()
    le_cat    = LabelEncoder()
    y_binary  = le_binary.fit_transform(df[binary_col].astype(str))
    y_cat     = le_cat.fit_transform(df[cat_col].astype(str))

    le_subcat = None
    y_subcat  = None
    if subcat_col:
        le_subcat = LabelEncoder()
        y_subcat  = le_subcat.fit_transform(df[subcat_col].astype(str))

    X = feature_df[all_feat_cols].copy()

    te = None
    if cat_cols:
        te = ce.TargetEncoder(cols=cat_cols, smoothing=1.0)
        X[cat_cols] = te.fit_transform(X[cat_cols], y_cat)

    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_tr_b, X_te_b, y_tr_b, y_te_b = train_test_split(X_scaled, y_binary, test_size=0.2, random_state=42, stratify=y_binary)
    X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(X_scaled, y_cat,    test_size=0.2, random_state=42, stratify=y_cat)

    gnb  = GaussianNB().fit(X_tr_b, y_tr_b)
    cnb  = ComplementNB(alpha=1.0).fit(X_tr_c, y_tr_c)

    gnb_sub = None
    if y_subcat is not None:
        X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(X_scaled, y_subcat, test_size=0.2, random_state=42, stratify=y_subcat)
        gnb_sub = ComplementNB(alpha=1.0).fit(X_tr_s, y_tr_s)

    metrics = {
        "binary": {
            "accuracy":  round(accuracy_score(y_te_b, gnb.predict(X_te_b)),  4),
            "f1":        round(f1_score(y_te_b, gnb.predict(X_te_b), average="weighted"), 4),
        },
        "category": {
            "accuracy":  round(accuracy_score(y_te_c, cnb.predict(X_te_c)),  4),
            "f1":        round(f1_score(y_te_c, cnb.predict(X_te_c), average="weighted", zero_division=0), 4),
        }
    }

    col_stats = {}
    for c in all_feat_cols:
        if c in num_cols:
            col_stats[c] = {
                "type": "numeric",
                "min":  float(feature_df[c].min()),
                "max":  float(feature_df[c].max()),
                "mean": float(feature_df[c].mean()),
                "std":  float(feature_df[c].std()),
            }
        else:
            col_stats[c] = {
                "type":   "categorical",
                "values": sorted([str(v) for v in df[c].dropna().unique().tolist()]),
            }

    return {
        "gnb": gnb, "cnb": cnb, "gnb_sub": gnb_sub,
        "scaler": scaler, "te": te,
        "le_binary": le_binary, "le_cat": le_cat, "le_subcat": le_subcat,
        "num_cols": num_cols, "cat_cols": cat_cols,
        "all_feat_cols": all_feat_cols,
        "col_stats": col_stats,
        "metrics": metrics,
        "df": df,
        "binary_col": binary_col,
        "cat_col": cat_col, "subcat_col": subcat_col,
        "cat_classes": list(le_cat.classes_),
        "subcat_classes": list(le_subcat.classes_) if le_subcat else [],
    }


def predict(state, user_input: dict):
    """Run full prediction pipeline on user input dict."""
    X = pd.DataFrame([user_input], columns=state["all_feat_cols"])

    for c in state["num_cols"]:
        if c not in X.columns: X[c] = 0.0
    for c in state["cat_cols"]:
        if c not in X.columns: X[c] = state["col_stats"][c]["values"][0]

    X = X[state["all_feat_cols"]]

    if state["te"] and state["cat_cols"]:
        X[state["cat_cols"]] = state["te"].transform(X[state["cat_cols"]])

    X_scaled = state["scaler"].transform(X)

    bin_pred  = state["gnb"].predict(X_scaled)[0]
    bin_proba = state["gnb"].predict_proba(X_scaled)[0]

    cat_pred  = state["cnb"].predict(X_scaled)[0]
    cat_proba = state["cnb"].predict_proba(X_scaled)[0]

    sub_pred  = None
    sub_proba = None
    if state["gnb_sub"]:
        sub_pred  = state["gnb_sub"].predict(X_scaled)[0]
        sub_proba = state["gnb_sub"].predict_proba(X_scaled)[0]

    return {
        "binary_label":  state["le_binary"].inverse_transform([bin_pred])[0],
        "binary_proba":  bin_proba,
        "cat_label":     state["le_cat"].inverse_transform([cat_pred])[0],
        "cat_proba":     cat_proba,
        "cat_classes":   state["cat_classes"],
        "sub_label":     state["le_subcat"].inverse_transform([sub_pred])[0] if sub_pred is not None else None,
        "sub_proba":     sub_proba,
        "sub_classes":   state["subcat_classes"],
    }


def predict_batch(state, df_input: pd.DataFrame):
    """Run predictions on a full DataFrame. Returns results DataFrame."""
    # Align columns: keep only known feature cols, fill missing with defaults
    X = pd.DataFrame(index=df_input.index)
    for c in state["all_feat_cols"]:
        if c in df_input.columns:
            X[c] = df_input[c]
        elif c in state["num_cols"]:
            X[c] = state["col_stats"][c]["mean"]
        else:
            X[c] = state["col_stats"][c]["values"][0]

    X = X[state["all_feat_cols"]].copy()

    # Fill NaNs
    for c in state["num_cols"]:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(state["col_stats"][c]["mean"])
    for c in state["cat_cols"]:
        if c in X.columns:
            X[c] = X[c].fillna(state["col_stats"][c]["values"][0]).astype(str)

    if state["te"] and state["cat_cols"]:
        X[state["cat_cols"]] = state["te"].transform(X[state["cat_cols"]])

    X_scaled = state["scaler"].transform(X)

    bin_preds  = state["gnb"].predict(X_scaled)
    bin_probas = state["gnb"].predict_proba(X_scaled)
    cat_preds  = state["cnb"].predict(X_scaled)
    cat_probas = state["cnb"].predict_proba(X_scaled)

    bin_labels = state["le_binary"].inverse_transform(bin_preds)
    cat_labels = state["le_cat"].inverse_transform(cat_preds)

    sub_labels = [None] * len(df_input)
    if state["gnb_sub"]:
        sub_preds  = state["gnb_sub"].predict(X_scaled)
        sub_labels = state["le_subcat"].inverse_transform(sub_preds)

    results = []
    for i in range(len(df_input)):
        bl = str(bin_labels[i])
        is_attack = bl not in ["0", "normal", "benign"]
        cat = str(cat_labels[i])
        icon, severity, color = SEVERITY.get(cat, ("⚪", "UNKNOWN", "#888"))
        bin_conf = float(np.max(bin_probas[i]))
        cat_conf = float(np.max(cat_probas[i]))
        results.append({
            "Row":         i + 1,
            "Detection":   "⚠️ Attack" if is_attack else "✅ Normal",
            "Category":    cat,
            "Severity":    severity,
            "Subcategory": str(sub_labels[i]) if sub_labels[i] else "N/A",
            "Bin Conf %":  round(bin_conf * 100, 1),
            "Cat Conf %":  round(cat_conf * 100, 1),
            "_is_attack":  is_attack,
            "_color":      color,
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            border-radius: 20px; padding: 30px 36px; margin-bottom: 28px;
            border: 1px solid #3730a3;">
  <div style="display:flex; align-items:center; gap:16px;">
    <span style="font-size:3rem;">🛡️</span>
    <div>
      <h1 style="margin:0; font-size:2rem; font-weight:800; color:#e0e0ff;">
        Intrusion Detection System
      </h1>
      <p style="margin:4px 0 0; color:#a0a0cc; font-size:0.95rem;">
        BoT-IoT Dataset &nbsp;•&nbsp; Target Encoding → Min-Max Scaling → Naive Bayes
      </p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SIDEBAR — Dataset Upload
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    uploaded = st.file_uploader("📂 Upload your CSV dataset", type=["csv"])
    csv_path = None

    if uploaded:
        tmp_path = f"/tmp/{uploaded.name}"
        with open(tmp_path, "wb") as f:
            f.write(uploaded.read())
        csv_path = tmp_path
        st.success(f"✅ Loaded: `{uploaded.name}`")
    else:
        default = "data_1.csv"
        if os.path.exists(default):
            csv_path = default
            st.info(f"📁 Using: `{default}`")
        else:
            st.warning("⚠️ Upload your `data_1.csv` to begin.")

    st.markdown("---")
    st.markdown("### 🧠 Model Info")
    st.markdown("""
- **Encoding**: Target Encoding  
- **Scaling**: Min-Max [0, 1]  
- **Binary**: GaussianNB  
- **Category**: ComplementNB  
- **Subcategory**: ComplementNB  
    """)

    st.markdown("---")
    st.markdown("### 📘 Attack Legend")
    for cat, (icon, sev, color) in SEVERITY.items():
        st.markdown(
            f'<span style="color:{color};font-weight:600;">{icon} {cat}</span> '
            f'<span style="color:#888;font-size:0.8rem;">— {sev}</span>',
            unsafe_allow_html=True
        )


# ─────────────────────────────────────────────
#  MAIN CONTENT
# ─────────────────────────────────────────────
if not csv_path:
    st.info("👆 Upload your dataset in the sidebar to get started.")
    st.stop()

try:
    state = load_or_train_models(csv_path)
except Exception as e:
    st.error(f"❌ Error loading/training models: {e}")
    st.stop()

metrics  = state["metrics"]
col_stats = state["col_stats"]

# ── Top KPI bar ──
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""<div class="card">
        <div class="card-title">Binary Accuracy</div>
        <div class="card-value">{metrics['binary']['accuracy']*100:.1f}%</div>
    </div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="card">
        <div class="card-title">Binary F1-Score</div>
        <div class="card-value">{metrics['binary']['f1']:.4f}</div>
    </div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class="card">
        <div class="card-title">Category Accuracy</div>
        <div class="card-value">{metrics['category']['accuracy']*100:.1f}%</div>
    </div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class="card">
        <div class="card-title">Category F1-Score</div>
        <div class="card-value">{metrics['category']['f1']:.4f}</div>
    </div>""", unsafe_allow_html=True)

# ── Tabs ──
tab1, tab2, tab3 = st.tabs([
    "🔮  Prediction",
    "📂  Batch CSV Test",
    "📊  Dataset Overview",

])


# ════════════════════════════════════════════
#  TAB 1 — PREDICTION
# ════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">🎛️ Configure Network Traffic Parameters</div>', unsafe_allow_html=True)

    feat_cols = state["all_feat_cols"]
    n = len(feat_cols)

    user_input = {}

    cols_per_row = 3
    rows = [feat_cols[i:i+cols_per_row] for i in range(0, n, cols_per_row)]

    with st.expander("📐 Set Feature Values", expanded=True):
        for row in rows:
            cols = st.columns(len(row))
            for col_ui, feat in zip(cols, row):
                info = col_stats[feat]
                with col_ui:
                    if info["type"] == "numeric":
                        val = col_ui.number_input(
                            label=feat,
                            min_value=float(info["min"]),
                            max_value=float(info["max"]),
                            value=float(info["mean"]),
                            step=max(float((info["max"] - info["min"]) / 100), 0.0001),
                            format="%.4f",
                            key=f"feat_{feat}"
                        )
                    else:
                        opts = info["values"]
                        val  = col_ui.selectbox(
                            label=feat,
                            options=opts,
                            index=0,
                            key=f"feat_{feat}"
                        )
                    user_input[feat] = val

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-header">⚡ Quick Presets (simulate known traffic patterns)</div>', unsafe_allow_html=True)
    p1, p2, p3, p4, p5 = st.columns(5)

    def apply_preset(name):
        st.session_state["preset"] = name

    with p1:
        if st.button("🟢 Normal Traffic"):   apply_preset("normal")
    with p2:
        if st.button("🔴 DDoS Attack"):      apply_preset("ddos")
    with p3:
        if st.button("🟠 DoS Attack"):       apply_preset("dos")
    with p4:
        if st.button("🟡 Recon Scan"):       apply_preset("recon")
    with p5:
        if st.button("🟣 Data Theft"):       apply_preset("theft")

    preset = st.session_state.get("preset", None)
    if preset:
        df_raw = state["df"]
        cat_col   = state["cat_col"]
        for feat in state["num_cols"]:
            if feat not in df_raw.columns:
                continue
            info = col_stats[feat]
            if preset == "normal":
                sub = df_raw[df_raw[cat_col].str.lower().str.contains("normal", na=False)][feat]
            elif preset == "ddos":
                sub = df_raw[df_raw[cat_col].str.lower().str.contains("ddos", na=False)][feat]
            elif preset == "dos":
                sub = df_raw[df_raw[cat_col].str.lower().str.contains("^dos", regex=True, na=False)][feat]
            elif preset == "recon":
                sub = df_raw[df_raw[cat_col].str.lower().str.contains("recon", na=False)][feat]
            elif preset == "theft":
                sub = df_raw[df_raw[cat_col].str.lower().str.contains("theft", na=False)][feat]
            else:
                sub = pd.Series(dtype=float)

            if len(sub) > 0:
                user_input[feat] = float(sub.sample(1).iloc[0])

    st.markdown("<br>", unsafe_allow_html=True)
    c_btn = st.columns([1, 2, 1])
    with c_btn[1]:
        run = st.button("🔍  ANALYZE TRAFFIC", use_container_width=True)

    if run or st.session_state.get("last_result"):
        if run:
            result = predict(state, user_input)
            st.session_state["last_result"] = result
        else:
            result = st.session_state["last_result"]

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">📋 Prediction Results</div>', unsafe_allow_html=True)

        cat   = result["cat_label"]
        sub   = result["sub_label"]
        icon, severity, color = SEVERITY.get(cat, ("⚪", "UNKNOWN", "#888"))
        is_attack = result["binary_label"] != "0" and str(result["binary_label"]).lower() not in ["0", "normal", "benign"]

        css_class = "result-safe" if not is_attack else "result-attack"
        desc = ""
        if cat in KNOWN_CATEGORIES:
            if sub and sub in KNOWN_CATEGORIES[cat]:
                desc = KNOWN_CATEGORIES[cat][sub]
            else:
                desc = list(KNOWN_CATEGORIES[cat].values())[0]

        st.markdown(f"""
        <div class="{css_class}">
            <p class="result-title" style="color:{color};">{icon} {cat}</p>
            <p class="result-sub">
                <b>Severity:</b> {severity} &nbsp;|&nbsp;
                <b>Subcategory:</b> {sub or "N/A"} &nbsp;|&nbsp;
                <b>Binary:</b> {"⚠️ Attack Detected" if is_attack else "✅ Normal Traffic"}
            </p>
            <p style="color:#94a3b8; font-size:0.9rem; margin-top:10px;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("**🧮 Category Probabilities**")
            cat_proba  = result["cat_proba"]
            cat_classes = result["cat_classes"]
            sorted_idx = np.argsort(cat_proba)[::-1]
            for idx in sorted_idx:
                label = cat_classes[idx]
                prob  = cat_proba[idx]
                bar_color = ATTACK_COLORS.get(label, "#6366f1")
                st.markdown(f"""
                <div style="margin-bottom:10px;">
                  <div style="display:flex; justify-content:space-between; color:#c7d2fe; font-size:0.85rem; margin-bottom:3px;">
                    <span>{label}</span><span>{prob*100:.1f}%</span>
                  </div>
                  <div class="prob-bar-wrap">
                    <div class="prob-bar-fill" style="width:{prob*100:.1f}%; background:{bar_color};"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        with col_r:
            st.markdown("**⚖️ Binary Detection Confidence**")
            bin_proba  = result["binary_proba"]
            bin_labels = [str(c) for c in state["le_binary"].classes_]
            for i, (label, prob) in enumerate(zip(bin_labels, bin_proba)):
                bar_col = "#10b981" if i == 0 else "#ef4444"
                st.markdown(f"""
                <div style="margin-bottom:10px;">
                  <div style="display:flex; justify-content:space-between; color:#c7d2fe; font-size:0.85rem; margin-bottom:3px;">
                    <span>{label}</span><span>{prob*100:.1f}%</span>
                  </div>
                  <div class="prob-bar-wrap">
                    <div class="prob-bar-fill" style="width:{prob*100:.1f}%; background:{bar_col};"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            if result["sub_label"] and result["sub_proba"] is not None:
                st.markdown("**🔬 Subcategory Probabilities**")
                sub_proba   = result["sub_proba"]
                sub_classes = result["sub_classes"]
                top_sub = np.argsort(sub_proba)[::-1][:5]
                for idx in top_sub:
                    label = sub_classes[idx]
                    prob  = sub_proba[idx]
                    st.markdown(f"""
                    <div style="margin-bottom:8px;">
                      <div style="display:flex; justify-content:space-between; color:#c7d2fe; font-size:0.82rem; margin-bottom:2px;">
                        <span>{label}</span><span>{prob*100:.1f}%</span>
                      </div>
                      <div class="prob-bar-wrap">
                        <div class="prob-bar-fill" style="width:{prob*100:.1f}%; background:#8b5cf6;"></div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">📡 Input Feature Profile</div>', unsafe_allow_html=True)
        top_features = state["num_cols"][:min(8, len(state["num_cols"]))]
        if top_features:
            vals = [user_input.get(f, col_stats[f]["mean"]) for f in top_features]
            mins = [col_stats[f]["min"] for f in top_features]
            maxs = [col_stats[f]["max"] for f in top_features]
            norm_vals = [
                (v - mn) / (mx - mn) if mx != mn else 0.5
                for v, mn, mx in zip(vals, mins, maxs)
            ]

            angles = np.linspace(0, 2 * np.pi, len(top_features), endpoint=False).tolist()
            norm_vals_c = norm_vals + [norm_vals[0]]
            angles_c    = angles + [angles[0]]
            labels      = [f[:12] for f in top_features]

            fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
            fig.patch.set_facecolor("#0d0d1a")
            ax.set_facecolor("#0d0d1a")
            ax.plot(angles_c, norm_vals_c, color=color, linewidth=2)
            ax.fill(angles_c, norm_vals_c, alpha=0.25, color=color)
            ax.set_xticks(angles)
            ax.set_xticklabels(labels, size=8, color="#c7d2fe")
            ax.set_yticklabels([], color="#555")
            ax.spines["polar"].set_color("#2d2d6e")
            ax.grid(color="#2d2d6e", linestyle="--", alpha=0.6)
            ax.set_title("Normalized Feature Profile", color="#c7d2fe", pad=16)
            _, radar_col, _ = st.columns([1, 2, 1])
            with radar_col:
                st.pyplot(fig, use_container_width=True)
            plt.close(fig)


# ════════════════════════════════════════════
#  TAB 2 — BATCH CSV TEST
# ════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">📂 Batch CSV Testing</div>', unsafe_allow_html=True)

   

    # ── Expected columns hint ──
    with st.expander("🔍 Expected feature columns (from training data)", expanded=False):
        st.markdown(
            "Your CSV should contain **some or all** of these columns "
            "(missing ones are filled with training-set means):"
        )
        feat_list = state["all_feat_cols"]
        cols_display = st.columns(4)
        for i, feat in enumerate(feat_list):
            cols_display[i % 4].markdown(
                f'<span style="color:#818cf8; font-size:0.82rem;">• {feat}</span>',
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── File uploader ──
    batch_file = st.file_uploader(
        "📤 Upload CSV for batch prediction",
        type=["csv"],
        key="batch_uploader",
        help="Rows without label columns — just raw feature data."
    )

    # ── Sample data download ──
    sample_cols = state["all_feat_cols"]
    sample_rows = []
    df_raw = state["df"]
    for _ in range(5):
        row = {}
        for c in sample_cols:
            if c in df_raw.columns:
                row[c] = df_raw[c].dropna().sample(1).iloc[0]
            elif state["col_stats"][c]["type"] == "numeric":
                row[c] = state["col_stats"][c]["mean"]
            else:
                row[c] = state["col_stats"][c]["values"][0]
        sample_rows.append(row)
    sample_df = pd.DataFrame(sample_rows)
    sample_csv = sample_df.to_csv(index=False)
    st.download_button(
        label="⬇️ Download sample template CSV",
        data=sample_csv,
        file_name="batch_template.csv",
        mime="text/csv",
        help="Download a 5-row template with the correct column structure."
    )

    if batch_file is not None:
        try:
            df_batch = pd.read_csv(batch_file)
            df_batch.columns = df_batch.columns.str.strip()

            # Drop target columns if present (user may have uploaded labeled data)
            drop_targets = [
                state["binary_col"], state["cat_col"], state["subcat_col"],
                "label", "labels", "category", "subcategory", "attack", "class"
            ]
            drop_targets = [c for c in drop_targets if c and c in df_batch.columns]
            df_batch_feats = df_batch.drop(columns=drop_targets, errors="ignore")

            st.success(f"✅ Loaded **{len(df_batch):,} rows** × **{len(df_batch.columns)} columns**")

            # ── Preview ──
            with st.expander("👁️ Preview uploaded data (first 5 rows)", expanded=False):
                st.dataframe(df_batch.head(), use_container_width=True)

            # ── Run batch prediction ──
            with st.spinner("⚙️ Running predictions on all rows…"):
                results_df = predict_batch(state, df_batch_feats)

            # ── Summary KPIs ──
            st.markdown('<div class="section-header">📊 Batch Summary</div>', unsafe_allow_html=True)

            total     = len(results_df)
            n_attack  = results_df["_is_attack"].sum()
            n_normal  = total - n_attack
            attack_pct = n_attack / total * 100 if total > 0 else 0

            bk1, bk2, bk3, bk4 = st.columns(4)
            with bk1:
                st.markdown(f"""<div class="card">
                    <div class="card-title">Total Rows</div>
                    <div class="card-value">{total:,}</div>
                </div>""", unsafe_allow_html=True)
            with bk2:
                st.markdown(f"""<div class="card">
                    <div class="card-title">⚠️ Attacks</div>
                    <div class="card-value" style="color:#ef4444;">{n_attack:,}</div>
                </div>""", unsafe_allow_html=True)
            with bk3:
                st.markdown(f"""<div class="card">
                    <div class="card-title">✅ Normal</div>
                    <div class="card-value" style="color:#10b981;">{n_normal:,}</div>
                </div>""", unsafe_allow_html=True)
            with bk4:
                st.markdown(f"""<div class="card">
                    <div class="card-title">Attack Rate</div>
                    <div class="card-value" style="color:#f97316;">{attack_pct:.1f}%</div>
                </div>""", unsafe_allow_html=True)

            # ── Category breakdown chart + confidence dist ──
            chart_l, chart_r = st.columns(2)

            with chart_l:
                st.markdown("**Category Breakdown**")
                cat_counts = results_df["Category"].value_counts()
                fig, ax = plt.subplots(figsize=(5, 3.5))
                fig.patch.set_facecolor("#0d0d1a")
                ax.set_facecolor("#1a1a3e")
                colors_bar = [ATTACK_COLORS.get(c, "#6366f1") for c in cat_counts.index]
                bars = ax.barh(cat_counts.index, cat_counts.values, color=colors_bar, edgecolor="#0d0d1a", height=0.55)
                for bar, val in zip(bars, cat_counts.values):
                    ax.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                            str(val), va="center", color="#c7d2fe", fontsize=9)
                ax.set_xlabel("Count", color="#a0a0cc")
                ax.tick_params(colors="#c7d2fe")
                ax.spines[:].set_color("#2d2d6e")
                ax.set_title("Predicted Categories", color="#c7d2fe", fontweight="bold")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            with chart_r:
                st.markdown("**Category Confidence Distribution**")
                fig, ax = plt.subplots(figsize=(5, 3.5))
                fig.patch.set_facecolor("#0d0d1a")
                ax.set_facecolor("#1a1a3e")
                ax.hist(results_df["Cat Conf %"], bins=20, color="#818cf8", edgecolor="#0d0d1a", alpha=0.85)
                ax.set_xlabel("Confidence %", color="#a0a0cc")
                ax.set_ylabel("Count", color="#a0a0cc")
                ax.tick_params(colors="#c7d2fe")
                ax.spines[:].set_color("#2d2d6e")
                ax.set_title("Category Prediction Confidence", color="#c7d2fe", fontweight="bold")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            # ── Results table ──
            st.markdown('<div class="section-header">📋 Row-by-Row Results</div>', unsafe_allow_html=True)

            display_df = results_df.drop(columns=["_is_attack", "_color"])

            # Color-code the Detection column
            def style_detection(val):
                if "Attack" in str(val):
                    return "color: #ef4444; font-weight: 600;"
                return "color: #10b981; font-weight: 600;"

            def style_severity(val):
                color_map = {
                    "CRITICAL": "#ef4444",
                    "HIGH":     "#f97316",
                    "MEDIUM":   "#eab308",
                    "SAFE":     "#10b981",
                    "UNKNOWN":  "#888",
                }
                return f"color: {color_map.get(str(val), '#c7d2fe')}; font-weight: 600;"

            styled = (
                display_df.style
                .applymap(style_detection, subset=["Detection"])
                .applymap(style_severity, subset=["Severity"])
                .format({"Bin Conf %": "{:.1f}%", "Cat Conf %": "{:.1f}%"})
                .set_properties(**{"background-color": "#1a1a3e", "color": "#c7d2fe"})
            )

            st.dataframe(styled, use_container_width=True, height=420)

            # ── Download results ──
            st.markdown("<br>", unsafe_allow_html=True)
            csv_out = display_df.to_csv(index=False)
            dl_col1, dl_col2, _ = st.columns([1, 1, 2])
            with dl_col1:
                st.download_button(
                    label="⬇️ Download Results CSV",
                    data=csv_out,
                    file_name="ids_batch_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with dl_col2:
                # Attack-only filter download
                attacks_only = display_df[results_df["_is_attack"]]
                if len(attacks_only) > 0:
                    st.download_button(
                        label="🚨 Download Attacks Only",
                        data=attacks_only.to_csv(index=False),
                        file_name="ids_attacks_only.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

        except Exception as e:
            st.error(f"❌ Error processing batch file: {e}")
            st.exception(e)

    else:
        # Empty state with instructions
        st.markdown("""
        <div style="border: 2px dashed #3730a3; border-radius: 16px; padding: 48px; text-align: center; margin-top: 24px;">
          <div style="font-size: 3rem; margin-bottom: 12px;">📤</div>
          <p style="color: #818cf8; font-size: 1.1rem; font-weight: 600; margin: 0;">
            Upload a CSV file above to test multiple traffic samples at once
          </p>
          <p style="color: #555; font-size: 0.88rem; margin-top: 8px;">
            Supports any number of rows · Missing columns auto-filled with training means · Results downloadable as CSV
          </p>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════
#  TAB 3 — DATASET OVERVIEW
# ════════════════════════════════════════════
with tab3:
    df = state["df"]
    st.markdown('<div class="section-header">📊 Dataset Summary</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Total Records",  f"{len(df):,}")
    with m2: st.metric("Total Features", f"{len(state['all_feat_cols'])}")
    with m3: st.metric("Numeric Cols",   f"{len(state['num_cols'])}")
    with m4: st.metric("Categorical",    f"{len(state['cat_cols'])}")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Category Distribution**")
        cat_counts = df[state["cat_col"]].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor("#0d0d1a")
        ax.set_facecolor("#1a1a3e")
        colors = [ATTACK_COLORS.get(c, "#6366f1") for c in cat_counts.index]
        bars = ax.barh(cat_counts.index, cat_counts.values, color=colors, edgecolor="#0d0d1a", height=0.6)
        for bar, val in zip(bars, cat_counts.values):
            ax.text(val * 1.01, bar.get_y() + bar.get_height()/2,
                    f"{val:,}", va="center", color="#c7d2fe", fontsize=8)
        ax.set_xlabel("Count", color="#a0a0cc")
        ax.tick_params(colors="#c7d2fe")
        ax.spines[:].set_color("#2d2d6e")
        ax.set_title("Attack Category Distribution", color="#c7d2fe", fontweight="bold")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_b:
        if state["subcat_col"]:
            st.markdown("**Subcategory Distribution**")
            sub_counts = df[state["subcat_col"]].value_counts().head(12)
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor("#0d0d1a")
            ax.set_facecolor("#1a1a3e")
            ax.barh(sub_counts.index, sub_counts.values, color="#818cf8", edgecolor="#0d0d1a", height=0.6)
            ax.set_xlabel("Count", color="#a0a0cc")
            ax.tick_params(colors="#c7d2fe")
            ax.spines[:].set_color("#2d2d6e")
            ax.set_title("Attack Subcategory Distribution", color="#c7d2fe", fontweight="bold")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.markdown("**Binary Label Distribution**")
            bin_counts = df[state["binary_col"]].value_counts()
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor("#0d0d1a")
            ax.set_facecolor("#1a1a3e")
            ax.bar(bin_counts.index.astype(str), bin_counts.values,
                   color=["#10b981", "#ef4444"][:len(bin_counts)], edgecolor="#0d0d1a")
            ax.set_ylabel("Count", color="#a0a0cc")
            ax.tick_params(colors="#c7d2fe")
            ax.spines[:].set_color("#2d2d6e")
            ax.set_title("Binary Label Distribution", color="#c7d2fe", fontweight="bold")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    st.markdown('<div class="section-header">📐 Numeric Feature Statistics</div>', unsafe_allow_html=True)
    if state["num_cols"]:
        stats_df = df[state["num_cols"]].describe().T.round(4)
        st.dataframe(
            stats_df.style.background_gradient(cmap="Blues", subset=["mean", "std"]),
            use_container_width=True, height=350
        )


# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; color:#555; font-size:0.8rem; margin-top:40px; padding:20px;
            border-top:1px solid #1e1e3f;">
  🛡️ IDS Dashboard &nbsp;•&nbsp; BoT-IoT Dataset &nbsp;•&nbsp; Built by Omar &nbsp;•&nbsp;
  Target Encoding · Min-Max Scaling · Naive Bayes
</div>
""", unsafe_allow_html=True)