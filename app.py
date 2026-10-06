import io
import json
import math
import time
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from churn_utils import predict_frame, risk_level

BASE = Path(__file__).parent

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

RISK_COLORS = {"High": "#D64545", "Medium": "#E0A100", "Low": "#2E9E6B"}
MAX_STYLED_ROWS = 2000  # skip Styler above this many rows (Styler is slow on big tables)

# NOTE: the Google Fonts @import was removed. It blocked the first render.
# System fonts load instantly.
st.markdown(
    """
<style>
html, body, [class*="css"], .stApp { font-family: system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif; }
.block-container { padding-top: 1.6rem; max-width: 1200px; }

.hero {
    background: linear-gradient(120deg, #1B2A41 0%, #2B3F66 60%, #3D5AFE 130%);
    color: #fff; border-radius: 20px; padding: 28px 34px; margin-bottom: 18px;
    display: flex; justify-content: space-between; align-items: center; gap: 20px; flex-wrap: wrap;
}
.hero h1 { color: #fff !important; font-size: 2rem; font-weight: 800; margin: 0; padding: 0; letter-spacing: -0.02em; }
.hero p { color: #D5DDF0 !important; margin: 6px 0 0 0; font-size: 0.98rem; }
.chips { display: flex; gap: 10px; flex-wrap: wrap; }
.chip { background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.25);
        border-radius: 12px; padding: 8px 14px; font-size: 0.85rem; color: #fff; }
.chip b { display: block; font-size: 1.05rem; font-weight: 700; }

.stTabs [data-baseweb="tab-list"] { gap: 6px; background: rgba(128,128,128,0.14); padding: 5px; border-radius: 14px; width: fit-content; }
.stTabs [data-baseweb="tab"] { border-radius: 10px; padding: 8px 20px; font-weight: 600; height: auto; }
.stTabs [aria-selected="true"] { background: #3D5AFE; box-shadow: 0 2px 8px rgba(61,90,254,0.35); }
.stTabs [aria-selected="true"] p { color: #fff !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

[data-testid="stForm"] { background: rgba(128,128,128,0.07); border: 1px solid rgba(128,128,128,0.28); border-radius: 18px; padding: 22px 24px; }
[data-testid="stMetric"] { background: rgba(128,128,128,0.08); border: 1px solid rgba(128,128,128,0.28); border-radius: 16px; padding: 14px 18px; }
[data-testid="stMetricValue"] { font-weight: 800; }
.stFormSubmitButton button, .stDownloadButton button {
    background: #3D5AFE; color: #fff; border: 0; border-radius: 12px; padding: 0.6rem 1.6rem; font-weight: 700;
}
.stFormSubmitButton button:hover, .stDownloadButton button:hover { background: #5470FF; color: #fff; }
[data-testid="stFileUploader"] section { background: rgba(128,128,128,0.07); border: 2px dashed rgba(128,128,128,0.45); border-radius: 16px; }
[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; border: 1px solid rgba(128,128,128,0.28); }

.result { background: rgba(128,128,128,0.08); border: 1px solid rgba(128,128,128,0.28); border-radius: 20px; padding: 22px 26px; margin-top: 16px;
          display: flex; align-items: center; gap: 28px; flex-wrap: wrap; }
.result .gauge { flex: 0 0 240px; text-align: center; color: inherit; }
.result .info { flex: 1 1 280px; }
.badge { display: inline-block; color: #fff; font-weight: 700; padding: 5px 14px; border-radius: 999px; font-size: 0.9rem; }
.result h3 { margin: 10px 0 6px 0; font-size: 1.35rem; }
.result p { margin: 0; opacity: 0.8; line-height: 1.55; }
.hint { opacity: 0.65; font-size: 0.88rem; margin-top: 8px; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_artifacts():
    t = time.time()
    bundle = joblib.load(BASE / "best_churn_model.pkl")
    meta = json.loads((BASE / "churn_app_meta.json").read_text())
    # Check this in: Manage app -> Logs
    print(f"[timing] model load: {time.time() - t:.1f}s")
    return bundle, meta


# Show a spinner so the first visit isn't a blank page while the model unpickles.
with st.spinner("Loading model..."):
    try:
        bundle, meta = load_artifacts()
    except FileNotFoundError:
        st.error("best_churn_model.pkl / churn_app_meta.json not found. Run Steps 14 and 15 in the notebook first.")
        st.stop()

THRESHOLD = bundle["threshold"]
label = lambda c: c.replace("_", " ")


def gauge_svg(p: float, threshold: float, color: str) -> str:
    r, cx, cy = 80, 100, 100
    arc_len = math.pi * r
    p = min(max(p, 0.0), 1.0)
    th = math.pi * (1 - min(max(threshold, 0.0), 1.0))
    x1, y1 = cx + (r - 12) * math.cos(th), cy - (r - 12) * math.sin(th)
    x2, y2 = cx + (r + 12) * math.cos(th), cy - (r + 12) * math.sin(th)
    return f"""
    <svg viewBox="0 0 200 125" width="240">
      <path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="rgba(128,128,128,0.28)" stroke-width="16" stroke-linecap="round"/>
      <path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="{color}" stroke-width="16" stroke-linecap="round"
            stroke-dasharray="{arc_len * p:.1f} {arc_len:.1f}"/>
      <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="currentColor" stroke-width="3" stroke-linecap="round"/>
      <text x="100" y="92" text-anchor="middle" font-size="30" font-weight="800" fill="currentColor"
            font-family="system-ui, sans-serif">{p:.0%}</text>
      <text x="100" y="112" text-anchor="middle" font-size="10" fill="currentColor" fill-opacity="0.65"
            font-family="system-ui, sans-serif">churn probability</text>
    </svg>"""


@st.cache_data(show_spinner="Scoring customers...", max_entries=3)
def score_file(file_bytes: bytes, filename: str):
    """Read + validate + score an uploaded file. Cached, so widget clicks
    (e.g. the download button) don't re-run the whole pipeline.

    Returns (out_df, missing_cols, n_missing, csv_bytes). out_df is None if columns are missing.
    """
    buf = io.BytesIO(file_bytes)
    df_in = pd.read_csv(buf) if filename.lower().endswith(".csv") else pd.read_excel(buf)

    need = list(meta["raw_num"]) + list(meta["raw_cat"])
    missing_cols = [c for c in need if c not in df_in.columns]
    no_date = "Signup_Date" not in df_in.columns and "Days_Since_Signup" not in df_in.columns
    if missing_cols or no_date:
        return None, missing_cols + (["Signup_Date"] if no_date else []), 0, None

    n_missing = int(df_in[need].isna().sum().sum())
    df_work = df_in.copy()
    for c, s in meta["raw_num"].items():
        df_work[c] = df_work[c].fillna(s["default"])
    for c, s in meta["raw_cat"].items():
        df_work[c] = df_work[c].fillna(s["default"])
    if "Signup_Date" not in df_work.columns:
        df_work["Days_Since_Signup"] = df_work["Days_Since_Signup"].fillna(meta["days_since_signup"]["default"])

    proba, pred = predict_frame(bundle, meta, df_work)
    out = df_in.copy()
    out["Churn_Probability"] = proba.round(4)
    out["Churn_Prediction"] = pred
    out["Risk_Level"] = [risk_level(p, THRESHOLD) for p in proba]
    out = out.sort_values("Churn_Probability", ascending=False)
    return out, [], n_missing, out.to_csv(index=False).encode("utf-8")


st.markdown(
    f"""
<div class="hero">
  <div>
    <h1>Customer Churn Predictor</h1>
    <p>Find customers likely to leave, so your team can reach out before they do.</p>
  </div>
  <div class="chips">
    <div class="chip">Model<b>{bundle['model_name']}</b></div>
    <div class="chip">Decision threshold<b>{THRESHOLD:.3f}</b></div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

tab_single, tab_batch, tab_model = st.tabs(["Single customer", "Batch (CSV / Excel)", "About model"])

with tab_single:
    with st.form("single"):
        st.markdown("#### Customer details")
        values = {}
        cols = st.columns(3)
        i = 0

        d = meta["days_since_signup"]
        values["Days_Since_Signup"] = cols[i % 3].number_input(
            "Days Since Signup", min_value=0, value=int(d["default"]), step=1,
            help=f"Training data range: 0 - {d['max']}")
        i += 1

        for c, s in meta["raw_num"].items():
            with cols[i % 3]:
                if s["binary"]:
                    values[c] = int(st.selectbox(label(c), [0, 1], index=int(s["default"]),
                                                 format_func=lambda v: "Yes" if v == 1 else "No"))
                elif s["is_int"] and 0 < (s["max"] - s["min"]) <= 20:
                    values[c] = st.slider(label(c), int(s["min"]), int(s["max"]), int(s["default"]))
                elif s["is_int"]:
                    values[c] = st.number_input(label(c), min_value=int(min(s["min"], 0)),
                                                value=int(s["default"]), step=1,
                                                help=f"Training range: {s['min']:g} - {s['max']:g}")
                else:
                    values[c] = st.number_input(label(c), min_value=float(min(s["min"], 0)),
                                                value=float(s["default"]), step=0.1,
                                                help=f"Training range: {s['min']:g} - {s['max']:g}")
            i += 1

        for c, s in meta["raw_cat"].items():
            with cols[i % 3]:
                opts = s["options"]
                values[c] = st.selectbox(label(c), opts, index=opts.index(s["default"]))
            i += 1

        submitted = st.form_submit_button("Predict churn", type="primary")

    if submitted:
        proba, pred = predict_frame(bundle, meta, pd.DataFrame([values]))
        p = float(proba[0])
        level = risk_level(p, THRESHOLD)
        color = RISK_COLORS.get(level, "#3D5AFE")

        if pred[0] == 1:
            headline = "Likely to churn"
            advice = "High churn risk. A good candidate for a retention offer or a call."
        elif level == "Medium":
            headline = "Likely to stay, but watch closely"
            advice = "Medium risk. Consider adding this customer to the watch list."
        else:
            headline = "Likely to stay"
            advice = "Low churn risk. No action needed right now."

        st.markdown(
            f"""
<div class="result">
  <div class="gauge">{gauge_svg(p, THRESHOLD, color)}
    <div class="hint">The dark tick marks the decision threshold ({THRESHOLD:.2f}).</div>
  </div>
  <div class="info">
    <span class="badge" style="background:{color}">{level} risk</span>
    <h3>{headline}</h3>
    <p>{advice}</p>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Churn probability", f"{p:.1%}")
        m2.metric("Prediction", "Will churn" if pred[0] == 1 else "Will stay")
        m3.metric("Risk level", level)

with tab_batch:
    st.markdown("#### Score many customers at once")
    st.write("Upload a file with the same raw columns as the original dataset "
             "(`Signup_Date` or `Days_Since_Signup` is required; `Customer_ID` is optional).")
    up = st.file_uploader("Upload file", type=["csv", "xlsx"])

    if up is not None:
        out, errs, n_missing, csv_bytes = score_file(up.getvalue(), up.name)

        if out is None:
            st.error(f"Missing columns: {errs}")
        else:
            if n_missing:
                st.warning(f"{n_missing} missing values were filled with the training median/mode.")

            pred = out["Churn_Prediction"]

            k1, k2, k3 = st.columns(3)
            k1.metric("Customers", f"{len(out):,}")
            k2.metric("Predicted to churn", f"{int(pred.sum()):,}")
            k3.metric("Churn share", f"{pred.mean():.1%}")

            counts = out["Risk_Level"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0).astype(int)
            st.markdown("##### Customers by risk level")
            st.bar_chart(counts, color="#3D5AFE", height=200)

            def tint(v):
                c = RISK_COLORS.get(v)
                return f"background-color: {c}22; color: {c}; font-weight: 700" if c else ""

            shown = out
            if len(out) <= MAX_STYLED_ROWS:
                try:
                    style_fn = getattr(out.style, "map", None) or out.style.applymap
                    shown = style_fn(tint, subset=["Risk_Level"]).format({"Churn_Probability": "{:.1%}"})
                except Exception:
                    shown = out
            else:
                st.caption(f"Colour styling skipped for tables above {MAX_STYLED_ROWS:,} rows (keeps the page fast).")

            st.dataframe(shown, use_container_width=True, height=420)
            st.download_button("Download predictions (CSV)", csv_bytes,
                               file_name="churn_predictions.csv", mime="text/csv")

with tab_model:
    st.markdown("#### Test-set performance")
    metrics = meta.get("metrics", {})
    if metrics:
        cols = st.columns(len(metrics))
        for col, (k, v) in zip(cols, metrics.items()):
            col.metric(k.replace("_", " ").upper(), f"{v:.3f}")
    imp = meta.get("top_features")
    if imp:
        st.markdown("#### Top feature importances")
        st.bar_chart(pd.DataFrame(imp, columns=["feature", "importance"]).set_index("feature"),
                     color="#3D5AFE")
    st.info("The threshold was selected during training to balance Precision and Recall (max F1). "
            "Risk level: High = at or above the threshold, Medium = at or above 50% of the threshold, "
            "Low = below that.")
