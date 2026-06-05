"""
DSC 148 Project — Working Demo
Streamlit app: predicts whether a song is a "hit" (top 10% popularity)
from its audio characteristics. Retrains XGBoost on launch (cached).
 
Run with: streamlit run hit_predictor.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, f1_score

HIT_PERCENTILE = 90
AUDIO_FEATURES = ["danceability", "energy", "key", "loudness", "mode",
                  "speechiness", "acousticness", "instrumentalness",
                  "liveness", "valence", "tempo", "duration_ms",
                  "time_signature"]
FEATURES = AUDIO_FEATURES + ["year"]
RNG = 42
 
st.set_page_config(page_title="Song Hit Predictor", page_icon="🎵", layout="wide")

@st.cache_resource(show_spinner="Loading data and training model…")
def load_and_train():
    import kagglehub
    path = kagglehub.dataset_download("amitanshjoshi/spotify-1million-tracks")
    df = pd.read_csv(os.path.join(path, "spotify_data.csv"))
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df = df.drop_duplicates(subset=["track_id"]).reset_index(drop=True)
    df = df.dropna(subset=AUDIO_FEATURES + ["popularity"]).reset_index(drop=True)
 
    thr = np.percentile(df["popularity"], HIT_PERCENTILE)
    df["is_hit"] = (df["popularity"] >= thr).astype(int)
 
    X, y = df[FEATURES], df["is_hit"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RNG)
 
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    model = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos,
        eval_metric="logloss", tree_method="hist", n_jobs=-1, random_state=RNG)
    model.fit(X_train, y_train)
 
    p_tr = model.predict_proba(X_train)[:, 1]
    prec, rec, ths = precision_recall_curve(y_train, p_tr)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    best_t = float(ths[np.argmax(f1[:-1])])
 
    test_f1 = f1_score(y_test, (model.predict_proba(X_test)[:, 1] >= best_t).astype(int))
 
    stats = df[FEATURES].describe().T
    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    return model, best_t, thr, stats, importances, test_f1, len(df)
 
 
model, best_t, pop_thr, stats, importances, test_f1, n_rows = load_and_train()

st.title("🎵 Song Hit Predictor")
st.markdown(
    f"Predicts whether a track is a **hit** (top {100-HIT_PERCENTILE}% popularity, "
    f"i.e. popularity ≥ **{pop_thr:.0f}**) from its audio characteristics. "
    f"Model: XGBoost trained on **{n_rows:,}** tracks · test F1 = **{test_f1:.3f}**."
)

st.subheader("Set the audio characteristics")

RATIO_FEATS = {"danceability", "energy", "speechiness", "acousticness",
               "instrumentalness", "liveness", "valence"}
 
col1, col2, col3 = st.columns(3)
user_input = {}

SPOTIFY_GREEN = "#1DB954"

st.markdown(f"""
<style>
/* the filled portion of the slider track */
div[data-baseweb="slider"] div[role="progressbar"] {{
    background: {SPOTIFY_GREEN} !important;
}}
/* the draggable thumb */
div[data-baseweb="slider"] div[role="slider"] {{
    background: {SPOTIFY_GREEN} !important;
    border-color: {SPOTIFY_GREEN} !important;
}}
/* the value label/pill above the thumb */
div[data-baseweb="slider"] [data-testid="stThumbValue"] {{
    color: {SPOTIFY_GREEN} !important;
}}
</style>
""", unsafe_allow_html=True)

slider_layout = [
    ("danceability", col1), ("energy", col1), ("valence", col1),
    ("acousticness", col1), ("instrumentalness", col1),
    ("speechiness", col2), ("liveness", col2), ("loudness", col2),
    ("tempo", col2), ("duration_ms", col2),
    ("year", col3), ("key", col3), ("mode", col3), ("time_signature", col3),
]
 
FEATURE_HELP = {
    "danceability":     "Higher = more suitable for dancing (steady tempo, strong beat). Lower = harder to dance to.",
    "energy":           "Higher = fast, loud, intense (e.g. metal). Lower = calm, mellow (e.g. a soft ballad).",
    "valence":          "Higher = more positive/happy/cheerful. Lower = more sad, angry, or moody.",
    "acousticness":     "Higher = more likely acoustic (no electronics). Lower = more electronic/produced.",
    "instrumentalness": "Higher = more likely to have no vocals (instrumental). Lower = contains vocals.",
    "speechiness":      "Higher = more spoken words (rap, podcast-like). Lower = mostly sung/musical.",
    "liveness":         "Higher = more likely recorded live (audience present). Lower = studio recording.",
    "loudness":         "Overall loudness in decibels. Higher (closer to 0) = louder. Lower (more negative) = quieter.",
    "tempo":            "Speed of the track in BPM. Higher = faster. Lower = slower.",
    "duration_ms":      "Track length in milliseconds. Higher = longer song. Lower = shorter song.",
    "year":             "Release year. Higher = more recent. Lower = older.",
    "key":              "Pitch class the song is in (0=C, 1=C♯/D♭, … 11=B). Categorical — no high/low ranking.",
    "mode":             "Musical mode: 1 = major (often brighter), 0 = minor (often darker).",
    "time_signature":   "Beats per bar (e.g. 3 = waltz feel, 4 = most pop/rock). Categorical — no high/low ranking.",
}

for feat, col in slider_layout:
    lo = float(stats.loc[feat, "min"])
    hi = float(stats.loc[feat, "max"])
    med = float(stats.loc[feat, "50%"])
    with col:
        if feat in RATIO_FEATS:
            user_input[feat] = st.slider(feat, 0.0, 1.0, round(med, 3), 0.01)
        elif feat == "mode":
            user_input[feat] = st.selectbox("mode (0=minor, 1=major)", [0, 1], index=int(med))
        elif feat in ("key", "time_signature", "year"):
            user_input[feat] = st.slider(feat, int(lo), int(hi), int(med), 1)
        else:  
            user_input[feat] = st.slider(feat, round(lo, 1), round(hi, 1), round(med, 1))
        st.caption(FEATURE_HELP[feat])
 

st.subheader("Prediction")
x = pd.DataFrame([[user_input[f] for f in FEATURES]], columns=FEATURES)
prob = float(model.predict_proba(x)[:, 1][0])
is_hit = prob >= best_t
 
c1, c2 = st.columns([1, 2])
with c1:
    if is_hit:
        st.success("### 🔥 HIT")
    else:
        st.error("### ❄️ Not a hit")
    st.metric("Hit probability", f"{prob:.1%}")
    st.caption(f"Decision threshold: {best_t:.1%}")
with c2:
    st.progress(min(prob, 1.0))
    st.bar_chart(importances.head(8), horizontal=True)
    st.caption("Most important audio characteristics (model-wide)")
 
with st.expander("What we defined as a 'hit'"):
    st.write(
        f"The label 'hit' can have different meanings, so we derived one: a track is a **hit** if its "
        f"Spotify popularity (0–100) is in the top {100-HIT_PERCENTILE}% of the dataset "
        f"(≥ {pop_thr:.0f}). This is imbalanced (~{100-HIT_PERCENTILE}% positives), which "
        f"is why we tune the decision threshold to {best_t:.1%} instead of using 0.5."
    )