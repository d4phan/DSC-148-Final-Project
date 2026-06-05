# DSC 148 Final Project — Predicting Song Hits from Audio Characteristics

**Research question:** What audio characteristics make a song likely to become a hit?

We frame this as a binary classification problem on the Spotify 1M Tracks dataset:
a track is a **hit** if its `popularity` is in the top 10% of all tracks. We compare
four models — a majority-class baseline, logistic regression, random forest, and
XGBoost — and ship an interactive demo where anyone can set a song's audio features
and get a live prediction.

This repo has two runnable pieces:
1. **`song_hit_prediction.ipynb`** — the full analysis notebook (EDA, modeling, results).
2. **`hit_predictor.py`** — a Streamlit web app demo.

---

## Requirements

- Python **3.11 or 3.12** recommended.
- A Kaggle account (the dataset downloads automatically via `kagglehub` on first run).

Install all dependencies once, from the project folder:

```bash
pip install -r requirements.txt
```

This installs: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`,
`kagglehub`, `streamlit`, and `jupyter`.

---

## Part 1 — Running the Notebook

The notebook covers how we utilized the data to analyze, train, and come up with our solution to our research question.

### Steps

1. From the project folder, launch Jupyter:

   ```bash
   jupyter notebook
   ```

   (or `jupyter lab` if you prefer). Your browser opens a file list.

2. Click **`song_hit_prediction.ipynb`** to open it.

3. Run every cell top to bottom: menu **Cell -> Run All** (or **Run -> Run All Cells**
   in Jupyter Lab).

### What happens

- The first code cell under *1. Dataset* will download the Spotify data via `kagglehub`. \
  This may take some time so be patient.
- All figures and tables are written to a **`figures/`** folder created next to the
  notebook.
- Every metric is also saved to **`figures/metrics.json`** for easy copy-paste into
  the report.

### Outputs you'll get in `figures/`

```
01_popularity_dist.png        popularity histogram + hit threshold
02_corr_heatmap.png           feature correlation matrix
03_hit_vs_nonhit.png          per-feature distributions, hits vs non-hits
04_popularity_by_year.png     popularity trend over release years
05_model_comparison.png       accuracy / F1 / ROC-AUC bar chart
06_roc_curves.png             ROC curves
07_pr_curves.png              precision-recall curves
08_confusion_best.png         confusion matrix of the best model
09_feature_importance.png     RF vs XGBoost feature importances
table_model_comparison.csv    all metrics, all models
table_feature_importance.csv  importance scores
table_feature_means.csv       mean feature value by hit status
metrics.json                  all metrics, machine-readable
```

### Notebook structure 
`1. Dataset` · `2. Predictive Task` · `3. Model` · `4. Literature` · `5. Results`.

---

## Part 2 — Running the Demo "hit_predictor.py"

`hit_predictor.py` is a Streamlit app. It will load the dataset, train the XGBoost model on
launch, and show the likelihood of a song being a "hit" using 14 adjustable sliders.
Drag a slider and the hit prediction updates live.

### Steps

1. From the project folder, run:

   ```bash
   streamlit run hit_predictor.py
   ```

2. A browser tab opens automatically at **`http://localhost:8501`**.

3. Adjust the sliders. The app shows the probability of a song being a "hit" ot "not a hit",
   and a chart of the most important features.

### If `streamlit` is "not recognized"

This happens when Streamlit's install folder isn't on your system PATH. Use the
module form instead — it always works:

```bash
python -m streamlit run hit_predictor.py
```

On Windows, if `python` points to the wrong version, use the full path, e.g.:

```bash
C:\python311\python.exe -m streamlit run hit_predictor.py
```

### Notes

- **First launch takes ~30-60 seconds** while the model trains. After that, every
  slider change is instant (the trained model is cached).
- The demo retrains on launch, so no saved model file is needed. If you'd rather make
  startup instant, save the model from the notebook with
  `model.save_model("model.json")` and load it in `hit_predictor.py` instead of retraining.
- The config at the top of `hit_predictor.py` (`HIT_PERCENTILE`, `AUDIO_FEATURES`) matches the
  notebook exactly, so the demo and the reported results stay consistent.
- To stop the app, press **Ctrl+C** in the terminal running it.

---

## Troubleshooting

**`NameError` or `ModuleNotFoundError` for a library** — a dependency didn't install
or you're on a Python version without a compatible build (e.g. very new releases).
Re-run `pip install -r requirements.txt`; if it persists, use Python 3.11 or 3.12.

**Kaggle authentication error on first download** — sign in via the link `kagglehub`
prints, or set up your `kaggle.json` API token, then rerun.

**Streamlit shows a stale version after editing** — fully stop the app (Ctrl+C) and
relaunch rather than just refreshing the browser; Streamlit caches the trained model
in memory across reruns.

**IMPORTANT** - make sure to have the necessary libraries and imports to ensure that both the 
notebook and demo function as intended
