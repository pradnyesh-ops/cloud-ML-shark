# 🦈 Marine Incident Fatality Prediction — Cloud ML Project
**National College of Ireland | MSc Cloud Computing | MSCCLOUD1JAN26I**  
**Module: Cloud Machine Learning | Weight: 50%**

---

## Research Question
> *Can we predict whether a shark attack or surf-zone incident will be fatal, based on demographic, environmental, and behavioural features?*

Three machine learning algorithms are each applied independently to the same three datasets (9 model runs total), then compared.

| Teammate | Algorithm | Notebook |
|---|---|---|
| pradnyesh  | Random Forest | `teammate1_random_forest.ipynb` |
| Teammate 2 | Support Vector Machine (SVM) | `teammate2_svm.ipynb` |
| Teammate 3 | XGBoost (Gradient Boosting) | `teammate3_xgboost.ipynb` |

---

## Datasets

| File | Records | Description |
|---|---|---|
| `attacks.csv` | 25 723 | Global Shark Attack File (GSAF) — raw |
| `geocoded-global-shark-attacks.csv` | 6 890 | GSAF with GPS coordinates |
| `Surf_Zone_Fatalities_2010-Present.csv` | 1 275 | NOAA US surf-zone fatalities |

---

## Running on Google Colab — Step-by-Step

### Step 1 — Upload Files to Google Drive

1. Go to [drive.google.com](https://drive.google.com) and sign in.
2. Create a folder: **My Drive → `CloudML`**
3. Upload all project files into that folder:
   ```
   CloudML/
   ├── attacks.csv
   ├── geocoded-global-shark-attacks.csv
   ├── Surf_Zone_Fatalities_2010-Present.csv
   ├── teammate1_random_forest.ipynb
   ├── teammate2_svm.ipynb
   ├── teammate3_xgboost.ipynb
   └── dashboard.ipynb
   ```

---

### Step 2 — Open a Notebook in Google Colab

**Option A — From Google Drive (recommended):**
1. In Google Drive, double-click `teammate1_random_forest.ipynb`.
2. Click **"Open with → Google Colaboratory"**.  
   *(If Colab is not listed: click "Connect more apps" and install it once.)*

**Option B — Direct Colab URL:**
1. Go to [colab.research.google.com](https://colab.research.google.com).
2. Click **File → Upload notebook** and select the `.ipynb` file.
3. Move the uploaded notebook to `My Drive/CloudML/` if it isn't already there.

---

### Step 3 — Connect to Runtime

1. In the top-right corner, click **"Connect"** (or **"Connect to a hosted runtime"**).
2. Wait for the green tick — you are now connected to a free GPU/CPU instance.
3. *(Optional)* For faster training: **Runtime → Change runtime type → GPU**.

---

### Step 4 — Run Teammate 1 (Random Forest)

Open `teammate1_random_forest.ipynb` and run cells **top to bottom**:

| Cell # | What it does |
|---|---|
| 1 | Markdown — title |
| 2 | Markdown — imports heading |
| **3** | **Installs & imports all libraries + sets mako colour palette** |
| **4** | **CSS styling cell** |
| **5** | **Displays mako palette swatch** |
| **6** | **Google Drive mount + sets `DATA_DIR = /content/drive/MyDrive/CloudML/`** ← Drive prompt appears here |
| **7** | **Section 2: Combined Dataset Model — load shared train/test split** |
| **8** | **Build & train Random Forest pipeline on combined data** |
| **9** | **Visualisations: confusion matrix, feature importance, ROC/PR curves** |
| **10** | **Learning curve with 5-fold stratified CV** |
| **11** | **Test data statistics & inference results** ← NEW |
| **12** | **Markdown — conclusions** |

> **When cell 6 runs**, a Google Drive authorisation prompt will appear.  
> Click **"Connect to Google Drive"** and allow access. The Drive mounts automatically.

**Estimated runtime:** ~8–12 minutes on a free Colab CPU.

---

### Step 5 — Run Teammate 2 (SVM)

Open `teammate2_svm.ipynb` and run all cells top to bottom.

> SVM is sub-sampled to 5 000 rows for DS1 and 4 000 rows for DS2  
> (SVM training is O(n²) — this is documented in the notebook).

**Estimated runtime:** ~5–10 minutes.

---

### Step 6 — Run Teammate 3 (XGBoost)

Open `teammate3_xgboost.ipynb` and run all cells top to bottom.

> Cell 3 auto-installs `xgboost` if it is not already available.

**Estimated runtime:** ~6–10 minutes.

---

### Step 7 — Run the Dashboard

> **Prerequisite:** Steps 4–6 must be complete so that the model `.pkl` files  
> exist at `My Drive/CloudML/models/`.

Open `dashboard.ipynb` and run all cells top to bottom.

| Cell # | What it does |
|---|---|
| 1 | Markdown — title |
| 2 | Install `gradio` and `folium` |
| 3 | Colab + Drive setup |
| 4 | Load all 9 trained models from Drive |
| 5 | Build label encoders + feature builder |
| 6 | Folium map builder |
| 7 | Prediction function |
| **8** | **Launch Gradio dashboard** ← public URL printed here |
| 9 | Batch inference demo |

**When cell 8 runs**, Colab will print two URLs:
```
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://xxxx.gradio.live
```
Share the **public URL** — it remains live for ~72 hours.

---

### Colab Session Tips

| Situation | Solution |
|---|---|
| Session disconnects | Re-run from **cell 6** (Drive mount) downwards |
| "Runtime disconnected" warning | Click **Reconnect** in the top-right |
| Out of RAM | Runtime → **Factory reset runtime**, then re-run |
| Want GPU for XGBoost | Runtime → **Change runtime type → T4 GPU** |
| Gradio URL expired | Re-run cell 8 in `dashboard.ipynb` |

---

## Test Data & Inference Strategy

Each training notebook includes a comprehensive **Test Data Statistics & Inference** section (section 2.1) that:

1. **Test Set Overview:**
   - Total samples, feature dimensions
   - Class distribution (Non-Fatal vs Fatal counts & percentages)

2. **Feature Statistics (on test set):**
   - Mean, Std, Min, Max, Missing count for each feature

3. **Inference Results:**
   - Prediction distribution (how many predicted as Fatal vs Non-Fatal)
   - Prediction probability statistics (mean, std, min, max of fatal probability)

4. **Confidence Distribution:**
   - Mean & median prediction confidence
   - High-confidence predictions (>0.8) vs medium-confidence (0.5–0.8)

5. **Sample Predictions:**
   - Table of first 20 & last 10 test samples showing:
     - Actual label (0=Non-Fatal, 1=Fatal)
     - Model prediction (0 or 1)
     - Probability of fatal (0.0–1.0)
     - Confidence level

6. **Confusion Breakdown:**
   - True Positives, True Negatives, False Positives, False Negatives
   - Sensitivity (Recall for Fatal class)
   - Specificity (Recall for Non-Fatal class)

This aligns with cloud ML best practices: **train/test separation**, **stratified evaluation**, and **interpretable inference metrics**.

---

## Project File Structure

```
CloudML/
│
├── 📓 data_preparation.ipynb          ← Step 1 — combine datasets, EDA, split
├── 📓 teammate1_random_forest.ipynb   ← Step 2 — Random Forest on combined data
├── 📓 teammate2_svm.ipynb             ← Step 3 — SVM on combined data
├── 📓 teammate3_xgboost.ipynb         ← Step 4 — XGBoost on combined data
├── 📓 dashboard.ipynb                 ← Step 5 — Client-facing Gradio + Folium app
│
├── 📄 attacks.csv                     ← Raw Dataset 1 (25 723 rows)
├── 📄 geocoded-global-shark-attacks.csv ← Raw Dataset 2 (6 890 rows, with GPS)
├── 📄 Surf_Zone_Fatalities_2010-Present.csv ← Raw Dataset 3 (1 275 rows)
│
├── 📄 combined_dataset.csv            ← Created by data_preparation.ipynb
├── 📄 combined_train.csv              ← 80% stratified split (shared across teams)
├── 📄 combined_test.csv               ← 20% stratified split (shared across teams)
├── 📄 activity_encoding.csv           ← Label encoder reference
├── 📄 inc_type_encoding.csv           ← Label encoder reference
│
├── 📄 README.md                       ← This file
│
└── models/                            ← Created automatically after training
    ├── random_forest_combined.pkl     ← Main combined model (RF)
    ├── svm_combined.pkl               ← Main combined model (SVM)
    └── xgboost_combined.pkl           ← Main combined model (XGBoost)
```

---

## Python Dependencies

All libraries are pre-installed in Colab or auto-installed by the notebooks.

| Library | Version | Purpose |
|---|---|---|
| `pandas` | ≥ 1.5 | Data manipulation |
| `numpy` | ≥ 1.23 | Numerical computing |
| `scikit-learn` | ≥ 1.2 | RF, SVM, metrics, pipelines |
| `xgboost` | ≥ 1.7 | Gradient boosting (auto-installed in NB3) |
| `matplotlib` | ≥ 3.6 | Static plotting |
| `seaborn` | ≥ 0.12 | Statistical visualisations (mako palette) |
| `gradio` | ≥ 4.0 | Web dashboard (auto-installed in dashboard) |
| `folium` | ≥ 0.14 | Interactive map (auto-installed in dashboard) |
| `joblib` | ≥ 1.2 | Model serialisation |

---

## What Each Notebook Produces

### Training Notebooks (1–3)
- **Visualisations:** Confusion matrices, feature importances, learning curves, ROC curves, Precision-Recall curves — all saved as `.png` at 400 DPI.
- **Metrics:** Accuracy, Precision, Recall, F1, ROC-AUC, training time, inference latency (µs/sample), throughput (samples/sec).
- **Baseline comparison:** `DummyClassifier` (stratified + most_frequent) vs trained model.
- **Test Data Statistics:** Test set size, class distribution, feature statistics (mean, std, min, max, missing values).
- **Inference Results:** Prediction distribution, prediction probabilities, confidence statistics, prediction summary table (first 20 and last 10 samples), confusion breakdown (TP/TN/FP/FN), sensitivity, specificity.
- **Saved models:** 1 × `.pkl` file per notebook (combined model) → `models/` folder on Drive.

### Dashboard (`dashboard.ipynb`)
- **Gradio interface** with input form (Activity, Age, Country, Attack Type, Gender).
- **Side-by-side predictions** from all 3 algorithms with confidence level.
- **Folium interactive map** showing historical attacks in the selected country (red = fatal, blue = non-fatal, coloured ring = model prediction zone).
- **Batch inference service** — processes the full geocoded CSV and saves predictions back to Drive.
- **Public URL** shared via `demo.launch(share=True)`.

---

## Notebook Execution Order (Required)

```
data_preparation.ipynb          ← MUST run first
        ↓  (saves combined_train.csv + combined_test.csv)
teammate1_random_forest.ipynb   ← loads shared split, trains RF
        ↓
teammate2_svm.ipynb             ← loads same split, trains SVM
        ↓
teammate3_xgboost.ipynb         ← loads same split, trains XGBoost
        ↓
dashboard.ipynb                 ← loads all 3 combined models
```

> All three teammate notebooks train and test on the **same** `combined_train.csv` / `combined_test.csv` to ensure a **fair algorithm comparison** (same data, different models).

---

## Cloud Technologies Used (LO4)

| Technology | Role |
|---|---|
| **Google Colab** | Cloud compute — runs all training and inference |
| **Google Drive** | Cloud storage — datasets, trained models, output plots |
| **Gradio** (`share=True`) | Cloud-served web endpoint — client-facing UI |
| **Folium / Leaflet.js** | Browser-side interactive map rendering |

---

## Troubleshooting

**`FileNotFoundError: attacks.csv`**  
→ Check that the CSV is in `My Drive/CloudML/` (not a subfolder).  
→ Verify the Drive mount succeeded (cell 6 should print a path starting with `/content/drive`).

**`ModuleNotFoundError: xgboost`**  
→ Run `!pip install xgboost` in a new cell, then restart the runtime and re-run from cell 1.

**`ModuleNotFoundError: gradio`**  
→ Cell 2 of `dashboard.ipynb` installs it automatically — wait for it to finish.

**Gradio shows `OSError: [Errno 98] Address already in use`**  
→ Run `import gradio as gr; gr.close_all()` in a new cell, then re-run the dashboard cell.

**Drive mount prompt does not appear**  
→ Runtime → **Factory reset runtime**, then re-run from cell 1.

---

