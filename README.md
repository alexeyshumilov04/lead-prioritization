# 🚗 Lead Prioritization — Predicting Successful Conversions 

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.x-yellow.svg)](https://catboost.ai/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-red.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.x-orange.svg)](https://shap.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-blue.svg)](https://scikit-learn.org/)
[![pandas](https://img.shields.io/badge/pandas-2.x-150458.svg)](https://pandas.pydata.org/)

An end-to-end machine learning pipeline that ranks incoming car-purchase leads by their probability of resulting in a successful transaction within **5 days** of assignment. The output is a calibrated `score ∈ [0, 1]` used to **prioritize leads within each day**.

---

## 🎯 Problem Statement

For every incoming lead (`lead_id`) we observe its assignment timestamp, a rich set of precomputed tabular aggregations (views, contacts, chats, previous outcomes, pricing, etc.), and a stream of behavioral events (`item_view`, `search`, `favorite`, `chat_open`, `call_click`).

**Goal:** at the moment of assignment, predict whether the lead will convert into a successful target action within the next 5 days.

**Evaluation metric:** *Daily Average Precision* — Average Precision computed separately for each assignment date, then averaged across days. The metric is **rank-based** and evaluated **day by day**, mirroring the real business use case where leads are prioritized within the same day.

---

## ✨ Key Features

- **Anti-leakage by design** — every event-based feature is built strictly from `event_ts < assignment_ts`; future events are dropped.
- **Five feature blocks** covering tabular conversions/trends, windowed event aggregates, price-fit + intent funnel, sequence shape of the last-5 events, and one-hot windowed expansions of the categorical fields `ctx_seq`, `src_slot`, and `event_type × ctx_seq`.
- **SHAP-guided forward feature selection** — correlation-based deduplication (`|r| > 0.95`) → SHAP ranking → forward selection on expanding-window CV, culminating at **65 features**.
- **Gradient-boosting ensemble** — CatBoost + XGBoost, blended by **within-day percentile ranks** (weight `0.45 / 0.55`).
- **Production-like validation** — expanding-window cross-validation (train on past, validate on the *next 3 days*), so hidden-test scoring is faithful to the training-time estimate.
- **Reproducible** — fixed seeds (`42, 7, 2026`), hardcoded Optuna-tuned hyperparameters, single output `submission.csv`.

---

## 📂 Project Structure

```text
LEAD-PRIORITIZATION/
│
├── data/                        # Input CSVs (not tracked)
│   ├── train.csv                # Training leads (13,694 × 119)
│   ├── test.csv                 # Test leads (4,306 × 118)
│   └── events.csv               # Behavioral event log (254,705 × 7)
│
├── solution/
│   └── solution.ipynb           # End-to-end notebook: EDA → FE → CV → submission
│
├── submissions/
│   └── submission.csv           # Generated predictions (lead_id, score)
│
├── requirements.txt             # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### 1. Installation

```bash
git clone https://github.com/alexeyshumilov04/LEAD-PRIORITIZATION.git
cd LEAD-PRIORITIZATION
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Data Setup

Place the three CSV files into `./data/`:

```text
data/
├── train.csv
├── test.csv
└── events.csv
```

Alternatively, point the notebook to a custom location via the `LEAD_CHALLENGE_DIR` environment variable:

```bash
export LEAD_CHALLENGE_DIR=/path/to/data
```

---

## 🛠️ Usage

Open `solution/solution.ipynb` and run all cells. The notebook will:

1. Load and validate the three CSVs, coerce timestamps.
2. Run EDA and the leakage audit.
3. Build the five feature blocks for `train` and `test`.
4. Encode categoricals, drop duplicates, compute SHAP rankings.
5. Run expanding-window forward-selection → pick `N* = 65` features.
6. Train CatBoost + XGBoost on 3 expanding folds, tune the blend weight.
7. Refit on the full training set with 3 seeds, blend, and write the result to `submission.csv`.

**Output:** `submission.csv` with two columns, `lead_id` and `score ∈ [0, 1]`.

---

## 🧠 Approach

### Anti-leakage audit
`~6.3 %` of events are dated *after* the lead's assignment and are discarded. Every aggregate is recomputed under the constraint `event_ts < assignment_ts`, so no future information contaminates the features.

### Feature engineering (5 blocks)

| Block | Function | Highlights |
|---|---|---|
| 1. Tabular conversions & trends | `add_combined_features` | `fav_per_view`, `contact_per_view`, `prev_pos_rate_{30d, 90d}`, momentum `*_trend`, price derivatives |
| 2. Windowed event aggregates | `build_event_features` | Counts per type over `1/3/7/14/30d`, recency, active days, gap statistics, night/weekend shares |
| 3. Price-fit + intent funnel | `build_tier1_event_features` | How close the lead's price is to viewed-item prices, funnel depth `item_view < favorite < chat < call` |
| 4. Sequence of last-5 events | `build_sequence_features` | Rank slope, type switches, bigram of the last 2 actions, tail price volatility |
| 5. Categorical expansions + cyclical time | `build_category_features` | Windowed counts and recencies for `ctx_seq` / `src_slot` / `event_type × ctx_seq`; `hour_sin/cos`, `weekday_sin/cos` |

Block 5 was the largest source of gain — the SHAP ranking shows `cat_ctx_set`, `cat_last_slot`, `slot_all__s24` and `slotrec__s2` in the top-20.

### Feature selection
1. Drop pairs with `|r| > 0.95` (keeping the higher-target-correlated one) → **423 → 506 candidates**.
2. Compute SHAP importance via `TreeSHAP` on an XGBoost model.
3. Forward-select on expanding-window CV → optimum at **`N* = 65` features**.

### Validation
**Expanding-window CV** — train on all days before `D`, validate on `[D, D+2]`, repeat as the window slides. Three folds:
- `2026-04-14 … 2026-04-16`
- `2026-04-17 … 2026-04-19`
- `2026-04-20 … 2026-04-22`

This mimics the deployment scenario (hidden test follows train) and avoids the optimism of random KFold.

### Ensemble
- **CatBoost** — native categorical handling, `PRAUC` eval metric.
- **XGBoost** — `enable_categorical=True` for native categoricals, `aucpr` eval metric.
- **Blend** — predictions are converted to **within-day percentile ranks** and mixed with weight `0.45` (CatBoost) / `0.55` (XGBoost), the weight chosen to maximize expanding-window Daily AP.

---

## 📊 Results

| Fold | CatBoost AP | XGBoost AP | Blend AP |
|---|---|---|---|
| `2026-04-14 … 2026-04-16` | 0.7076 | 0.7105 | **0.7156** |
| `2026-04-17 … 2026-04-19` | 0.7405 | 0.7416 | **0.7462** |
| `2026-04-20 … 2026-04-22` | 0.7327 | 0.7350 | **0.7392** |
| **Mean (expanding CV)** | — | — | **0.7337** |

For reference, the positive rate in train is `≈ 20.75 %`, so the model lifts the daily AP from a ~0.21 random-order baseline to ~0.73.

---

## 🔧 Tech Stack

- **Data:** `pandas`, `numpy`
- **Models:** `catboost`, `xgboost`
- **Interpretability:** `shap`
- **Utilities:** `scikit-learn`, `matplotlib`
- **Hyperparameter tuning:** Optuna (offline, results hardcoded into the notebook)

---

## ⚠️ Limitations & Notes

- **No LLM / external API calls, no hand-labeled test data, no per-`lead_id` hardcoding** — everything is derived from observable past behavior.
- **Hyperparameters are frozen** into the notebook (from an offline Optuna run on the same expanding CV) so the pipeline runs end-to-end in ~30 min on a laptop.
- The dataset is a **2026 snapshot**; the feature engineering logic is date-agnostic, but retraining on newer data is recommended for production use.

---

## 📝 Workflow Overview

1. **Load** `train.csv`, `test.csv`, `events.csv` and coerce timestamp columns.
2. **Audit leakage** — filter events to `event_ts < assignment_ts`.
3. **Feature engineering** — apply the 5 blocks to `train` and `test` symmetrically.
4. **Encode categoricals** — unified vocabulary across both splits, `"missing"` for gaps.
5. **Feature selection** — dedup by correlation, rank by SHAP, forward-select on expanding-window CV.
6. **Validate** — 3 expanding folds, CatBoost + XGBoost, tune blend weight on within-day ranks.
7. **Final training** — refit both models on all of `train` with 3 seeds, `1.15×` the best CV iterations.
8. **Predict** — blend per-day ranks, min-max normalize to `[0, 1]`, write `submission.csv`.
