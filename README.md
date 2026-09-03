# Customer Intent Intelligence

Fine-grained customer-intent classification for banking support queries, built
on the [BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets)
dataset. The project compares lexical (TF-IDF) and semantic (sentence
embedding) representations across five classifiers, analyses where the best
model fails, and ships the winner behind a Streamlit app.

**Final model:** Sentence embeddings (`all-MiniLM-L6-v2`) + Linear SVM —
**92.76% test accuracy**, **0.9273 macro F1** across 77 intents.

---

## Dataset

| | |
|---|---|
| Source | PolyAI BANKING77 |
| Training queries | 10,003 |
| Test queries | 3,080 |
| Intents | 77 |
| Class balance | 187 max / 35 min per intent (5.34× ratio) |

The dataset ships with a predefined train/test split. The test set is used
**once**, for final reporting only.

---

## Approach

### Preprocessing

All preprocessing lives in [`src/preprocessing.py`](src/preprocessing.py) and is
imported by both the notebooks and the app, so training and serving cannot
drift apart.

1. Lowercase
2. Remove non-alphabetic characters (this also strips digits)
3. Normalize whitespace
4. Tokenize with NLTK, then rejoin

Two deliberate choices:

- **Stopwords are kept.** Intent frequently hinges on negation and function
  words — "I *cannot* use my card" and "I can use my card" are different
  intents. Removing them would destroy signal.
- **Lemmatization is not applied.** BANKING77 queries are short and already
  close to their base forms.

Step 4 is not redundant after step 2: NLTK splits `cannot` into `can not`,
which changes the unigram/bigram vocabulary.

### Representations

| Representation | Configuration | Dimensionality |
|---|---|---|
| TF-IDF | unigrams + bigrams, `min_df=2`, `max_df=0.95`, `sublinear_tf=True` | 10,243 features |
| Sentence embeddings | `all-MiniLM-L6-v2` | 384 dense dimensions |

### Model selection

Every model is tuned and selected on a stratified **validation split held out
of the training data**. The test set is never used for selection — it is
scored once, after the final model is chosen, so the reported figure is a
genuine held-out estimate.

---

## Results

Selection metric is validation macro F1; the remaining columns are held-out
test performance.

| Model | Validation Macro F1 | Test Accuracy | Test Macro F1 | Test Weighted F1 |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | 0.8735 | 0.8912 | 0.8915 | 0.8915 |
| TF-IDF + Linear SVM | 0.8820 | 0.8906 | 0.8906 | 0.8906 |
| Embeddings + Logistic Regression | 0.9028 | 0.9081 | 0.9079 | 0.9079 |
| **Embeddings + Linear SVM** | **0.9261** | **0.9276** | **0.9273** | **0.9273** |
| Embeddings + MLP | 0.8930 | 0.9068 | 0.9073 | 0.9073 |

Semantic embeddings beat the best lexical baseline by **3.7 macro F1 points**
(0.9273 vs 0.8906), which is the project's main finding: intent is carried by
meaning more than by surface wording.

Note that the two TF-IDF models swap order between validation and test. Their
gap is well within noise, which is exactly why selection is anchored to the
validation split rather than to whichever model happens to peak on test.

### Error analysis

223 of 3,080 test queries are misclassified. The errors are not random — they
concentrate in genuinely adjacent intent pairs:

| Actual | Predicted | Count |
|---|---|---|
| `card_arrival` | `card_delivery_estimate` | 6 |
| `card_payment_not_recognised` | `compromised_card` | 5 |
| `get_disposable_virtual_card` | `getting_virtual_card` | 5 |
| `pending_transfer` | `transfer_timing` | 3 |

Hardest intents by error rate: `declined_transfer` and
`card_payment_not_recognised` (25% each), then
`balance_not_updated_after_bank_transfer` and `pending_transfer` (22.5%).

These are pairs where the boundary is genuinely thin — "where is my card" vs
"when will my card arrive" is a distinction of phrasing, not of customer need.

Accuracy is essentially flat across query lengths (~93% for 1–50 words) and
drops only for queries above 50 words (85.7%), where the sample is very small.

---

## Project structure

```
├── app/app.py              Streamlit inference app
├── src/preprocessing.py    Canonical preprocessing (shared by notebooks + app)
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb   EDA, preprocessing, processed datasets
│   ├── 02_baseline_model.ipynb      TF-IDF + LR / Linear SVM, C tuning
│   ├── 03_semantic_models.ipynb     Embeddings + LR / SVM / MLP
│   ├── 04_error_analysis.ipynb      Confusion pairs, per-intent + length errors
│   └── 05_final_model.ipynb         Model selection, inference pipeline, config
├── data/
│   ├── raw/banking77/      Original train/test CSVs
│   └── processed/          Preprocessed datasets + cached embeddings
├── models/                 Trained models, vectorizer, final_config.pkl
└── reports/
    ├── results/            Per-model metrics and predictions
    └── error_analysis/     Confusion pairs, error rates, length analysis
```

---

## Setup

Requires Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The NLTK tokenizer data is downloaded automatically on the first run of
notebook 01, or manually:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

## Running

**Reproduce the pipeline** — run the notebooks in order (01 → 05). Each one
writes the artifacts the next one reads, so the order matters. Notebook 01
downloads BANKING77 on first run and uses the local copy thereafter.

```bash
jupyter lab
```

**Launch the app:**

```bash
streamlit run app/app.py
```

The app reads `models/final_config.pkl` to determine which model to load, so
re-running notebook 05 after retraining is enough to swap the deployed model —
no code change required.

---

## Inference behaviour

The app returns the predicted intent plus a calibrated sense of how much to
trust it, which differs by model family:

- **Probability models** (Logistic Regression, MLP) report the predicted
  probability and flag anything below **0.60** as low confidence.
- **Margin models** (Linear SVM, the current default) have no probability
  estimate, so they report the raw decision score and the **top-2 margin** —
  the gap between the best and second-best intent. A margin below **0.10** is
  flagged as ambiguous, with the runner-up intent shown alongside.

The margin flag is what makes the model safe to deploy against 77 closely
related classes: a vague query like "I need help with something" produces a
margin of 0.027 and is surfaced as uncertain rather than answered confidently.

Queries containing no ASCII letters (`12345`, `!!!`, or non-Latin script)
reduce to an empty string under preprocessing and are rejected before reaching
the model, rather than being answered with an arbitrary intent.
