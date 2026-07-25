# AI-Powered Behavioral Anomaly Detection for Cybersecurity

A behavioral anomaly-detection system for security operations. It learns a
per-entity model of normal behaviour for users, service accounts, and industrial
edge devices from access logs, then detects, classifies, and explains deviations on
a streaming, heavily imbalanced event feed. The work is framed around OT and
industrial-edge security (SCADA/HMI consoles, PLC gateways, historians, service
accounts, edge devices).

## Overview

Signature-based tools miss novel and slow-moving attacks, and analysts have a very
limited time budget for reviewing alerts. This system addresses both problems:

- It baselines each entity's normal behaviour and scores every event for how far it
  deviates, using an unsupervised detector that never sees ground-truth labels at
  inference time.
- It attributes each anomaly to a likely attack category, and escalates behaviour
  that matches no known pattern as "novel/unknown" rather than forcing a guess.
- It produces a concise, human-readable justification and a counterfactual for every
  alert, so an analyst can act on it quickly.
- It is evaluated with imbalance-appropriate metrics (PR-AUC, precision/recall at a
  realistic alert budget), never raw accuracy.

## Repository structure

```
generator/     Synthetic access-log generator (entity profiles, 8 behaviours)
models/        Feature layer, detectors, classifier, explanations, metrics
dashboard/     Streamlit analyst console
report/        Technical report, presentation content, walkthrough
figures/       Evaluation figures
data/          Generated data and model outputs
run_pipeline.py    Runs the full pipeline end to end
make_submission.py Builds the submission archive
requirements.txt
```

## Installation

Requires Python 3.11.

```
python -m pip install -r requirements.txt
```

Core dependencies: NumPy, pandas, pyarrow, Faker, scikit-learn, PyTorch, SHAP,
Streamlit, matplotlib.

## Reproducing the results

This source archive contains the code only. The generated datasets
(`data/*.parquet`) and trained model weights (`models/*.joblib`, `*.pt`) are not
included; they are regenerated exactly from the fixed seed by the command below.
Run it once before launching the dashboard. The full pipeline is driven by a single
fixed random seed and completes in roughly 75 seconds on commodity hardware.

```
python run_pipeline.py
```

This runs, in order: data generation, feature extraction, the Isolation Forest
baseline, the GRU sequence autoencoder and score fusion, the attack classifier, the
per-alert explanations, and the full metrics suite. Individual stages can also be
run directly, for example `python models/baseline.py`.

To launch the analyst console:

```
python -m streamlit run dashboard/app.py
```

## Method summary

### 1. Synthetic data (`generator/`)

Generates a time-ordered stream of access logs for 100 entities over 30 days
(approximately 47,000 events), matching the required schema. Each entity is built
from a fixed behavioural profile (habitual hours, home location, usual resources,
stable device, typical session length). Benign traffic includes deliberate noise so
that normal and attack behaviour overlap realistically.

Eight behaviours are represented: a normal baseline, six attacks (brute force,
impossible travel, credential stuffing, lateral movement, device spoofing, and
low-and-slow exfiltration), and one benign edge case (insider drift). Attacks are
injected at approximately 1.4 percent of events. Ground-truth labels are written to
a separate file, so the event log used for inference is label-blind by construction.

Two design decisions are documented in the code: failed authentications are encoded
as zero-duration `auth/login` events (the schema has no explicit success flag), and
each entity's location is fixed per day so that only the injected impossible-travel
pattern produces implausible geo-velocities.

### 2. Feature layer (`models/features.py`)

A single causal pass over the event stream maintains a small per-entity and per-IP
running state and emits nine features per event: geo-velocity, failed-authentication
count in a window, resource novelty, device mismatch, off-hours rarity,
session-duration z-score, command-sequence novelty, source-IP fan-in, and a
cold-start flag. Every feature is incrementally computable, so the same logic runs
on a live stream. New entities (fewer than five events) have their personal-history
features neutralised so they are not incorrectly flagged.

### 3. Detection (`models/baseline.py`, `models/sequence.py`)

Two unsupervised detectors are combined:

- An Isolation Forest over the nine features, which is strong on single-event
  anomalies.
- A GRU sequence autoencoder that reconstructs the window of an entity's recent
  events; high reconstruction error indicates behaviour that is surprising given
  recent history, which captures order-dependent attacks.

Each detector's score is normalised against the training-period distribution and the
two are fused by taking the elementwise maximum, so an attack detected by either
model remains flagged. Evaluation uses a temporal split (train on the first 70
percent of the timeline, test on the remainder).

### 4. Classification (`models/classifier.py`)

A three-stage hybrid names each anomaly: high-precision rules for well-defined
signatures (including recon and egress command tokens), a supervised RandomForest for
the remainder, and a novel/unknown category for anomalous events that match no rule
and cannot be confidently classified.

### 5. Explainability (`models/explain.py`)

For each alert, the top three contributing features are identified from the rule that
fired, SHAP values on the classifier, and per-feature reconstruction error from the
autoencoder. These are rendered into a deterministic, templated sentence with a
counterfactual. No large language model is used, so the output cannot hallucinate.

### 6. Metrics (`models/metrics.py`)

Reports PR-AUC, ROC-AUC, precision and recall at several alert budgets, per-attack
recall, a per-class confusion matrix, an isotonic calibration of the risk score, and
alert de-duplication. Accuracy is reported only to illustrate why it is misleading on
imbalanced data. Figures are written to `figures/`.

### 7. Dashboard (`dashboard/app.py`)

A Streamlit application with four views: a ranked alert queue with per-alert
explanations and analyst feedback, per-entity investigation timelines, a model
evaluation view with all metrics and figures, and a pipeline overview. Operational
views are label-blind; a clearly-labelled toggle can reveal ground truth for
demonstration only.

## Results

Evaluated on the temporal test split (approximately 1.3 percent anomalies):

| Metric | Value |
|--------|-------|
| PR-AUC (unified detector) | 0.42 (random baseline 0.01) |
| ROC-AUC | 0.97 |
| Recall at top-2% alert budget | 0.67 |
| Recall at top-5% alert budget | 0.87 |
| Attack-type naming accuracy (in budget) | 1.00 |
| Calibration Brier score (before/after) | 0.27 / 0.007 |

Per-attack recall at the 2 percent budget confirms coverage across categories rather
than only the most frequent attacks.

## Known limitations

- At a 2 percent budget the queue still contains a majority of benign events; this is
  inherent to a 1-2 percent base rate and is mitigated by de-duplication and
  calibrated ranking.
- Attack command signatures in the synthetic data are cleaner than in real logs,
  which contributes to the high naming accuracy.
- Geo-velocity produces occasional false alarms on legitimate fast travel, which is
  why each alert includes a counterfactual for analyst confirmation.

## Submission artifacts

- `report/report.md` - technical report (export to PDF for submission).
- `report/Honeywell_Q4_Idea_Submission.pptx` - presentation on the official template.
- `report/TECHNICAL_WALKTHROUGH.md` - detailed technical walkthrough.
- `make_submission.py` - builds `Honeywell_Q4_Submission.zip` with all deliverables.
