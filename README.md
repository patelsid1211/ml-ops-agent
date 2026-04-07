# ML-Ops Agent

An end-to-end ML pipeline combined with an LLM-powered autonomous agent that monitors model performance and takes action without human intervention.

---

## What it does

- Ingests and validates raw air quality sensor data with schema checks and sentinel value handling
- Engineers 19 features including time-based and rolling average features, stored in a versioned feature store
- Trains a Random Forest regression model and logs all experiments with MLflow
- Evaluates model health using RMSE, MAE, and R² against defined thresholds
- Runs an LLM agent (Anthropic API) that autonomously decides whether to send an alert, trigger retraining, or report healthy status
- Exposes the full pipeline via a FastAPI REST API with `/health`, `/evaluate`, and `/monitor` endpoints

---

## Model Performance

| Metric | Value |
|---|---|
| RMSE | 0.3772 |
| MAE | 0.2489 |
| R² | 0.9271 |
| Training rows | 5,875 |
| Test rows | 1,469 |
| Top feature | PT08.S2(NMHC) — 88.7% importance |

---

## Architecture

```
Raw Data (UCI Air Quality)
   └── pipeline/ingest.py          — load, validate, clean
          └── pipeline/features.py — engineer 19 features
                 └── pipeline/feature_store.py — versioned storage
                        └── training/train.py  — Random Forest + MLflow
                               └── training/evaluate.py — RMSE, MAE, R²
                                      └── agent/tools.py — 4 action tools
                                             └── agent/orchestrator.py — LLM agent loop
                                                    └── api/main.py — FastAPI REST API
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Pipeline status and feature store version |
| GET | `/evaluate` | Run evaluation and return current metrics |
| POST | `/monitor` | Trigger full agent monitoring cycle |

---

## Stack

| Layer | Tool |
|---|---|
| Data pipeline | Python, pandas, PySpark |
| ML training | scikit-learn |
| Experiment tracking | MLflow |
| Agent framework | Anthropic API, LangGraph |
| Serving | FastAPI, Uvicorn |
| Dataset | UCI Air Quality (public) |

---

## Project Structure

```
ml-ops-agent/
├── pipeline/
│   ├── ingest.py            # Load, validate, and clean raw data
│   ├── features.py          # Feature engineering with time and rolling features
│   └── feature_store.py     # Save and load versioned feature sets
├── training/
│   ├── train.py             # Train Random Forest model, log with MLflow
│   └── evaluate.py          # Metrics, threshold checks, prediction plot
├── agent/
│   ├── tools.py             # query_feature_store, run_evaluation, trigger_retrain, send_alert
│   └── orchestrator.py      # Agentic loop with tool-calling and mock mode
├── api/
│   └── main.py              # FastAPI endpoints
├── data/
│   ├── sample.csv           # UCI Air Quality dataset
│   ├── feature_store/       # Versioned feature sets
│   ├── plots/               # Predicted vs actual plots
│   └── alerts.log.sample    # Sample agent alert output
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/your-username/ml-ops-agent.git
cd ml-ops-agent
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Add your Anthropic API key to a `.env` file:
```
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Running the pipeline

```bash
# Run full data pipeline and save to feature store
python -m pipeline.feature_store

# Train the model
python -m training.train

# Evaluate the model
python -m training.evaluate

# Run the agent monitoring cycle
python -m agent.orchestrator

# Start the API server
python -m uvicorn api.main:app --reload
```

---

## Agent decision logic

The agent evaluates model metrics against thresholds and autonomously selects one of three actions:

| Status | Condition | Action |
|---|---|---|
| healthy | RMSE < 0.5 and R² > 0.85 | log info alert |
| alert | R² borderline | send warning notification |
| retrain | RMSE > 0.5 or R² < 0.85 | trigger retraining + send alert |

---

## Status

Live — all modules complete and tested end to end.