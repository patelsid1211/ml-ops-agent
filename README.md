# ml-ops-agent
End-to-end ML pipeline with an LLM-powered agentic monitoring system
# ML-Ops Agent

An end-to-end ML pipeline combined with an LLM-powered autonomous agent that monitors model performance and takes action without human intervention.

---

## What it does

- Ingests and validates raw data, engineers features, and stores them in a versioned feature store
- Trains and evaluates ML models with full metric logging via MLflow
- Runs an LLM agent (Anthropic API + LangGraph) that monitors model performance and autonomously decides whether to query data, trigger retraining, or send alerts
- Exposes the agent via a FastAPI webhook endpoint for production-style triggering

---

## Architecture

```
Raw Data
   └── Ingestion + Validation
          └── Feature Engineering + Feature Store
                 └── Training Pipeline (scikit-learn / Spark ML)
                        └── Evaluation Harness (MLflow tracking)
                               └── LLM Agent (Anthropic API + LangGraph)
                                      ├── query_feature_store tool
                                      ├── run_evaluation tool
                                      ├── trigger_retrain tool
                                      └── send_alert tool
```

---

## Stack

| Layer | Tool |
|---|---|
| Data pipeline | Python, pandas, PySpark |
| ML training | scikit-learn, Spark ML |
| Experiment tracking | MLflow |
| Agent framework | LangGraph, Anthropic API |
| Serving | FastAPI |
| Dataset | UCI Air Quality (public) |

---

## Project Structure

```
ml-ops-agent/
├── pipeline/
│   ├── ingest.py           # Load and validate raw data
│   ├── features.py         # Feature engineering
│   └── feature_store.py    # Save and load versioned features
├── training/
│   ├── train.py            # Train model, log with MLflow
│   └── evaluate.py         # Metrics, confusion matrix, precision-recall
├── agent/
│   ├── orchestrator.py     # LLM agent logic
│   └── tools.py            # Tool definitions
├── api/
│   └── main.py             # FastAPI webhook endpoint
├── data/
│   └── sample.csv          # Sample dataset
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/your-username/ml-ops-agent.git
cd ml-ops-agent
pip install -r requirements.txt
```

---

## Status

actively building — pipeline and training modules complete, agent layer in progress.