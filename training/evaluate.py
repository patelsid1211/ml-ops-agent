import logging
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, required for worker threads
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

from pipeline.feature_store import load_features, get_latest_version

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "ml-ops-agent"
TEST_SIZE       = 0.2
RANDOM_STATE    = 42
PLOTS_DIR       = Path("data/plots")

# Thresholds for agent decision making
RMSE_THRESHOLD  = 0.5   # if RMSE exceeds this, agent should consider retraining
R2_THRESHOLD    = 0.85  # if R2 drops below this, agent should alert


def get_latest_run_id() -> str:
    """Fetch the most recent MLflow run ID for this experiment."""
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found. Run train.py first.")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    if not runs:
        raise ValueError("No runs found. Run train.py first.")
    run_id = runs[0].info.run_id
    logger.info(f"Latest MLflow run_id: {run_id}")
    return run_id


def load_model(run_id: str):
    """Load a trained model from MLflow by run ID."""
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    logger.info(f"Model loaded from MLflow (run_id={run_id})")
    return model


def compute_metrics(y_true, y_pred) -> dict:
    """Compute RMSE, MAE, and R2 metrics."""
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred))
    }
    logger.info(f"RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f} | R2: {metrics['r2']:.4f}")
    return metrics


def plot_predictions(y_true, y_pred, run_id: str) -> Path:
    """Save a predicted vs actual scatter plot."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = PLOTS_DIR / f"pred_vs_actual_{run_id[:8]}.png"

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, color="steelblue")
    ax.plot([y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()], "r--", linewidth=1.5, label="Perfect fit")
    ax.set_xlabel("Actual CO(GT)")
    ax.set_ylabel("Predicted CO(GT)")
    ax.set_title("Predicted vs Actual CO Levels")
    ax.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=100)
    plt.close()
    logger.info(f"Plot saved to {plot_path}")
    return plot_path


def check_thresholds(metrics: dict) -> dict:
    """Check if metrics are within acceptable thresholds.
    
    Returns a status dict the agent uses to decide next action:
    - healthy: no action needed
    - retrain: model performance has degraded significantly
    - alert: metrics are borderline, human should be notified
    """
    status = "healthy"
    reasons = []

    if metrics["rmse"] > RMSE_THRESHOLD:
        status = "retrain"
        reasons.append(f"RMSE {metrics['rmse']:.4f} exceeds threshold {RMSE_THRESHOLD}")

    if metrics["r2"] < R2_THRESHOLD:
        status = "retrain" if status == "retrain" else "alert"
        reasons.append(f"R2 {metrics['r2']:.4f} below threshold {R2_THRESHOLD}")

    result = {"status": status, "reasons": reasons, "metrics": metrics}
    logger.info(f"Threshold check: {status} — {reasons if reasons else 'all good'}")
    return result


def evaluate(run_id: str = None, version: str = None) -> dict:
    """Full evaluation pipeline — called by the agent or standalone.
    
    Returns evaluation report with metrics, status, and threshold check.
    """
    # Load features
    if version is None:
        version = get_latest_version()
    X, y = load_features(version)

    # Reproduce same test split as training
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Load model
    if run_id is None:
        run_id = get_latest_run_id()
    model = load_model(run_id)

    # Predict and compute metrics
    y_pred  = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)

    # Save plot
    plot_path = plot_predictions(y_test, y_pred, run_id)

    # Check thresholds
    report = check_thresholds(metrics)
    report["run_id"]     = run_id
    report["version"]    = version
    report["plot_path"]  = str(plot_path)

    # Log evaluation metrics back to MLflow
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("eval_rmse", metrics["rmse"])
        mlflow.log_metric("eval_mae",  metrics["mae"])
        mlflow.log_metric("eval_r2",   metrics["r2"])
        mlflow.log_artifact(str(plot_path))
    logger.info("Evaluation metrics logged to MLflow")

    return report


if __name__ == "__main__":
    report = evaluate()
    print("\nEvaluation Report:")
    print(f"  Status  : {report['status']}")
    print(f"  RMSE    : {report['metrics']['rmse']:.4f}")
    print(f"  MAE     : {report['metrics']['mae']:.4f}")
    print(f"  R2      : {report['metrics']['r2']:.4f}")
    print(f"  Reasons : {report['reasons'] if report['reasons'] else 'none'}")
    print(f"  Plot    : {report['plot_path']}")