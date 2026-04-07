import logging
from datetime import datetime
from pipeline.feature_store import load_features, get_latest_version, list_versions
from training.evaluate import evaluate
from training.train import train

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def query_feature_store() -> dict:
    """Tool 1: Query the feature store for latest data statistics.
    
    The agent calls this to understand the current state of the data
    before making any decisions about retraining or alerting.
    """
    version = get_latest_version()
    if version is None:
        return {"error": "No versions found in feature store"}

    X, y = load_features(version)

    stats = {
        "latest_version"  : version,
        "all_versions"    : list_versions(),
        "num_rows"        : int(X.shape[0]),
        "num_features"    : int(X.shape[1]),
        "target_mean"     : round(float(y.mean()), 4),
        "target_std"      : round(float(y.std()), 4),
        "target_min"      : round(float(y.min()), 4),
        "target_max"      : round(float(y.max()), 4),
    }

    logger.info(f"Feature store queried — version: {version}, rows: {stats['num_rows']}")
    return stats


def run_evaluation() -> dict:
    """Tool 2: Run the evaluation harness against the latest model.

    Returns a full evaluation report including metrics and status.
    The agent reads the status field to decide next action:
    - healthy  → no action needed
    - alert    → notify a human
    - retrain  → trigger retraining
    """
    logger.info("Running evaluation harness...")
    report = evaluate()
    logger.info(f"Evaluation complete — status: {report['status']}")
    return report


def trigger_retrain() -> dict:
    """Tool 3: Trigger a new training run using the latest feature version.

    Called by the agent when evaluation status is 'retrain'.
    Returns the new MLflow run_id and updated metrics.
    """
    logger.info("Triggering retraining pipeline...")
    run_id = train()

    result = {
        "status"    : "retrained",
        "run_id"    : run_id,
        "timestamp" : datetime.now().isoformat(),
        "message"   : f"New model trained successfully. MLflow run_id: {run_id}"
    }

    logger.info(f"Retraining complete — new run_id: {run_id}")
    return result


def send_alert(message: str, level: str = "warning") -> dict:
    """Tool 4: Send an alert notification.

    In production this would send a Slack message or email.
    For now it logs the alert and writes it to a local alerts file
    so we can verify the agent is calling it correctly.
    """
    timestamp = datetime.now().isoformat()
    alert = {
        "level"     : level,
        "message"   : message,
        "timestamp" : timestamp
    }

    # Log to console
    logger.warning(f"ALERT [{level.upper()}]: {message}")

    # Write to alerts log file
    with open("data/alerts.log", "a") as f:
        f.write(f"[{timestamp}] [{level.upper()}] {message}\n")

    return {"status": "alert_sent", **alert}