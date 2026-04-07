import logging
import mlflow
import mlflow.sklearn
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from pipeline.feature_store import load_features, get_latest_version

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# MLflow experiment name — all runs are grouped under this
EXPERIMENT_NAME = "ml-ops-agent"

# Model hyperparameters
N_ESTIMATORS = 100   # number of decision trees in the forest
MAX_DEPTH    = 10    # how deep each tree can grow
TEST_SIZE    = 0.2   # 20% of data held out for evaluation
RANDOM_STATE = 42    # seed for reproducibility


def train(version: str = None) -> str:
    """Load features, train a Random Forest model, log everything to MLflow.
    
    Returns the MLflow run_id so the evaluate module can find this run later.
    """

    # --- 1. Load features ---
    if version is None:
        version = get_latest_version()
    logger.info(f"Loading features from version: {version}")
    X, y = load_features(version)

    # --- 2. Train / test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # --- 3. Start MLflow run ---
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run started: {run_id}")

        # --- 4. Train model ---
        model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            random_state=RANDOM_STATE,
            n_jobs=-1   # use all CPU cores
        )
        model.fit(X_train, y_train)
        logger.info("Model training complete")

        # --- 5. Evaluate on test set ---
        y_pred = model.predict(X_test)
        rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
        r2     = r2_score(y_test, y_pred)
        logger.info(f"RMSE: {rmse:.4f} | R2: {r2:.4f}")

        # --- 6. Log params and metrics to MLflow ---
        mlflow.log_param("n_estimators",  N_ESTIMATORS)
        mlflow.log_param("max_depth",     MAX_DEPTH)
        mlflow.log_param("test_size",     TEST_SIZE)
        mlflow.log_param("feature_version", version)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2",   r2)

        # --- 7. Log feature importance ---
        importances = dict(zip(X.columns, model.feature_importances_))
        top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info("Top 5 important features:")
        for feat, score in top_features:
            logger.info(f"  {feat}: {score:.4f}")
            safe_feat = feat.replace("(", "_").replace(")", "").replace(".", "_")
            mlflow.log_metric(f"importance_{safe_feat}", score)

        # --- 8. Save model to MLflow ---
        mlflow.sklearn.log_model(model, artifact_path="model")
        logger.info(f"Model logged to MLflow (run_id={run_id})")

    return run_id


if __name__ == "__main__":
    run_id = train()
    print(f"\nTraining complete. MLflow run_id: {run_id}")
    print("To view results run: mlflow ui")