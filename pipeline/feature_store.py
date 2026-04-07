import pandas as pd
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

STORE_DIR = Path("data/feature_store")


def _versioned_path(version: str) -> Path:
    """Build the folder path for a given version."""
    return STORE_DIR / version


def save_features(X: pd.DataFrame, y: pd.Series, version: str = None) -> str:
    """Save feature matrix X and target y to the feature store.

    If no version is given, one is auto-generated from the current timestamp.
    Each version is saved in its own folder so nothing gets overwritten.
    """
    if version is None:
        version = datetime.now().strftime("v_%Y%m%d_%H%M%S")

    path = _versioned_path(version)
    path.mkdir(parents=True, exist_ok=True)

    X.to_csv(path / "X.csv", index=False)
    y.to_csv(path / "y.csv", index=False)

    logger.info(f"Features saved to {path} (version={version})")
    return version


def load_features(version: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load feature matrix X and target y for a given version."""
    path = _versioned_path(version)

    if not path.exists():
        raise FileNotFoundError(f"No feature store found for version: {version}")

    X = pd.read_csv(path / "X.csv")
    y = pd.read_csv(path / "y.csv").squeeze()  # load as Series not DataFrame

    logger.info(f"Loaded features from {path} — X: {X.shape}, y: {y.shape}")
    return X, y


def list_versions() -> list[str]:
    """List all saved versions in the feature store."""
    if not STORE_DIR.exists():
        return []
    versions = sorted([d.name for d in STORE_DIR.iterdir() if d.is_dir()])
    logger.info(f"Available versions: {versions}")
    return versions


def get_latest_version() -> str | None:
    """Return the most recently saved version."""
    versions = list_versions()
    if not versions:
        logger.warning("No versions found in feature store")
        return None
    latest = versions[-1]
    logger.info(f"Latest version: {latest}")
    return latest


if __name__ == "__main__":
    from pipeline.ingest import ingest
    from pipeline.features import engineer_features

    # Run full pipeline and save to feature store
    df = ingest()
    X, y = engineer_features(df)
    version = save_features(X, y)

    # Verify we can load it back
    print(f"\nSaved version: {version}")
    print(f"All versions: {list_versions()}")

    X_loaded, y_loaded = load_features(version)
    print(f"\nLoaded X shape: {X_loaded.shape}")
    print(f"Loaded y shape: {y_loaded.shape}")