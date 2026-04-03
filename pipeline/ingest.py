import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_PATH = Path("data/sample.csv")

REQUIRED_COLUMNS = [
    "Date", "Time", "CO(GT)", "PT08.S1(CO)", "NMHC(GT)",
    "C6H6(GT)", "PT08.S2(NMHC)", "NOx(GT)", "PT08.S3(NOx)",
    "NO2(GT)", "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH"
]


def load_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path, sep=";", decimal=",")
    df = df.dropna(axis=1, how="all")  # drop empty trailing columns
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    return df


def validate_schema(df: pd.DataFrame) -> bool:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        logger.error(f"Missing columns: {missing}")
        return False
    logger.info("Schema validation passed")
    return True


def remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    initial = len(df)
    # UCI dataset uses -200 as a sentinel for missing values
    df = df.replace(-200, pd.NA)
    df = df.dropna(subset=["CO(GT)", "C6H6(GT)", "T", "RH"])
    logger.info(f"Removed {initial - len(df)} invalid rows. {len(df)} rows remaining")
    return df


def ingest(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    df = load_data(path)
    if not validate_schema(df):
        raise ValueError("Schema validation failed. Check your data file.")
    df = remove_invalid_rows(df)
    logger.info("Ingestion complete")
    return df


if __name__ == "__main__":
    df = ingest()
    print(df.head())