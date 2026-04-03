import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Sensor columns we use as input features
SENSOR_COLS = [
    "PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)",
    "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH"
]

# What we want to predict
TARGET_COL = "CO(GT)"


def parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Combine Date and Time into a single datetime column."""
    df = df.copy()
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H.%M.%S",
        errors="coerce"
    )
    df = df.dropna(subset=["datetime"])
    logger.info("Datetime parsing complete")
    return df


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hour, day of week, and month from datetime.
    
    These help the model learn time-based pollution patterns:
    - rush hour (hour)
    - weekday vs weekend (day_of_week)
    - seasonal patterns (month)
    """
    df = df.copy()
    df["hour"]        = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek  # 0=Monday, 6=Sunday
    df["month"]       = df["datetime"].dt.month
    logger.info("Time features extracted: hour, day_of_week, month")
    return df


def add_rolling_features(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Add rolling mean for each sensor column.

    Sensors are noisy — a single spike reading can mislead the model.
    A rolling average smooths this out by averaging the last N readings.
    window=3 means we average the current + previous 2 hourly readings.
    """
    df = df.copy()
    for col in SENSOR_COLS:
        df[f"{col}_roll{window}"] = (
            df[col].rolling(window=window, min_periods=1).mean()
        )
    logger.info(f"Rolling mean features added with window={window}")
    return df


def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select final feature matrix X and target vector y."""
    time_cols    = ["hour", "day_of_week", "month"]
    rolling_cols = [f"{col}_roll3" for col in SENSOR_COLS]
    feature_cols = SENSOR_COLS + time_cols + rolling_cols

    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()

    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Target column: {TARGET_COL}, shape: {y.shape}")
    return X, y


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Full feature engineering pipeline — called by downstream modules."""
    df = parse_datetime(df)
    df = extract_time_features(df)
    df = add_rolling_features(df)
    X, y = select_features(df)
    logger.info("Feature engineering complete")
    return X, y


if __name__ == "__main__":
    from pipeline.ingest import ingest
    df = ingest()
    X, y = engineer_features(df)
    print("\nFeature matrix (first 3 rows):")
    print(X.head(3))
    print("\nTarget (first 3 values):")
    print(y.head(3))