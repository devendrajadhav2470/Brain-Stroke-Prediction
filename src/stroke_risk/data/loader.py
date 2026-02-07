"""Data loading utilities."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_RAW_PATH = Path("data/raw/brain_stroke.csv")
DEFAULT_HOLDOUT_PATH = Path("data/raw/full_filled_stroke_data (1).csv")


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the brain stroke dataset as a pandas DataFrame.

    Parameters
    ----------
    path : str or Path or None
        Local path to the CSV. If None, uses the default raw path.

    Returns
    -------
    pd.DataFrame
        The loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist at the given path.
    """
    path = Path(path) if path else DEFAULT_RAW_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Please place the CSV file in data/raw/."
        )

    df = pd.read_csv(path)
    logger.info("Loaded dataset from %s: %d rows, %d columns.", path, len(df), len(df.columns))

    return df


def load_holdout_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the holdout / external evaluation dataset.

    Parameters
    ----------
    path : str or Path or None
        Local path to the holdout CSV.  If None, uses the default holdout path.

    Returns
    -------
    pd.DataFrame
        The loaded holdout dataset.

    Raises
    ------
    FileNotFoundError
        If the holdout file does not exist at the given path.
    """
    path = Path(path) if path else DEFAULT_HOLDOUT_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Holdout dataset not found at {path}. "
            "Please place the CSV file in data/raw/."
        )

    df = pd.read_csv(path)
    logger.info("Loaded holdout dataset from %s: %d rows, %d columns.", path, len(df), len(df.columns))

    return df
