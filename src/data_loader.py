"""STEP 0 - Load the raw drug review CSV file."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config import Config

logger = logging.getLogger(__name__)


class DataLoader:
    """Reads every .csv in data/raw into one DataFrame."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()

    @staticmethod
    def _read(path: Path) -> pd.DataFrame:
        try:
            df = pd.read_csv(path, sep=",", encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, sep=",", encoding="latin-1")
        # Drop any unnamed leading index column.
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        df["source_file"] = path.name
        return df

    def load(self) -> pd.DataFrame:
        files = sorted(Path(self.config.raw_dir).glob("*.csv"))
        if not files:
            raise FileNotFoundError(
                f"No .csv files found in {self.config.raw_dir}. "
                "Place your drug review CSV file there."
            )
        df = pd.concat([self._read(f) for f in files], ignore_index=True)
        missing = set(self.config.required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing expected columns: {sorted(missing)}")
        logger.info("Loaded %d rows from %d file(s)", len(df), len(files))
        return df
