import argparse
from pathlib import Path
import pandas as pd
from transform import (
    prepare_historical_data,
    aggregate_saa_allocations,
    add_saa_domicile_and_currency,
    add_strategic_growth_allocation,
)


# Locate the project folder containing this script.
PROJECT_ROOT = Path(__file__).resolve().parent

# Define the existing data folders.
# SOURCE_EDITION = Path(__file__).resolve().parent
RAW_BASE_DIR = PROJECT_ROOT / "data" / "raw" / "apra"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Find one historical CSV using terms in its filename.
# This supports APRA filenames containing additional words or spaces.
def find_single_csv(directory: Path, required_terms: list[str]) -> Path:
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Raw data folder does not exist: {directory}"
        )

    matches = [
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".csv"
        and all(
            term.lower() in path.name.lower()
            for term in required_terms
        )
    ]

    # Stop if the source file is missing or the match is ambiguous.
    if len(matches) != 1:
        raise ValueError(
            f"Expected one CSV containing {required_terms} in {directory}. "
            f"Found: {[path.name for path in matches]}"
        )

    return matches[0]


# Load every original column without guessing identifier/date formats.
# Empty fields and quoted-empty fields are both read as missing values.
# Numeric and date conversion will be added in the transformation step.
def load_csv(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, dtype="string", keep_default_na=True)

    if data.empty:
        raise ValueError(f"The CSV contains no observations: {path}")

    print(f"Loaded: {path.name}")
    print(f"Rows: {data.shape[0]:,} | Columns: {data.shape[1]:,}\n")

    return data

# Run the loading steps in order.
# The original CSVs stay on disk and all their rows remain in these dataframes.
def main() -> None:
    # Read the edition supplied in the terminal.
    parser = argparse.ArgumentParser(
        description="Load an APRA historical data edition."
    )
    parser.add_argument(
        "--edition",
        required=True,
        help="Reporting edition in YYYY-MM format, for example 2026-03.",
    )
    args = parser.parse_args()

    # Build the selected edition's folder path.
    RAW_DATA_DIR = RAW_BASE_DIR / args.edition
    
    print(f"Source edition: {args.edition}")
    print(f"Raw data folder: {RAW_DATA_DIR}\n")

    historical_performance_path = find_single_csv(
        RAW_DATA_DIR, ["historical", "performance"]
    )
    historical_saa_path = find_single_csv(
        RAW_DATA_DIR, ["historical", "saa"]
    )

    historical_performance = load_csv(historical_performance_path)
    historical_saa = load_csv(historical_saa_path)

    # Apply the initial transformations defined in transform.py.
    historical_performance, historical_saa = prepare_historical_data(
        historical_performance,
        historical_saa,
    )

    # Basic source checks before adding the column/date validation step.
    print("Both historical datasets were loaded successfully.")
    print(f"Performance observations: {len(historical_performance):,}")
    print(f"SAA observations: {len(historical_saa):,}")
    
    # Aggregate the prepared SAA rows to option-quarter sector allocations.
    option_period_saa_features = aggregate_saa_allocations(
        historical_saa
    )

    # Extend the sector-allocation table with domicile and currency fields.
    option_period_saa_features = add_saa_domicile_and_currency(
        option_period_saa_features,
        historical_saa,
    )

    # Add strategic growth allocation using the established notebook weights.
    option_period_saa_features = add_strategic_growth_allocation(
        option_period_saa_features,
        historical_saa,
    )

# ------------ TRANSFORM ---------------------------

# --------------------------------------------------
if __name__ == "__main__":
    main()