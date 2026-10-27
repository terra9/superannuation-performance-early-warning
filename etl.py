import argparse
from pathlib import Path
import pandas as pd
from transform import (
    prepare_historical_data,
    aggregate_saa_allocations,
    add_saa_domicile_and_currency,
    add_strategic_growth_allocation,
    add_currency_hedging_features,
    add_performance_history_and_volatility,
)
from transform import merge_performance_and_saa, build_model_base
from transform import select_mysuper_population
from transform import add_future_4q_returns, add_peer_relative_target
from transform import (
    add_historical_return_features,
    add_current_peer_features,
)
from transform import add_volatility_and_strategy_features
from transform import validate_model_features


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

    # ------------ TRANSFORM ---------------------------
    
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

    # Complete the SAA table with the established hedging features.
    option_period_saa_features = add_currency_hedging_features(
        option_period_saa_features,
        historical_saa,
    )

    # Prepare the performance table for its later join with SAA.
    hist_enriched = add_performance_history_and_volatility(
        historical_performance
    )

    # Join the completed SAA features to the enriched performance history.
    hist_enriched_saa = merge_performance_and_saa(
        hist_enriched,
        option_period_saa_features,
    )

    # Collapse repeated fee arrangements to the established modelling grain.
    hist_model_base = build_model_base(hist_enriched_saa)

    # Select MySuper while retaining the complete modelling base separately.
    hist_model_mysuper = select_mysuper_population(hist_model_base)

    # Calculate future returns within each original representation.
    hist_model_mysuper_target = add_future_4q_returns(
        hist_model_mysuper
    )

    # Construct distinct-option peer targets and option-level availability.
    hist_model_mysuper_target_peer = add_peer_relative_target(
        hist_model_mysuper_target
    )

    # Construct historical predictors from the complete MySuper history.
    hist_model_mysuper_features = add_historical_return_features(
        hist_model_mysuper_target_peer
    )

    # Add current-quarter peer comparisons.
    hist_model_mysuper_features = add_current_peer_features(
        hist_model_mysuper_features
    )

    # Complete the remaining predictors using the existing dataframe.
    hist_model_mysuper_features = add_volatility_and_strategy_features(
        hist_model_mysuper_features
    )

    # STEP 23: Validate and save the complete prepared dataset.
    validate_model_features(hist_model_mysuper_features)

    # Reuse the existing processed-data folder.
    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Include the selected edition in the filename.
    output_path = (
        processed_dir
        / f"hist_model_mysuper_features_{args.edition}.parquet"
    )

    # Preserve identifiers, all engineered columns, targets and unavailable-target rows.
    hist_model_mysuper_features.to_parquet(
        output_path,
        index=False,
    )

    print(f"Prepared dataset saved: {output_path}")

    # --------------------------------------------------
if __name__ == "__main__":
    main()