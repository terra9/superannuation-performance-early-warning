"""First transformation stage: validate columns and convert dates/numbers."""
import pandas as pd

# Define needed columns
# Additional source columns will also be retained.
PERFORMANCE_NUMERIC_COLUMNS = [
    "return_measurement_comparison_percent",
    "return_investment_five_year_volatility_comparison_percent",
    "return_investment_ten_year_volatility_comparison_percent",
]

SAA_NUMERIC_COLUMNS = [
    "InvestmentBenchmarkAllocationPercent",
    "InvestmentCurrencyHedgingRatioPercent",
]

PERFORMANCE_REQUIRED_COLUMNS = [
    "rse_abn",
    "abn_product_identifier",
    "abn_investment_menu_identifier",
    "abn_investment_option_identifier",
    "fees_and_costs_arrangement_identifier",
    "period_end_date",
    "product_name",
    "product_type",
    "product_phase",
] + PERFORMANCE_NUMERIC_COLUMNS

SAA_REQUIRED_COLUMNS = [
    "rse_abn",
    "investment_option_identifier",
    "time_key",
    "ConsolidatedSectorType",
    "ConsolidatedListingType",
    "ConsolidatedDomicileType",
    "InvestmentStrategicSectorType",
] + SAA_NUMERIC_COLUMNS


# Check required columns before start changing values
def check_columns(data, required_columns, dataset_name):
    missing = sorted(set(required_columns) - set(data.columns))

    if missing:
        raise ValueError(
            f"{dataset_name}: missing required columns: {missing}"
        )

# Convert date formats
# Missing, unrecognised or non-quarter-end dates stop the process.
def convert_dates(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        converted = series.dt.normalize()
    else:
        text = series.astype("string").str.strip()
        converted = pd.Series(
            pd.NaT,
            index=series.index,
            dtype="datetime64[ns]",
        )

        # Old performance, newer ISO, and old SAA formats.
        for date_format in ["%d/%m/%Y", "%Y-%m-%d", "%Y%m%d"]:
            candidate = pd.to_datetime(
                text,
                format=date_format,
                errors="coerce",
            )
            converted = converted.fillna(candidate)

    # Check failures after trying all supported formats.
    invalid = converted.isna() | ~converted.dt.is_quarter_end

    if invalid.any():
        examples = series.loc[invalid].head(5).tolist()
        raise ValueError(
            f"{series.name}: invalid or missing quarter-end dates: "
            f"{examples}"
        )

    return converted

# Convert numerical fields and retain missing values
# Keep the reported units, unexpected text stops the process
def convert_numbers(data, numeric_columns, dataset_name):
    for column in numeric_columns:
        text = (
            data[column]
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
        )

        try:
            data[column] = pd.to_numeric(
                text,
                errors="raise",
            ).astype(float)
        except (ValueError, TypeError) as error:
            raise ValueError(
                f"{dataset_name}.{column}: "
                "unexpected non-numeric values."
            ) from error

# Apply the initial transformations to copies.
# Preserve all source rows, columns, identifiers and names.
def prepare_historical_data(historical_performance, historical_saa):
    performance = historical_performance.copy()
    saa = historical_saa.copy()

    check_columns(
        performance,
        PERFORMANCE_REQUIRED_COLUMNS,
        "historical_performance",
    )
    check_columns(
        saa,
        SAA_REQUIRED_COLUMNS,
        "historical_saa",
    )

    performance["period_end_date"] = convert_dates(
        performance["period_end_date"]
    )
    saa["time_key"] = convert_dates(saa["time_key"])

    convert_numbers(
        performance,
        PERFORMANCE_NUMERIC_COLUMNS,
        "historical_performance",
    )
    convert_numbers(
        saa,
        SAA_NUMERIC_COLUMNS,
        "historical_saa",
    )

    print("Initial column, date and numeric checks passed.")

    return performance, saa