"""First transformation stage: validate columns and convert dates/numbers."""
import pandas as pd

# Define the same entity keys and key order as the notebooks.
ENTITY_KEYS = [
    "rse_abn",
    "abn_product_identifier",
    "abn_investment_menu_identifier",
    "abn_investment_option_identifier",
]

SAA_ENTITY_KEYS = [
    "rse_abn",
    "investment_option_identifier",
]

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


# STEP 5: Apply the notebook's identifier-formatting convention.
# Missing required identifiers stop the run for review.
def clean_identifier(series):
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.upper()
        .replace({
            "": pd.NA,
            "NAN": pd.NA,
            "NONE": pd.NA,
        })
    )

    if cleaned.isna().any():
        raise ValueError(
            f"{series.name}: missing required identifiers."
        )

    return cleaned


# Keep ABNs as text and pad to 11 digits.
# Accept an integer-style '.0' suffix without converting through floats.
def clean_abn(series):
    cleaned = clean_identifier(series).str.replace(
        r"\.0+$",
        "",
        regex=True,
    )

    invalid = ~cleaned.str.fullmatch(
        r"[0-9]{1,11}",
        na=False,
    )

    if invalid.any():
        examples = series.loc[invalid].head(5).tolist()
        raise ValueError(
            f"{series.name}: invalid ABNs: {examples}"
        )

    return cleaned.str.zfill(11)

# Apply the initial transformations to copies.
# Preserve all source rows, columns, identifiers and names.
def prepare_historical_data(historical_performance, historical_saa):
    performance = historical_performance.copy()
    saa = historical_saa.copy()

    # Validate the required source columns.
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

    # Convert reporting dates before chronological sorting.
    performance["period_end_date"] = convert_dates(
        performance["period_end_date"]
    )
    saa["time_key"] = convert_dates(saa["time_key"])

    # Convert numerical fields using the existing helper.
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

    # Standardise the join identifiers consistently across both sources.
    performance["rse_abn"] = clean_abn(performance["rse_abn"])
    saa["rse_abn"] = clean_abn(saa["rse_abn"])

    for column in ENTITY_KEYS[1:]:
        performance[column] = clean_identifier(
            performance[column]
        )

    saa["investment_option_identifier"] = clean_identifier(
        saa["investment_option_identifier"]
    )

    # Group histories by entity, with dates ascending within each entity.
    # Repeated fee arrangements and SAA asset rows remain at this stage.
    performance = (
        performance
        .sort_values(
            ENTITY_KEYS + ["period_end_date"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    saa = (
        saa
        .sort_values(
            SAA_ENTITY_KEYS + ["time_key"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    print(
        "Initial preparation passed: columns, dates, numbers, "
        "identifiers and sorting."
    )

    return performance, saa