import pandas as pd
import numpy as np


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


# Aggregate prepared SAA data to fund/option/quarter.
def aggregate_saa_allocations(historical_saa):
    # Define the aggregation keys and allocation field.
    option_period_keys = SAA_ENTITY_KEYS + ["time_key"]
    allocation_column = "InvestmentBenchmarkAllocationPercent"

    check_columns(
        historical_saa,
        SAA_REQUIRED_COLUMNS,
        "historical_saa",
    )

    # Retain every option-quarter, including those with missing allocations.
    option_period_base = (
        historical_saa[option_period_keys]
        .drop_duplicates()
    )

    # Exclude Currency Exposure from physical asset-allocation totals.
    currency_rows = (
        historical_saa["InvestmentStrategicSectorType"]
        .eq("Currency Exposure")
        .fillna(False)
    )
    asset_rows = historical_saa.loc[~currency_rows].copy()

    # Aggregate each sector, preserving all-missing groups as missing.
    sector_allocation_long = (
        asset_rows
        .groupby(
            option_period_keys + ["ConsolidatedSectorType"],
            as_index=False,
            dropna=False,
        )
        .agg(
            sector_allocation=(
                allocation_column,
                lambda values: values.sum(min_count=1),
            )
        )
    )

    # Use the allocation column names established in the audit notebook.
    sector_rename = {
        "Equity": "allocation_equity",
        "Property": "allocation_property",
        "Fixed Income": "allocation_fixed_income",
        "Infrastructure": "allocation_infrastructure",
        "Cash": "allocation_cash",
        "Fixed Income Excluding Credit":
            "allocation_fixed_income_excluding_credit",
        "Alternatives": "allocation_alternatives",
        "Credit": "allocation_credit",
        "Defensive Alternatives": "allocation_defensive_alternatives",
        "Growth Alternatives": "allocation_growth_alternatives",
    }

    sector_allocation_wide = (
        sector_allocation_long
        .pivot(
            index=option_period_keys,
            columns="ConsolidatedSectorType",
            values="sector_allocation",
        )
        .rename(columns=sector_rename)
        .reset_index()
    )

    allocation_columns = list(sector_rename.values())

    # Maintain the expected columns when a sector is absent from an edition.
    # Conditional zero-filling is applied after calculating total availability.
    for column in allocation_columns:
        if column not in sector_allocation_wide:
            sector_allocation_wide[column] = np.nan

    # Calculate the reported total independently of the sector features.
    allocation_quality = (
        asset_rows
        .groupby(option_period_keys, as_index=False)
        .agg(
            total_benchmark_allocation=(
                allocation_column,
                lambda values: values.sum(min_count=1),
            )
        )
    )

    allocation_quality["allocation_approx_100"] = np.isclose(
        allocation_quality["total_benchmark_allocation"],
        1.0,
        atol=0.001,
    )

    # Join aggregates to the complete option-quarter population.
    features = (
        option_period_base
        .merge(
            sector_allocation_wide,
            on=option_period_keys,
            how="left",
            validate="one_to_one",
        )
        .merge(
            allocation_quality,
            on=option_period_keys,
            how="left",
            validate="one_to_one",
        )
    )

    features["benchmark_allocation_missing"] = (
        features["total_benchmark_allocation"].isna()
    )

    # Fill absent sectors only when an allocation total is observed.
    # All-missing allocation groups remain missing.
    available = features["total_benchmark_allocation"].notna()

    features.loc[available, allocation_columns] = (
        features.loc[available, allocation_columns]
        .fillna(0)
    )

    # Preserve the notebook's allocation-composition checks.
    features["sector_feature_total"] = (
        features[allocation_columns]
        .sum(axis=1, min_count=1)
    )

    features["sector_features_match_total"] = np.isclose(
        features["sector_feature_total"],
        features["total_benchmark_allocation"],
        atol=1e-6,
        equal_nan=True,
    )

    # Fix the output column order and retain any additional sector columns.
    ordered_columns = option_period_keys + allocation_columns + [
        "total_benchmark_allocation",
        "allocation_approx_100",
        "benchmark_allocation_missing",
        "sector_feature_total",
        "sector_features_match_total",
    ]
    extra_columns = [
        column
        for column in features
        if column not in ordered_columns
    ]

    features = (
        features[ordered_columns + extra_columns]
        .sort_values(option_period_keys, kind="stable")
        .reset_index(drop=True)
    )

    print(
        f"SAA sector allocations prepared: "
        f"{len(features):,} option-quarters, "
        f"{len(features.columns):,} columns."
    )

    return features


# Add domicile allocations and separate Currency Exposure fields.
def add_saa_domicile_and_currency(
    option_period_saa_features,
    historical_saa,
):
    # Use the prepared SAA data and the allocation table from Step 8.
    option_period_keys = SAA_ENTITY_KEYS + ["time_key"]
    allocation_column = "InvestmentBenchmarkAllocationPercent"

    check_columns(
        historical_saa,
        SAA_REQUIRED_COLUMNS,
        "historical_saa",
    )
    check_columns(
        option_period_saa_features,
        option_period_keys + ["total_benchmark_allocation"],
        "option_period_saa_features",
    )

    # Identify Currency Exposure rows separately from asset allocations.
    currency_rows = (
        historical_saa["InvestmentStrategicSectorType"]
        .eq("Currency Exposure")
        .fillna(False)
    )

    currency_exposure = (
        historical_saa.loc[currency_rows]
        .groupby(option_period_keys, as_index=False)
        .agg(
            currency_exposure_allocation=(
                allocation_column,
                lambda values: values.sum(min_count=1),
            )
        )
    )

    # A row can be reported even when its allocation value is missing.
    currency_exposure["currency_exposure_reported"] = True

    features = (
        option_period_saa_features.copy()
        .merge(
            currency_exposure,
            on=option_period_keys,
            how="left",
            validate="one_to_one",
        )
    )

    features["currency_exposure_reported"] = (
        features["currency_exposure_reported"]
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )

    # Aggregate physical asset allocations by domicile.
    # Currency Exposure remains excluded from these totals.
    domicile_long = (
        historical_saa.loc[~currency_rows]
        .groupby(
            option_period_keys + ["ConsolidatedDomicileType"],
            as_index=False,
            dropna=False,
        )
        .agg(
            domicile_allocation=(
                allocation_column,
                lambda values: values.sum(min_count=1),
            )
        )
    )

    domicile_rename = {
        "Australian Domicile": "allocation_australian_domicile",
        "International Domicile": "allocation_international_domicile",
        "Not Specified": "allocation_domicile_not_specified",
        "Not Applicable": "allocation_domicile_not_applicable",
    }

    domicile_wide = (
        domicile_long
        .pivot(
            index=option_period_keys,
            columns="ConsolidatedDomicileType",
            values="domicile_allocation",
        )
        .rename(columns=domicile_rename)
        .reset_index()
    )

    domicile_columns = list(domicile_rename.values())

    # Maintain expected columns when a category is absent from an edition.
    for column in domicile_columns:
        if column not in domicile_wide:
            domicile_wide[column] = np.nan

    features = features.merge(
        domicile_wide,
        on=option_period_keys,
        how="left",
        validate="one_to_one",
    )

    # Fill absent domiciles only when a total allocation is observed.
    # Groups with entirely missing allocations remain missing.
    available = features["total_benchmark_allocation"].notna()

    features.loc[available, domicile_columns] = (
        features.loc[available, domicile_columns]
        .fillna(0)
    )

    # Retain the notebook's domicile composition checks.
    features["domicile_feature_total"] = (
        features[domicile_columns]
        .sum(axis=1, min_count=1)
    )

    features["domicile_features_match_total"] = np.isclose(
        features["domicile_feature_total"],
        features["total_benchmark_allocation"],
        atol=1e-6,
        equal_nan=True,
    )

    # Preserve the option-quarter population and chronological ordering.
    features = (
        features
        .sort_values(option_period_keys, kind="stable")
        .reset_index(drop=True)
    )

    print(
        f"SAA domicile and Currency Exposure fields added: "
        f"{len(features):,} option-quarters, "
        f"{len(features.columns):,} columns."
    )

    return features


# Calculate strategic growth allocation and its diagnostic flags.
def add_strategic_growth_allocation(
    option_period_saa_features,
    historical_saa,
):
    option_period_keys = SAA_ENTITY_KEYS + ["time_key"]
    allocation_column = "InvestmentBenchmarkAllocationPercent"

    check_columns(
        historical_saa,
        SAA_REQUIRED_COLUMNS,
        "historical_saa",
    )
    check_columns(
        option_period_saa_features,
        option_period_keys,
        "option_period_saa_features",
    )

    # Exclude Currency Exposure, consistently with the asset-allocation totals.
    currency_rows = (
        historical_saa["InvestmentStrategicSectorType"]
        .eq("Currency Exposure")
        .fillna(False)
    )
    growth_rows = historical_saa.loc[~currency_rows].copy()

    sector = growth_rows["ConsolidatedSectorType"]
    listing = growth_rows["ConsolidatedListingType"]

    # Preserve the sector weights established in the notebook.
    sector_weights = {
        "Equity": 1.00,
        "Growth Alternatives": 0.75,
        "Alternatives": 0.50,
        "Credit": 0.30,
        "Fixed Income": 0.25,
        "Defensive Alternatives": 0.25,
        "Fixed Income Excluding Credit": 0.00,
        "Cash": 0.00,
    }

    # Property and infrastructure weights depend on listing status.
    listing_weights = {
        "Listed": 1.00,
        "Unlisted": 0.75,
        "Not Specified": 0.875,
        "Not Applicable": 0.875,
    }

    growth_rows["growth_weight"] = sector.map(sector_weights)

    property_or_infrastructure = sector.isin(
        ["Property", "Infrastructure"]
    )

    growth_rows.loc[
        property_or_infrastructure, "growth_weight"
    ] = (
        listing.loc[property_or_infrastructure]
        .map(listing_weights)
    )

    # Stop if a reported allocation has an unrecognised classification.
    # A new classification needs review before assigning a growth weight.
    unknown = (
        growth_rows[allocation_column].notna()
        & growth_rows["growth_weight"].isna()
    )

    if unknown.any():
        examples = (
            growth_rows.loc[
                unknown,
                ["ConsolidatedSectorType", "ConsolidatedListingType"],
            ]
            .drop_duplicates()
            .head(5)
            .to_dict("records")
        )
        raise ValueError(
            f"Unrecognised SAA growth classifications: {examples}"
        )

    # Track the notebook's 50/50 listed/unlisted assumption.
    listing_assumption = (
        property_or_infrastructure
        & listing.isin(["Not Specified", "Not Applicable"])
    )

    growth_rows["growth_allocation_component"] = (
        growth_rows[allocation_column]
        * growth_rows["growth_weight"]
    )

    growth_rows["listing_assumption_allocation_component"] = np.where(
        listing_assumption,
        growth_rows[allocation_column],
        0,
    )

    # Aggregate the weighted numerator and reported allocation base.
    growth = (
        growth_rows
        .groupby(option_period_keys, as_index=False)
        .agg(
            growth_allocation_numerator=(
                "growth_allocation_component",
                lambda values: values.sum(min_count=1),
            ),
            growth_allocation_base=(
                allocation_column,
                lambda values: values.sum(min_count=1),
            ),
            listing_assumption_allocation=(
                "listing_assumption_allocation_component",
                "sum",
            ),
        )
    )

    # Express growth relative to the allocation that was actually reported.
    growth["strategic_growth_allocation"] = (
        growth["growth_allocation_numerator"]
        / growth["growth_allocation_base"]
    )

    growth["listing_assumption_share"] = (
        growth["listing_assumption_allocation"]
        / growth["growth_allocation_base"]
    )

    # Preserve the notebook's diagnostic definitions.
    growth["growth_allocation_low_coverage"] = (
        growth["growth_allocation_base"].notna()
        & growth["growth_allocation_base"].lt(0.90)
    )

    growth["growth_allocation_missing"] = (
        growth["growth_allocation_base"].isna()
    )

    growth["growth_listing_assumption_used"] = (
        growth["listing_assumption_share"].gt(0)
    )

    growth["strategic_growth_above_one"] = (
        growth["strategic_growth_allocation"].gt(1)
    )

    # Merge the seven exported fields without removing option-quarters.
    growth_columns = option_period_keys + [
        "strategic_growth_allocation",
        "growth_allocation_base",
        "listing_assumption_share",
        "growth_allocation_low_coverage",
        "growth_allocation_missing",
        "growth_listing_assumption_used",
        "strategic_growth_above_one",
    ]

    features = (
        option_period_saa_features.copy()
        .merge(
            growth[growth_columns],
            on=option_period_keys,
            how="left",
            validate="one_to_one",
        )
        .sort_values(option_period_keys, kind="stable")
        .reset_index(drop=True)
    )

    print(
        f"Strategic growth fields added: "
        f"{len(features):,} option-quarters, "
        f"{len(features.columns):,} columns."
    )

    return features