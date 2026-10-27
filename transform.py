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


# Apply the notebook's identifier-formatting convention.
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
    # Use the prepared SAA data and the allocation table from previous steps.
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


# Add the final currency-hedging fields.
def add_currency_hedging_features(
    option_period_saa_features,
    historical_saa,
):
    option_period_keys = SAA_ENTITY_KEYS + ["time_key"]
    allocation_column = "InvestmentBenchmarkAllocationPercent"
    hedging_column = "InvestmentCurrencyHedgingRatioPercent"

    check_columns(
        historical_saa,
        SAA_REQUIRED_COLUMNS,
        "historical_saa",
    )
    check_columns(
        option_period_saa_features,
        option_period_keys + ["growth_allocation_missing"],
        "option_period_saa_features",
    )

    # Apply the notebook's domicile rule across all SAA rows.
    # Qualifying Currency Exposure rows are included here.
    domicile = (
        historical_saa["ConsolidatedDomicileType"]
        .astype("string")
        .str.strip()
        .str.casefold()
    )

    applicable = domicile.str.contains(
        r"international|not specified",
        na=False,
    )

    positive_applicable = (
        applicable
        & historical_saa[allocation_column].gt(0)
    )

    valid = (
        positive_applicable
        & historical_saa[hedging_column].between(0, 1)
    )
    valid_rows = historical_saa.loc[valid].copy()

    # Round only for distinct counts; retain original ratios for weighting.
    valid_rows["hedging_ratio_rounded"] = (
        valid_rows[hedging_column].round(6)
    )
    valid_rows["weighted_component"] = (
        valid_rows[hedging_column]
        * valid_rows[allocation_column]
    )

    hedging = (
        valid_rows
        .groupby(option_period_keys, as_index=False)
        .agg(
            weighted_numerator=("weighted_component", "sum"),
            valid_allocation=(allocation_column, "sum"),
            currency_hedging_distinct_ratios=(
                "hedging_ratio_rounded",
                "nunique",
            ),
        )
    )

    hedging["weighted_currency_hedging_ratio"] = (
        hedging["weighted_numerator"]
        / hedging["valid_allocation"]
    )

    # Coverage includes positive applicable allocations regardless of ratio validity.
    coverage = (
        historical_saa.loc[positive_applicable]
        .groupby(option_period_keys, as_index=False)
        .agg(
            currency_hedging_applicable_allocation=(
                allocation_column,
                lambda values: values.sum(min_count=1),
            )
        )
    )

    # Surface groups affected by the notebook's valid-ratio matching limitation.
    no_valid_ratio_count = len(coverage) - len(hedging)

    if no_valid_ratio_count:
        print(
            f"Review {no_valid_ratio_count:,} option-quarters: "
            "positive applicable allocations have no valid hedging ratio. "
            "Notebook matching behaviour is retained; do not interpret "
            "an unmatched record as proof of no exposure."
        )

    # Attach coverage using the same valid-ratio population as the notebook.
    hedging = hedging.merge(
        coverage,
        on=option_period_keys,
        how="left",
        validate="one_to_one",
    )

    hedging_columns = option_period_keys + [
        "weighted_currency_hedging_ratio",
        "currency_hedging_applicable_allocation",
        "currency_hedging_distinct_ratios",
    ]

    features = (
        option_period_saa_features.copy()
        .merge(
            hedging[hedging_columns],
            on=option_period_keys,
            how="left",
            validate="one_to_one",
            indicator="_hedging_merge",
        )
    )

    # Reproduce the nullable applicability flag from the matching result.
    matched = features["_hedging_merge"].eq("both")

    observed_saa = (
        features["growth_allocation_missing"]
        .astype("boolean")
        .eq(False)
        .fillna(False)
    )

    unmatched_observed_saa = (
        features["_hedging_merge"].eq("left_only")
        & observed_saa
    )

    features["has_applicable_hedging_allocation"] = pd.Series(
        pd.NA,
        index=features.index,
        dtype="boolean",
    )

    features.loc[
        matched, "has_applicable_hedging_allocation"
    ] = True

    features.loc[
        unmatched_observed_saa,
        "has_applicable_hedging_allocation",
    ] = False

    features.loc[
        unmatched_observed_saa,
        "currency_hedging_applicable_allocation",
    ] = 0

    # Preserve the notebook's hedging-state definitions.
    ratio = features["weighted_currency_hedging_ratio"]

    features["currency_hedging_fully_unhedged"] = (
        ratio.notna() & np.isclose(ratio, 0)
    )
    features["currency_hedging_fully_hedged"] = (
        ratio.notna() & np.isclose(ratio, 1)
    )
    features["currency_hedging_partial"] = (
        ratio.notna() & ratio.gt(0) & ratio.lt(1)
    )

    features["currency_hedging_distinct_ratios"] = (
        features["currency_hedging_distinct_ratios"]
        .astype("Int64")
    )

    # Remove the temporary matching indicator and retain chronological order.
    features = (
        features
        .drop(columns="_hedging_merge")
        .sort_values(option_period_keys, kind="stable")
        .reset_index(drop=True)
    )

    print(
        f"SAA hedging fields added: "
        f"{len(features):,} option-quarters, "
        f"{len(features.columns):,} columns."
    )

    return features

# Add CSV-derived performance-history and volatility fields.
def add_performance_history_and_volatility(historical_performance):
    metric_5y = (
        "return_investment_five_year_volatility_comparison_percent"
    )
    metric_10y = (
        "return_investment_ten_year_volatility_comparison_percent"
    )

    check_columns(
        historical_performance,
        ENTITY_KEYS + ["period_end_date", metric_5y, metric_10y],
        "historical_performance",
    )

    performance = historical_performance.copy()

    # Describe each entity's history in the loaded source snapshot.
    # Record count includes fee-arrangement rows, matching the notebook.
    grouped_dates = (
        performance
        .groupby(ENTITY_KEYS, dropna=False)["period_end_date"]
    )

    performance["first_historical_period"] = (
        grouped_dates.transform("min")
    )
    performance["last_historical_period"] = (
        grouped_dates.transform("max")
    )
    performance["historical_record_count"] = (
        grouped_dates.transform("size")
    )

    # Reproduce the observed-history anniversary calculation.
    performance["historical_ten_year_anniversary"] = (
        performance["first_historical_period"]
        + pd.DateOffset(years=10)
    )
    performance["has_ten_years_observed_history"] = (
        performance["period_end_date"].ge(
            performance["historical_ten_year_anniversary"]
        )
    )

    # Preserve the notebook's order: create missingness flags first.
    performance["volatility_10y_missing"] = (
        performance[metric_10y].isna().astype("int8")
    )
    performance["volatility_5y_missing"] = (
        performance[metric_5y].isna().astype("int8")
    )

    # Apply the existing ten-year infinity cleanup.
    performance[metric_10y] = performance[metric_10y].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Preserve the notebook's column names and reported finite values.
    performance[f"{metric_10y}_raw"] = performance[metric_10y]
    performance[f"{metric_10y}_clean"] = performance[metric_10y]

    performance["volatility_10y_quality_status"] = "reported"

    # Retain values above 0.30 and flag them rather than clipping them.
    performance["volatility_10y_high_flag"] = (
        performance[metric_10y].gt(0.30)
    )

    print(
        f"Performance history and volatility fields added: "
        f"{len(performance):,} rows, "
        f"{len(performance.columns):,} columns."
    )

    return performance

# Attach SAA features without removing performance records.
def merge_performance_and_saa(hist_enriched, option_period_saa_features):
    saa_merge_keys = [
        "rse_abn",
        "abn_investment_option_identifier",
        "period_end_date",
    ]

    check_columns(
        hist_enriched,
        ENTITY_KEYS + ["period_end_date"],
        "hist_enriched",
    )
    check_columns(
        option_period_saa_features,
        SAA_ENTITY_KEYS + ["time_key", "weighted_currency_hedging_ratio"],
        "option_period_saa_features",
    )

    # Rename SAA keys to match performance; preserve the original SAA table.
    saa_prepared = option_period_saa_features.rename(
        columns={
            "investment_option_identifier": "abn_investment_option_identifier",
            "time_key": "period_end_date",
        }
    ).copy()

    # Stop accidental repeated joins that would create ambiguous duplicate columns.
    reserved_columns = [
        "_saa_merge",
        "saa_matched",
        "has_applicable_international_allocation",
    ]
    overlap = sorted(
        set(hist_enriched.columns)
        & (
            (set(saa_prepared.columns) - set(saa_merge_keys))
            | set(reserved_columns)
        )
    )
    if overlap:
        raise ValueError(
            f"hist_enriched already contains SAA columns: {overlap}"
        )

    # Several performance representations may share one SAA option-quarter.
    # A left join retains performance rows without matching SAA.
    merged = hist_enriched.merge(
        saa_prepared,
        on=saa_merge_keys,
        how="left",
        validate="many_to_one",
        indicator="_saa_merge",
        sort=False,
    )

    # Preserve two diagnostics from the notebook's earlier hedging join.
    # These describe valid hedging matches.
    # The separate _saa_merge column describes the full SAA match.
    merged["saa_matched"] = (
        merged["weighted_currency_hedging_ratio"].notna()
    )
    merged["has_applicable_international_allocation"] = pd.Series(
        True,
        index=merged.index,
        dtype="object",
    ).where(merged["saa_matched"])

    print(
        f"SAA join: {len(merged):,} performance rows retained; "
        f"{merged['_saa_merge'].eq('both').sum():,} matched rows."
    )
    return merged


# Collapse fee arrangements after checking all other fields agree.
def build_model_base(hist_enriched_saa):
    model_grain = ENTITY_KEYS + ["period_end_date"]
    fee_column = "fees_and_costs_arrangement_identifier"

    check_columns(
        hist_enriched_saa,
        model_grain + [fee_column],
        "hist_enriched_saa",
    )
    if "fee_arrangement_count" in hist_enriched_saa.columns:
        raise ValueError(
            "fee_arrangement_count already exists; use arrangement-level data."
        )

    # Check every non-fee column, including metadata and missing values.
    # Separate products and menus remain separate modelling observations.
    other_columns = [
        column
        for column in hist_enriched_saa.columns
        if column not in model_grain + [fee_column]
    ]
    consistency = (
        hist_enriched_saa
        .groupby(model_grain, dropna=False)[other_columns]
        .nunique(dropna=False)
    )

    varying_columns = consistency.columns[
        consistency.gt(1).any()
    ].tolist()

    if varying_columns:
        raise ValueError(
            "Cannot collapse fee arrangements: conflicting non-fee columns "
            f"within the modelling grain: {varying_columns}. "
            "Review these records first."
        )

    # Count distinct arrangements, matching the notebook's treatment of missing IDs.
    fee_counts = (
        hist_enriched_saa
        .groupby(model_grain, dropna=False)[fee_column]
        .nunique(dropna=False)
        .rename("fee_arrangement_count")
        .reset_index()
    )

    # Retain one identical record per modelling key and attach its fee count.
    model_base = (
        hist_enriched_saa
        .drop(columns=fee_column)
        .drop_duplicates(subset=model_grain)
        .merge(
            fee_counts,
            on=model_grain,
            how="left",
            validate="one_to_one",
        )
        .reset_index(drop=True)
    )

    print(
        f"Modelling base: {len(hist_enriched_saa):,} arrangement rows -> "
        f"{len(model_base):,} product-menu-option-quarter rows."
    )
    return model_base

# Select MySuper using the audit notebook's exact rule.
def select_mysuper_population(hist_model_base):
    model_grain = ENTITY_KEYS + ["period_end_date"]

    check_columns(
        hist_model_base,
        model_grain + ["product_type", "product_phase"],
        "hist_model_base",
    )

    # Create a separate MySuper dataframe, preserving every existing column.
    # Missing predictors or unavailable future targets do not remove rows here.
    mysuper = (
        hist_model_base.loc[
            hist_model_base["product_type"].eq("MySuper Product")
        ]
        .copy()
        .reset_index(drop=True)
    )

    # Stop if the expected label is absent or modelling keys are duplicated.
    if mysuper.empty:
        raise ValueError(
            "No 'MySuper Product' rows found. "
            "Check the source product_type labels."
        )

    if mysuper.duplicated(subset=model_grain).any():
        raise ValueError(
            "MySuper records are duplicated at the established modelling grain."
        )

    # The notebook's MySuper population was entirely Accumulation.
    # Check that assumption for each edition without silently filtering phases.
    if not mysuper["product_phase"].eq("Accumulation").fillna(False).all():
        raise ValueError(
            "MySuper includes missing or non-Accumulation product phases. "
            "Review the population before continuing."
        )

    print(
        f"MySuper population: {len(mysuper):,} rows retained from "
        f"{len(hist_model_base):,} modelling rows; all are Accumulation."
    )

    return mysuper

# STEP 16: Calculate each representation's next-four-quarter return.
def add_future_4q_returns(hist_model_mysuper):
    return_column = "return_measurement_comparison_percent"
    model_grain = ENTITY_KEYS + ["period_end_date"]

    check_columns(
        hist_model_mysuper,
        model_grain + [return_column],
        "hist_model_mysuper",
    )
    if hist_model_mysuper.duplicated(subset=model_grain).any():
        raise ValueError(
            "Future returns require one row per modelling key."
        )

    # Sort within each original product/menu/option history before shifting.
    target_data = (
        hist_model_mysuper.copy()
        .sort_values(model_grain, kind="stable")
        .reset_index(drop=True)
    )
    grouped = target_data.groupby(ENTITY_KEYS, dropna=False)
    current_quarter = target_data["period_end_date"].dt.to_period("Q")
    consecutive = pd.Series(True, index=target_data.index)

    # Confirm that shifted rows really represent the next four calendar quarters.
    for i in range(1, 5):
        target_data[f"return_t_plus_{i}"] = (
            grouped[return_column].shift(-i)
        )
        target_data[f"date_t_plus_{i}"] = (
            grouped["period_end_date"].shift(-i)
        )

        consecutive &= (
            target_data[f"date_t_plus_{i}"].dt.to_period("Q")
            == current_quarter + i
        )

    # Require all four future returns; exclude the current quarter's return.
    return_columns = [f"return_t_plus_{i}" for i in range(1, 5)]
    date_columns = [f"date_t_plus_{i}" for i in range(1, 5)]

    complete_returns = target_data[return_columns].notna().all(axis=1)
    available = consecutive & complete_returns

    target_data["future_4q_target_available"] = available
    target_data["future_4q_cumulative_return"] = np.nan

    target_data.loc[available, "future_4q_cumulative_return"] = (
        (1 + target_data.loc[available, return_columns]).prod(axis=1) - 1
    )

    # Record why this representation cannot supply its own future outcome.
    dates_missing = target_data[date_columns].isna().any(axis=1)

    target_data["future_4q_unavailable_reason"] = np.select(
        [dates_missing, ~consecutive, ~complete_returns],
        [
            "insufficient_future_history",
            "nonconsecutive_quarters",
            "missing_future_return",
        ],
        default="available",
    )

    print(
        f"Representation outcomes: {available.sum():,} available; "
        f"{(~available).sum():,} unavailable. "
        f"All {len(target_data):,} rows retained."
    )

    return target_data


# Construct peer targets using distinct investment options per quarter.
def add_peer_relative_target(hist_model_mysuper_target):
    peer_keys = [
        "rse_abn",
        "abn_investment_option_identifier",
        "period_end_date",
    ]
    return_column = "future_4q_cumulative_return"

    check_columns(
        hist_model_mysuper_target,
        peer_keys + [return_column, "future_4q_target_available"],
        "hist_model_mysuper_target",
    )

    # Only complete future windows can supply peer returns.
    available = hist_model_mysuper_target[
        "future_4q_target_available"
    ].eq(True)

    peer_returns = hist_model_mysuper_target.loc[
        available,
        peer_keys + [return_column],
    ].copy()

    # Verify agreement before retaining one return per option-quarter.
    consistency = (
        peer_returns
        .groupby(peer_keys)[return_column]
        .nunique(dropna=False)
    )

    if consistency.gt(1).any():
        raise ValueError(
            "Conflicting future returns across representations of the same "
            "investment option-quarter. "
            "Review before constructing peer targets."
        )

    peer_returns = peer_returns.drop_duplicates(subset=peer_keys)

    # Each distinct option has one vote in its quarter's peer threshold.
    peer_returns["future_4q_q25_threshold"] = (
        peer_returns
        .groupby("period_end_date")[return_column]
        .transform(lambda values: values.quantile(0.25))
    )

    # Include ties at the threshold, matching the notebook.
    peer_returns["future_4q_bottom_quartile"] = (
        peer_returns[return_column]
        <= peer_returns["future_4q_q25_threshold"]
    )

    # Preserve representation-level outcomes separately from option-level labels.
    target_data = hist_model_mysuper_target.rename(
        columns={
            "future_4q_target_available":
                "future_4q_target_available_representation",
            return_column:
                "future_4q_cumulative_return_representation",
        }
    ).merge(
        peer_returns[
            peer_keys
            + [
                "future_4q_q25_threshold",
                "future_4q_bottom_quartile",
            ]
        ],
        on=peer_keys,
        how="left",
        validate="many_to_one",
        sort=False,
    )

    # A nullable Boolean preserves missing labels for unavailable outcomes.
    target_data["future_4q_bottom_quartile"] = (
        target_data["future_4q_bottom_quartile"].astype("boolean")
    )
    target_data["future_4q_target_available_option"] = (
        target_data["future_4q_bottom_quartile"].notna()
    )

    # Record the outcome-window end date for temporal purging.
    target_data["target_end_date"] = (
        (target_data["period_end_date"].dt.to_period("Q") + 4)
        .dt.to_timestamp(how="end")
        .dt.normalize()
    )

    option_available = target_data["future_4q_target_available_option"]
    target_data.loc[~option_available, "target_end_date"] = pd.NaT

    gained = (
        option_available
        & ~target_data["future_4q_target_available_representation"]
    ).sum()

    print(
        f"Option-level targets: {option_available.sum():,} available; "
        f"{(~option_available).sum():,} unavailable; "
        f"{gained:,} gained through another representation of the same option."
    )

    return target_data