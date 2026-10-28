"""Score the latest MySuper quarter and save a separate folder for each run."""

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import sklearn

from transform import (
    ENTITY_KEYS, REDUCED_FEATURE_COLS, SAA_ENTITY_KEYS, SAA_REQUIRED_COLUMNS,
    SAA_NUMERIC_COLUMNS, check_columns, clean_abn, clean_identifier,
    convert_dates, convert_numbers, aggregate_saa_allocations,
    add_saa_domicile_and_currency,
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_GRAIN = ENTITY_KEYS + ["period_end_date"]
OPTION_GRAIN = ["rse_abn", "abn_investment_option_identifier", "period_end_date"]
SAA_DETAIL_COLUMNS = [
    "rse_abn", "investment_option_identifier", "time_key", "investment_option_name",
    "investment_option_type", "investment_option_category", "rse_name", "rsl_name",
    "Public_Offer_Status", "ConsolidatedSectorType", "ConsolidatedListingType",
    "ConsolidatedDomicileType", "ConsolidatedInternationalEconomyType",
    "InvestmentStrategicSectorType", "InvestmentStrategicSectorListingType",
    "InvestmentStrategicSectorDomicileType", "InvestmentStrategicSectorInternationalEconomyType",
    "InvestmentStrategicSubsectorType", "InvestmentStrategicSubsectorListingType",
    "InvestmentStrategicSubsectorDomicileType", "InvestmentStrategicSubsectorInternationalEconomyType",
    "InvestmentBenchmarkAllocationPercent", "InvestmentCurrencyHedgingRatioPercent",
]


# Save readable JSON and CSV files. Keep IDs as text when importing CSVs.
def write_json(path, values):
    path.write_text(json.dumps(values, indent=2, default=str, allow_nan=False), encoding="utf-8")


def write_csv(data, path):
    exported = data.copy()
    for column in exported:
        if pd.api.types.is_bool_dtype(exported[column].dtype):
            exported[column] = exported[column].astype("Int64")
        elif isinstance(exported[column].dtype, pd.PeriodDtype):
            exported[column] = exported[column].astype("string")
    exported.to_csv(path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")


# Preserve matching SAA components separately so they do not multiply scores.
# Domicile describes each allocation component, rather than the whole option.
def get_saa_detail(raw_dir, latest, run_id):
    if raw_dir is None:
        return pd.DataFrame(columns=["run_id"] + SAA_DETAIL_COLUMNS), {
            "status": "not_supplied", "rows": 0, "source_file": None,
        }
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw SAA folder not found: {raw_dir}")
    files = [p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"
             and "historical" in p.name.lower() and "saa" in p.name.lower()]
    if len(files) != 1:
        raise ValueError(f"Expected one historical SAA CSV in {raw_dir}; found {len(files)}.")
    saa = pd.read_csv(files[0], dtype="string")
    check_columns(saa, SAA_REQUIRED_COLUMNS, "Raw SAA detail")
    saa["rse_abn"] = clean_abn(saa["rse_abn"])
    saa["investment_option_identifier"] = clean_identifier(saa["investment_option_identifier"])
    saa["time_key"] = convert_dates(saa["time_key"])
    convert_numbers(saa, SAA_NUMERIC_COLUMNS, "Raw SAA detail")
    saa = saa.loc[saa["time_key"].eq(latest["period_end_date"].max())].copy()

    # Check a few shared allocation totals to catch a mismatched edition/revision.
    # These checks do not recalculate or replace the model's prepared features.
    compare = [c for c in ["total_benchmark_allocation", "allocation_equity", "allocation_cash",
               "allocation_australian_domicile", "allocation_international_domicile"] if c in latest]
    if saa.empty:
        if latest[compare].notna().any().any():
            raise ValueError("Raw SAA has no latest-quarter rows although prepared allocations exist.")
    else:
        aggregate = add_saa_domicile_and_currency(aggregate_saa_allocations(saa), saa).rename(
            columns={"investment_option_identifier": "abn_investment_option_identifier",
                     "time_key": "period_end_date"})
        checked = latest[OPTION_GRAIN + compare].drop_duplicates(OPTION_GRAIN).merge(
            aggregate[OPTION_GRAIN + compare], on=OPTION_GRAIN, how="left",
            validate="one_to_one", suffixes=("_prepared", "_raw"))
        for column in compare:
            left = checked[column + "_prepared"].to_numpy(dtype=float, na_value=np.nan)
            right = checked[column + "_raw"].to_numpy(dtype=float, na_value=np.nan)
            if not np.isclose(left, right, atol=1e-8, rtol=1e-8, equal_nan=True).all():
                raise ValueError(f"Raw SAA differs from prepared {column}; use the same edition/revision.")

    # Keep one set of SAA components per economic option, even with several menus.
    lookup = latest[OPTION_GRAIN].drop_duplicates().rename(columns={
        "abn_investment_option_identifier": "investment_option_identifier", "period_end_date": "time_key",
    })
    detail = saa.merge(lookup, on=SAA_ENTITY_KEYS + ["time_key"], how="inner", validate="many_to_one")
    detail = detail.reindex(columns=list(dict.fromkeys(SAA_DETAIL_COLUMNS + list(detail.columns))))
    detail.insert(0, "run_id", run_id)
    return detail, {"status": "archived", "rows": len(detail), "source_file": str(files[0].resolve())}


# Load the existing fitted model and check its fixed feature order/threshold.
def create_prediction_run(project_root, edition, model_path=None, raw_dir=None, publication_date=None):
    project_root = Path(project_root)
    model_path = Path(model_path) if model_path else project_root / "outputs" / "final_rf_reduced29_t035" / "model.joblib"
    data_path = project_root / "data" / "processed" / f"hist_model_mysuper_features_{edition}.parquet"
    for label, path in [("Prepared data", data_path), ("Saved model", model_path)]:
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    model_bytes = model_path.read_bytes()
    bundle = joblib.load(io.BytesIO(model_bytes))
    if not isinstance(bundle, dict) or not {"model", "specification"}.issubset(bundle):
        raise ValueError("Expected a saved bundle containing model and specification.")
    model, specification = bundle["model"], bundle["specification"]
    features, threshold = list(specification["features"]), float(specification["threshold"])
    if features != REDUCED_FEATURE_COLS or not np.isclose(threshold, 0.35):
        raise ValueError("Expected Reduced 29 at threshold 0.35.")
    if list(model.feature_names_in_) != features or set(model.classes_) != {0, 1}:
        raise ValueError("Saved model columns/classes disagree with its specification.")
    if specification["sklearn_version"] != sklearn.__version__:
        raise ValueError("Use the same scikit-learn version as the saved training environment.")
    # A fingerprint identifies the saved model; no separate model registry is created.
    model_version = hashlib.sha256(model_bytes).hexdigest()

    # STEP 4: Select the latest quarter and score only the saved 29 predictors.
    data = pd.read_parquet(data_path)
    check_columns(data, MODEL_GRAIN + features, "Prepared data")
    if data.empty:
        raise ValueError("Prepared data is empty.")
    data["period_end_date"] = convert_dates(data["period_end_date"])
    if data[MODEL_GRAIN].isna().any().any() or data.duplicated(MODEL_GRAIN).any():
        raise ValueError("Prepared data has missing or duplicate modelling keys.")
    for column in ENTITY_KEYS:
        data[column] = data[column].astype("string")
    origin = data["period_end_date"].max()
    if specification.get("training_cutoff") and origin < pd.Timestamp(specification["training_cutoff"]):
        raise ValueError("Origin predates the model's training cutoff; use a temporal backtest instead.")
    latest = data.loc[data["period_end_date"].eq(origin)].sort_values(
        ["period_end_date"] + ENTITY_KEYS).reset_index(drop=True)
    x = latest[features].copy()
    if any(not pd.api.types.is_numeric_dtype(x[c]) for c in features):
        raise ValueError("All selected predictors must be numeric or Boolean.")
    if np.isinf(x.to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("Infinite predictor values found; review the ETL output.")
    positive_column = int(np.flatnonzero(model.classes_ == 1)[0])
    probabilities = model.predict_proba(x)[:, positive_column]

    # STEP 5: Prepare predictions, descriptive fields, and the exact scored inputs.
    generated_at = datetime.now(timezone.utc)
    run_id = generated_at.strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid4().hex[:8]
    forecast_start = (origin.to_period("Q") + 1).start_time.normalize()
    forecast_end = (origin.to_period("Q") + 4).end_time.normalize()
    future_columns = [c for c in latest if c.startswith(
        ("future_", "return_t_plus_", "date_t_plus_")) or c == "target_end_date"]
    predictions = latest.drop(columns=future_columns).copy()
    # Available names, menu/option types and other non-future fields stay in this CSV.
    predictions["run_id"] = run_id
    predictions["predicted_probability"] = probabilities
    predictions["warning"] = (probabilities >= threshold).astype("int8")
    predictions["threshold"] = threshold
    predictions["model_name"] = specification["model"]
    predictions["model_version"] = model_version
    predictions["source_edition"] = edition
    predictions["forecast_start_date"] = forecast_start
    predictions["forecast_end_date"] = forecast_end
    predictions["prediction_generated_at_utc"] = generated_at.isoformat()
    predictions = predictions.sort_values("predicted_probability", ascending=False, kind="stable")
    scored = latest[MODEL_GRAIN + features].copy()
    scored.insert(0, "run_id", run_id)

    if raw_dir is None:
        default_raw = project_root / "data" / "raw" / "apra" / edition
        raw_dir = default_raw if default_raw.is_dir() else None
    detail, detail_info = get_saa_detail(raw_dir, latest, run_id)
    metadata = {
        "status": "complete", "run_id": run_id, "source_edition": edition,
        "prediction_generated_at_utc": generated_at.isoformat(),
        "origin_date": str(origin.date()), "forecast_start_date": str(forecast_start.date()),
        "forecast_end_date": str(forecast_end.date()),
        "source_publication_date": str(pd.Timestamp(publication_date).date()) if publication_date else None,
        "source_dataset": str(data_path.resolve()), "model_file": str(model_path.resolve()),
        "model_name": specification["model"], "model_version": model_version,
        "threshold": threshold, "features": features, "rows": len(predictions),
        "distinct_options": len(latest[OPTION_GRAIN[:2]].drop_duplicates()),
        "warnings": int(predictions["warning"].sum()), "saa_detail": detail_info,
    }

    # Save five files in a new folder. Existing run folders are untouched.
    # Metadata is written last, so later scripts can identify completed runs.
    destination = project_root / "outputs" / "prediction_runs" / run_id
    destination.mkdir(parents=True, exist_ok=False)
    try:
        write_csv(predictions, destination / "predictions.csv")
        scored.to_parquet(destination / "scored_snapshot.parquet", index=False)
        write_csv(detail, destination / "saa_detail.csv")
        write_json(destination / "specification.json", specification)
        write_json(destination / "run_metadata.json", metadata)
    except BaseException:
        shutil.rmtree(destination)
        raise
    print(f"Saved run {run_id}: {len(predictions):,} modelling rows, origin {origin.date()}.")
    print(f"Folder: {destination}")
    if detail_info["status"] == "not_supplied":
        print("Raw SAA detail was not supplied; saa_detail.csv contains column headers only.")
    return destination


# Run this script from the terminal with the same edition used by your ETL.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edition", required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--raw-dir", type=Path, help="Edition folder containing historical SAA CSV")
    parser.add_argument("--publication-date", help="Known source release date, YYYY-MM-DD; optional")
    args = parser.parse_args()
    create_prediction_run(PROJECT_ROOT, args.edition, args.model, args.raw_dir, args.publication_date)


if __name__ == "__main__":
    main()
