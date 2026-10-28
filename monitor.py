"""Monitor saved MySuper predictions without retraining or generating new scores."""

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from transform import ENTITY_KEYS, check_columns, convert_dates

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_GRAIN = ENTITY_KEYS + ["period_end_date"]
OPTION_GRAIN = ["rse_abn", "abn_investment_option_identifier", "period_end_date"]
TARGET = "future_4q_bottom_quartile"
AVAILABLE = "future_4q_target_available_option"

# Review guides for this project, not statistical significance or retraining rules.
# Shares use 0-1 units: 0.10 is a ten-percentage-point change.
REVIEW_LIMITS = {
    "probability_distance": 0.05,
    "warning_rate_change": 0.10,
    "missing_share_change": 0.10,
    "scaled_feature_distance": 0.50,
}


# Small file helpers; no workflow_common.py or saved model is needed here.
def write_csv(data, path):
    exported = data.copy()
    for column in exported:
        if pd.api.types.is_bool_dtype(exported[column].dtype):
            exported[column] = exported[column].astype("Int64")
    exported.to_csv(path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")


def prepare_keys(data, keys, label):
    check_columns(data, keys, label)
    result = data.copy()
    result["period_end_date"] = convert_dates(result["period_end_date"])
    for column in set(keys) & set(ENTITY_KEYS):
        result[column] = result[column].astype("string")
    if result[keys].isna().any().any() or result.duplicated(keys).any():
        raise ValueError(f"{label}: missing or duplicate keys.")
    return result


# Read completed archives and verify that scores, inputs and metadata agree.
# Both the simplified run_metadata.json and earlier manifest.json are supported.
def read_runs(project_root):
    parent = Path(project_root) / "outputs" / "prediction_runs"
    runs, skipped = [], []
    for folder in sorted(parent.iterdir()) if parent.is_dir() else []:
        if not folder.is_dir():
            continue
        marker = folder / "run_metadata.json"
        if not marker.is_file():
            marker = folder / "manifest.json"
        if folder.name.endswith(".pending") or not marker.is_file():
            skipped.append(folder.name)
            continue
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        if metadata.get("status") != "complete":
            skipped.append(folder.name)
            continue
        spec = json.loads((folder / "specification.json").read_text(encoding="utf-8"))
        features = list(metadata["features"])
        if not features or features != spec["features"] or not np.isclose(metadata["threshold"], spec["threshold"]):
            raise ValueError(f"{folder.name}: specification differs from run metadata.")
        if spec.get("target", TARGET) != TARGET:
            raise ValueError(f"{folder.name}: unexpected target definition.")
        predictions = pd.read_csv(folder / "predictions.csv", dtype={c: "string" for c in ENTITY_KEYS})
        check_columns(predictions, MODEL_GRAIN + [
            "run_id", "model_version", "predicted_probability", "warning", "threshold", "forecast_end_date",
        ], folder.name)
        predictions = prepare_keys(predictions, MODEL_GRAIN, folder.name)
        snapshot = prepare_keys(pd.read_parquet(folder / "scored_snapshot.parquet"),
                                MODEL_GRAIN, folder.name + " scored snapshot")
        check_columns(snapshot, features, "Scored inputs")
        if predictions.empty or len(predictions) != metadata["rows"]:
            raise ValueError(f"{folder.name}: empty predictions or unexpected row count.")
        matched = predictions[MODEL_GRAIN].merge(snapshot[MODEL_GRAIN], on=MODEL_GRAIN,
                                                how="outer", indicator=True, validate="one_to_one")
        if not matched["_merge"].eq("both").all():
            raise ValueError(f"{folder.name}: scores and feature snapshot have different records.")
        if any(not pd.api.types.is_numeric_dtype(snapshot[c]) for c in features):
            raise ValueError(f"{folder.name}: nonnumeric scored inputs.")
        if np.isinf(snapshot[features].to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError(f"{folder.name}: infinite scored inputs.")
        origin = pd.Timestamp(metadata["origin_date"])
        expected_end = (origin.to_period("Q") + 4).end_time.normalize()
        predictions["forecast_end_date"] = pd.to_datetime(predictions["forecast_end_date"], errors="raise")
        if not predictions["period_end_date"].eq(origin).all() or not predictions["forecast_end_date"].eq(expected_end).all():
            raise ValueError(f"{folder.name}: origin or forecast dates disagree with the metadata.")
        if not predictions["run_id"].eq(metadata["run_id"]).all() or not predictions["model_version"].eq(metadata["model_version"]).all():
            raise ValueError(f"{folder.name}: run/model identifiers disagree.")
        for column in ["predicted_probability", "warning", "threshold"]:
            predictions[column] = pd.to_numeric(predictions[column], errors="raise")
        p = predictions["predicted_probability"]
        threshold = float(metadata["threshold"])
        if not np.isfinite(p).all() or not p.between(0, 1).all():
            raise ValueError(f"{folder.name}: invalid probabilities.")
        if not predictions["warning"].isin([0, 1]).all() or not predictions["warning"].eq(p.ge(threshold).astype(int)).all():
            raise ValueError(f"{folder.name}: warnings do not match the archived threshold.")
        if not np.isclose(predictions["threshold"], threshold).all():
            raise ValueError(f"{folder.name}: inconsistent thresholds.")
        issued = pd.to_datetime(metadata["prediction_generated_at_utc"], utc=True)
        runs.append({"metadata": metadata, "predictions": predictions, "snapshot": snapshot,
                     "origin": origin, "issued": issued, "features": features})
    if not runs:
        raise ValueError("No completed prediction archives found. Run predict.py first.")
    runs.sort(key=lambda r: (r["issued"], r["metadata"]["run_id"]))
    seen, versions = set(), {}
    for run in runs:
        meta = run["metadata"]
        key = (meta["model_version"], meta["origin_date"])
        run["first_for_origin"] = key not in seen
        seen.add(key)
        settings = (tuple(run["features"]), float(meta["threshold"]))
        if versions.setdefault(meta["model_version"], settings) != settings:
            raise ValueError("The same model version has conflicting feature lists or thresholds.")
    return runs, skipped


# Compare probabilities and warning volume with a previous origin quarter.
def prediction_summary(current, reference):
    data = current["predictions"]
    row = {"rows": len(data), "distinct_options": len(data[OPTION_GRAIN[:2]].drop_duplicates()),
           "mean_probability": data["predicted_probability"].mean(),
           "median_probability": data["predicted_probability"].median(),
           "p10_probability": data["predicted_probability"].quantile(.10),
           "p90_probability": data["predicted_probability"].quantile(.90),
           "warning_rate": data["warning"].mean(), "reference_rows": 0,
           "reference_warning_rate": np.nan, "probability_wasserstein": np.nan,
           "warning_rate_change": np.nan, "review_required": pd.NA,
           "comparison_status": "no_prior_origin_same_model"}
    if reference is not None:
        prior = reference["predictions"]
        distance = wasserstein_distance(prior["predicted_probability"], data["predicted_probability"])
        change = data["warning"].mean() - prior["warning"].mean()
        row.update(reference_rows=len(prior), reference_warning_rate=prior["warning"].mean(),
                   probability_wasserstein=distance, warning_rate_change=change,
                   review_required=bool(distance >= REVIEW_LIMITS["probability_distance"]
                                        or abs(change) >= REVIEW_LIMITS["warning_rate_change"]),
                   comparison_status="previous_available_origin_same_model")
    return row


# Compare observed feature distributions and missingness without imputation.
# IQR scaling only makes the distance interpretable; model inputs are not transformed.
def feature_summary(current, reference, features):
    rows = []
    for column in features:
        values = current[column].to_numpy(dtype=float, na_value=np.nan)
        observed = values[np.isfinite(values)]
        row = {"feature": column, "current_rows": len(values), "current_observed_rows": len(observed),
               "current_missing_share": float(np.isnan(values).mean()),
               "current_median": float(np.median(observed)) if len(observed) else np.nan,
               "current_p10": float(np.quantile(observed, .10)) if len(observed) else np.nan,
               "current_p90": float(np.quantile(observed, .90)) if len(observed) else np.nan,
               "reference_rows": 0, "reference_observed_rows": 0, "reference_missing_share": np.nan,
               "reference_median": np.nan, "missing_share_change": np.nan,
               "wasserstein_distance": np.nan, "reference_scale": np.nan,
               "scaled_wasserstein": np.nan, "review_required": pd.NA,
               "comparison_status": "no_prior_origin_same_model", "review_reason": ""}
        if reference is not None:
            prior = reference[column].to_numpy(dtype=float, na_value=np.nan)
            prior_observed = prior[np.isfinite(prior)]
            change = float(np.isnan(values).mean() - np.isnan(prior).mean())
            reasons = ["missingness_shift"] if abs(change) >= REVIEW_LIMITS["missing_share_change"] else []
            row.update(reference_rows=len(prior), reference_observed_rows=len(prior_observed),
                       reference_missing_share=float(np.isnan(prior).mean()),
                       reference_median=float(np.median(prior_observed)) if len(prior_observed) else np.nan,
                       missing_share_change=change, comparison_status="insufficient_observed_values")
            if len(observed) and len(prior_observed):
                distance = float(wasserstein_distance(prior_observed, observed))
                scale = float(np.quantile(prior_observed, .75) - np.quantile(prior_observed, .25))
                if scale == 0:
                    scale = float(np.std(prior_observed))
                scaled = distance / scale if scale > 0 else (0.0 if distance == 0 else np.nan)
                row.update(wasserstein_distance=distance, reference_scale=scale,
                           scaled_wasserstein=scaled, comparison_status="previous_available_origin_same_model")
                if scale > 0 and scaled >= REVIEW_LIMITS["scaled_feature_distance"]:
                    reasons.append("distribution_shift")
                if scale == 0 and not np.isclose(observed, prior_observed[0], atol=1e-12, rtol=1e-9).all():
                    reasons.append("constant_reference_changed")
                row["review_required"] = bool(reasons)
            elif reasons:
                row["review_required"] = True
            row["review_reason"] = ";".join(reasons)
        rows.append(row)
    return pd.DataFrame(rows)


# Read later option-level outcomes, retaining unknown labels as missing.
def read_outcomes(contents):
    data = prepare_keys(pd.read_parquet(io.BytesIO(contents)), MODEL_GRAIN, "Outcome data")
    if data.empty:
        raise ValueError("Outcome data is empty.")
    check_columns(data, [TARGET, AVAILABLE, "target_end_date"], "Outcome data")
    coverage_end = data["period_end_date"].max()
    labels = data[OPTION_GRAIN + [TARGET, AVAILABLE, "target_end_date"]].rename(columns={
        TARGET: "actual_target", AVAILABLE: "actual_target_available", "target_end_date": "actual_target_end_date",
    }).copy()
    for column in ["actual_target", "actual_target_available"]:
        values = pd.to_numeric(labels[column], errors="raise")
        if not values.dropna().isin([0, 1]).all():
            raise ValueError(f"Outcome data: {column} must be binary or missing.")
        labels[column] = values.astype("Int64")
    labels["actual_target_end_date"] = pd.to_datetime(labels["actual_target_end_date"], errors="raise")
    available = labels["actual_target_available"].eq(1)
    if available.isna().any() or not available.eq(labels["actual_target"].notna()).all():
        raise ValueError("Option-level availability disagrees with the actual target.")
    expected_end = (labels["period_end_date"].dt.to_period("Q") + 4).dt.to_timestamp(how="end").dt.normalize()
    if not labels.loc[available, "actual_target_end_date"].eq(expected_end.loc[available]).all():
        raise ValueError("Available targets have an unexpected four-quarter end date.")
    if labels.loc[available, "actual_target_end_date"].gt(coverage_end).any():
        raise ValueError("Available targets extend beyond the outcome data's reporting coverage.")
    if labels.loc[~available, "actual_target_end_date"].notna().any():
        raise ValueError("Unavailable targets unexpectedly have a target end date.")
    if labels.groupby(OPTION_GRAIN)[["actual_target", "actual_target_available", "actual_target_end_date"]].nunique(dropna=False).gt(1).any().any():
        raise ValueError("Conflicting outcomes across representations of the same option-quarter.")
    labels = labels.drop_duplicates(OPTION_GRAIN)
    # A representation with a complete outcome may supply the realised return.
    return_column = "future_4q_cumulative_return_representation"
    if return_column in data:
        returns = data[OPTION_GRAIN + [return_column]].dropna(subset=[return_column])
        if returns.groupby(OPTION_GRAIN)[return_column].nunique().gt(1).any():
            raise ValueError("Conflicting realised returns for the same option-quarter.")
        labels = labels.merge(returns.drop_duplicates(OPTION_GRAIN).rename(
            columns={return_column: "actual_future_4q_return"}), on=OPTION_GRAIN, how="left", validate="one_to_one")
    else:
        labels["actual_future_4q_return"] = np.nan
    return labels, coverage_end


# Match labels to the original predictions; never call predict_proba again.
def evaluate_predictions(predictions, labels, coverage_end):
    result = predictions.merge(labels, on=OPTION_GRAIN, how="left", validate="many_to_one", indicator="_label_join")
    mature = result["forecast_end_date"].le(coverage_end)
    known = result["actual_target_available"].eq(1).fillna(False).astype(bool)
    if not result.loc[known, "actual_target_end_date"].eq(result.loc[known, "forecast_end_date"]).all():
        raise ValueError("The actual target end date differs from its archived forecast end date.")
    result["evaluation_status"] = np.select(
        [~mature, result["_label_join"].eq("left_only"), known],
        ["pending_window", "option_not_in_label_snapshot", "label_available"],
        default="mature_unavailable",
    )
    return result.drop(columns="_label_join")


# Calculate performance only where actual labels are available.
# Undefined metrics remain blank, including ROC AUC when only one class is present.
def performance_metrics(evaluated, coverage_end):
    known = evaluated.loc[evaluated["evaluation_status"].eq("label_available")]
    mature = evaluated["forecast_end_date"].le(coverage_end)
    metrics = {"issued_rows": len(evaluated), "mature_rows": int(mature.sum()), "labelled_rows": len(known),
               "pending_rows": int(evaluated["evaluation_status"].eq("pending_window").sum()),
               "unavailable_mature_rows": int((mature & ~evaluated["evaluation_status"].eq("label_available")).sum()),
               "label_coverage": len(known) / int(mature.sum()) if mature.any() else np.nan,
               "positive_rate": np.nan, "labelled_warning_rate": np.nan, "accuracy": np.nan,
               "balanced_accuracy": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan,
               "roc_auc": np.nan, "average_precision": np.nan, "brier_score": np.nan,
               "true_positive": 0, "false_positive": 0, "true_negative": 0, "false_negative": 0}
    if known.empty:
        return metrics
    y, p, w = known["actual_target"].astype(int).to_numpy(), known["predicted_probability"].to_numpy(), known["warning"].to_numpy()
    tp, fp = int(((y == 1) & (w == 1)).sum()), int(((y == 0) & (w == 1)).sum())
    tn, fn = int(((y == 0) & (w == 0)).sum()), int(((y == 1) & (w == 0)).sum())
    recall = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    metrics.update(true_positive=tp, false_positive=fp, true_negative=tn, false_negative=fn,
                   positive_rate=float(y.mean()), labelled_warning_rate=float(w.mean()),
                   accuracy=(tp + tn) / len(y), precision=tp / (tp + fp) if tp + fp else np.nan,
                   recall=recall, balanced_accuracy=(recall + specificity) / 2,
                   f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan,
                   roc_auc=roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan,
                   average_precision=average_precision_score(y, p) if y.sum() else np.nan,
                   brier_score=brier_score_loss(y, p))
    return metrics


# Evaluate all completed runs against this outcome edition and save a new batch.
def run_monitoring(project_root, edition):
    project_root = Path(project_root)
    runs, skipped = read_runs(project_root)
    label_path = project_root / "data" / "processed" / f"hist_model_mysuper_features_{edition}.parquet"
    if not label_path.is_file():
        raise FileNotFoundError(f"Prepared outcome data not found: {label_path}")
    contents = label_path.read_bytes()
    labels, coverage_end = read_outcomes(contents)
    generated = datetime.now(timezone.utc)
    monitoring_id = generated.strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid4().hex[:8]
    predictions, features, evaluations, performance = [], [], [], []
    for run in runs:
        meta = run["metadata"]
        # A rerun of the same quarter is not a new reference quarter.
        # Do not use a reference that was issued after the run being assessed.
        candidates = [r for r in runs if r["first_for_origin"]
                      and r["metadata"]["model_version"] == meta["model_version"]
                      and r["origin"] < run["origin"] and r["issued"] < run["issued"]]
        reference = max(candidates, key=lambda r: r["origin"]) if candidates else None
        common = {
            "monitoring_id": monitoring_id, "run_id": meta["run_id"], "model_version": meta["model_version"],
            "model_name": meta["model_name"], "source_edition": meta["source_edition"],
            "origin_date": meta["origin_date"], "threshold": meta["threshold"],
            "is_first_run_for_origin": run["first_for_origin"],
        }
        reference_info = {
            "reference_run_id": reference["metadata"]["run_id"] if reference else None,
            "reference_origin_date": reference["metadata"]["origin_date"] if reference else None,
            "origin_quarters_apart": run["origin"].to_period("Q").ordinal - reference["origin"].to_period("Q").ordinal if reference else np.nan,
        }
        predictions.append({**common, **reference_info, **prediction_summary(run, reference)})
        table = feature_summary(run["snapshot"], reference["snapshot"] if reference else None, run["features"])
        for column, value in {**common, **reference_info}.items():
            table[column] = value
        features.append(table)
        evaluated = evaluate_predictions(run["predictions"], labels, coverage_end)
        for column, value in common.items():
            evaluated[column] = value
        evaluated["outcome_edition"] = edition
        evaluated["evaluated_at_utc"] = generated.isoformat()
        issue_day = run["issued"].tz_localize(None).normalize()
        evaluated["issued_after_forecast_end"] = issue_day > evaluated["forecast_end_date"]
        evaluations.append(evaluated)
        performance.append({**common, "outcome_edition": edition, "evaluated_at_utc": generated.isoformat(),
                            "forecast_end_date": str(run["predictions"]["forecast_end_date"].iloc[0].date()),
                            "issued_after_forecast_end": bool(evaluated["issued_after_forecast_end"].iloc[0]),
                            **performance_metrics(evaluated, coverage_end)})
    tables = {
        "prediction_drift": pd.DataFrame(predictions),
        "feature_drift": pd.concat(features, ignore_index=True),
        "evaluated_predictions": pd.concat(evaluations, ignore_index=True),
        "performance": pd.DataFrame(performance),
    }
    for name in ["prediction_drift", "feature_drift"]:
        tables[name]["review_required"] = tables[name]["review_required"].astype("boolean")
    destination = project_root / "outputs" / "monitoring_runs" / monitoring_id
    destination.mkdir(parents=True, exist_ok=False)
    try:
        for name, table in tables.items():
            write_csv(table, destination / f"{name}.csv")
        metadata = {
            "status": "complete", "monitoring_id": monitoring_id, "evaluated_at_utc": generated.isoformat(),
            "outcome_edition": edition, "outcome_source": str(label_path.resolve()),
            "outcome_source_sha256": hashlib.sha256(contents).hexdigest(),
            "label_coverage_end": str(coverage_end.date()), "metric_grain": "modelling_rows",
            "drift_reference": "previous_available_origin_same_model",
            "run_ids": [r["metadata"]["run_id"] for r in runs], "skipped_folders": skipped,
            "review_limits": REVIEW_LIMITS, "review_limits_are_heuristics": True,
            "automatic_retraining": False, "point_in_time_availability_verified": False,
        }
        (destination / "monitoring_metadata.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    except BaseException:
        shutil.rmtree(destination)
        raise
    print(f"Monitoring saved: {destination}")
    print(f"Assessed {len(runs)} archived runs; outcome data through {coverage_end.date()}.")
    print(tables["performance"][["run_id", "labelled_rows", "pending_rows", "unavailable_mature_rows", "roc_auc", "average_precision"]].to_string(index=False))
    if skipped:
        print(f"Skipped {len(skipped)} incomplete/unrecognised folders; see monitoring_metadata.json.")
    return destination


# Supply the latest prepared edition containing the outcomes currently known.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edition", required=True, help="Prepared edition to use for actual outcomes")
    args = parser.parse_args()
    run_monitoring(PROJECT_ROOT, args.edition)


if __name__ == "__main__":
    main()

