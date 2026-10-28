# Preserve prediction runs: simplified version

Replace predict.py in your project root, beside your existing etl.py and transform.py.
This version does not import workflow_common.py. The existing ETL and fitted model stay as they are.

## Run

Use the same virtual environment used to save your model:

    python predict.py --edition 2026-06

The default prepared input is:

    data/processed/hist_model_mysuper_features_2026-06.parquet

The default saved-model input is:

    outputs/final_rf_reduced29_t035/model.joblib

Optional paths:

    python predict.py --edition 2026-06 --model "D:\models\model.joblib" --raw-dir "D:\data\raw\2026-06"

The edition selects a prepared dataset. It does not download or transform raw data.

## Five files per run

Every execution creates a new folder under outputs/prediction_runs/<run_id>/.
The run ID contains the actual UTC timestamp plus a short random suffix, so rerunning the same edition creates another folder.

| File | What it contains | Later use |
| --- | --- | --- |
| predictions.csv | Original identifiers, available descriptive fields, probabilities, warnings, run/model information and forecast dates | Prediction drift, later outcome evaluation and Power BI |
| run_metadata.json | Run time, edition, source/model paths, model version, threshold, forecast dates and counts | Identify and compare runs |
| scored_snapshot.parquet | Run ID, original modelling keys and the exact 29 inputs used for the latest-quarter scores | Feature distribution and missingness monitoring |
| saa_detail.csv | Raw SAA components for the scored economic options and origin quarter | Allocation, domicile, listing and sector breakdowns |
| specification.json | The specification supplied with the saved fitted model, including ordered predictors and threshold | Document the model settings used |

Descriptive fields are inside predictions.csv; they do not require another file. All available non-future fields from the prepared latest-quarter rows are retained, including names, investment_menu_type, investment_option_type, investment_option_category, life_cycle_name and open_to_new_mems when supplied.

The scored snapshot contains only the latest-quarter keys and selected predictors, rather than the full historical dataset. Native missing values are preserved.

The script creates no model registry, full-history snapshot, duplicate prediction Parquet file or file-integrity manifest. A single fingerprint in the run metadata identifies the saved model version. The model specification describes the fitted model but does not contain the fitted trees: keep your existing model.joblib, and save a new model version separately if you retrain.

Earlier run folders and legacy flat prediction CSVs are left unchanged. Older archives created by the previous script are not converted or deleted. Later consolidation should read the new run_metadata.json format and explicitly support older manifest.json archives if you want to include them.

## SAA detail and descriptive fields

By default, raw SAA is read from data/raw/apra/<edition>/.
The folder must contain one CSV with both historical and saa in its filename.

If the default raw folder is absent, prediction still runs. saa_detail.csv then contains headers only, and run_metadata.json records status not_supplied.
An explicitly supplied raw folder that does not exist produces an error.

A few shared allocation totals are checked against the prepared input before SAA detail is attached. This helps catch the use of a different edition/revision; it is not an exhaustive equality check of every raw SAA field.

SAA detail has several rows per option because it describes allocation components. It is not joined directly onto every prediction row. Use run_id, rse_abn, the option identifier and quarter to connect it to a unique option/run table during later dashboard preparation.
The SAA names investment_option_identifier and time_key correspond to abn_investment_option_identifier and period_end_date in predictions.

Domicile describes components, not one domicile for the entire option. Currency Exposure rows are retained; exclude them when plotting asset-allocation totals. Do not normalize reported allocations automatically.

## Which file supports which monitoring check?

- Prediction drift: compare predicted_probability distributions and warning rates from predictions.csv across origin quarters, using a consistent model version and threshold.
- Input drift: compare the 29 feature distributions and missingness in scored_snapshot.parquet against a chosen reference.
- Prediction performance: compare the original archived probabilities/warnings with actual targets from a later prepared dataset, once the four-quarter outcomes are available.

A shift in predicted probabilities does not by itself establish that predictive performance has deteriorated.
Keep reruns of the same origin quarter identifiable; they should not be counted as additional independent periods.

The original modelling grain is preserved: fund, product, menu, investment option and quarter. Several representations of the same economic option may therefore have separate predictions. The distinct option count uses fund ABN plus option identifier.

## Dates and completion

period_end_date is the origin quarter. forecast_start_date and forecast_end_date describe the following four-quarter return window.
prediction_generated_at_utc is the actual issue time; it is not backdated to the origin quarter.
A known APRA publication date can optionally be recorded with --publication-date YYYY-MM-DD. Recording it does not certify that every input was available at that date.

Metadata is written last, after the other four files. Future readers should include only folders with run_metadata.json and status complete.
If a normal write fails, the newly created folder is removed. An abruptly terminated process can leave a partial folder without metadata; leave it out of consolidation.

## What this update does

This script scores and archives. It does not train, retune or evaluate the model.
The selected Reduced 29 feature order, saved threshold 0.35 and original source data remain unchanged.

Monitoring and the consolidated Power BI export are separate later stages. Power BI can eventually read stable consolidated tables built from these archives, while each original run remains preserved.

