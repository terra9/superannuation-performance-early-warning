# Superannuation Performance & Underperformance Early-Warning System

A data analytics and machine-learning project using Australian Prudential Regulation Authority (APRA) data to identify MySuper accumulation investment options at risk of future underperformance.

The selected model estimates whether an option's **compounded return over the next four quarters will fall in the bottom quartile of its MySuper peers**. A warning is issued when the predicted probability is **at least 0.35**.

## Project status

**In progress: modelling completed; scripted prediction and monitoring implemented.**

| Area | Status |
| --- | --- |
| Data auditing, target construction and feature engineering | Developed in the notebooks and extracted into reusable transformation functions |
| Model comparison and selection | Completed; Reduced 29 Random Forest selected, with Full 33 retained as the reference |
| Reserved final-test evaluation | Completed after selecting the feature set and threshold |
| Edition-based ETL and prediction archives | Implemented |
| Prediction drift, input drift and outcome monitoring | Implemented in monitor.py |
| Versioned model download and clean-environment release verification | Pending |
| Consolidated Power BI export and dashboard | Planned |

This is a research prototype. The historical test results do not establish real-time early-warning performance or production readiness.

## Objective and prediction unit

The project investigates whether historical returns, peer-relative performance, volatility, strategic asset allocation, currency hedging and allocation changes provide signals of future bottom-quartile performance.

The modelling grain is one record per:

- Fund ABN: `rse_abn`
- Product: `abn_product_identifier`
- Investment menu: `abn_investment_menu_identifier`
- Investment option: `abn_investment_option_identifier`
- Origin quarter: `period_end_date`

An economic investment option can appear through several product/menu representations. These representations are preserved in the modelling data. Peer targets use distinct fund/option/origin-quarter combinations, so repeated representations do not receive extra votes when calculating the quartile threshold.

The target is `future_4q_bottom_quartile`. It equals 1 when an option's next-four-quarter compounded return is at or below the relevant peer group's 25th-percentile return; otherwise it equals 0. Ties can make the positive share differ from exactly 25%.

Future returns are outcome fields used to construct the target. They are not predictors. Rows whose outcomes are unavailable remain in the full dataset and are excluded only when constructing eligible supervised samples.

## APRA data

Download data from the official [Quarterly Superannuation Product Statistics page](https://www.apra.gov.au/news-and-publications/quarterly-superannuation-product-statistics).

Under **Historical Performance and SAA**, download the two CSV datasets:

| Dataset | Used for |
| --- | --- |
| Historical performance | Returns, volatility, identifiers, descriptive fields and future-outcome construction |
| Historical SAA | Asset allocations, strategic growth allocation, currency hedging and allocation-change features |

The final predictor/target pipeline uses these two historical datasets. Other QSPS tables, including product-structure information used in notebook audits, are not required inputs for the selected model's feature construction.

The model-development snapshot covers **31 December 2014 to 31 March 2026**. Newer APRA editions can be processed separately for prediction and monitoring.

### Store each edition separately

Place both historical CSVs from one APRA edition in:

```text
data/raw/apra/<edition>/
```

For example, the June 2026 edition belongs in `data/raw/apra/2026-06/`. Keep one matching historical performance CSV and one matching historical SAA CSV in that folder.

The `--edition` argument selects that folder. The current workflow uses manually downloaded files; it does not check for or download new APRA releases automatically.

Each historical CSV is a history snapshot. Do not concatenate overlapping editions to extend the time series. Preserve editions separately because APRA may revise historical values, names or coverage. The transformation code handles the observed date formats and missing-value representations while retaining original identifiers and available descriptive fields.

## Selected model and historical evaluation

The selected estimator is scikit-learn's `RandomForestClassifier`, using **29 predictors** and a **0.35 warning threshold**.

| Setting | Value |
| --- | --- |
| Trees | 500 |
| Maximum depth | 6 |
| Minimum samples per leaf | 20 |
| Features considered per split | sqrt |
| Class weights | balanced |
| Random seed | 42 |
| Parallel workers | -1 |
| Missing values | Native Random Forest handling; no added imputation |
| Scaling | No scaler applied to the Random Forest inputs |

The estimator uses balanced class weights; it is not an under-sampling BalancedRandomForestClassifier.

The complete feature dataset is retained. The Reduced 29 predictor list omits these four features from the Full 33 reference:

- `current_return_minus_peer_median`
- `return_trailing_4q`
- `return_min_4q`
- `return_std_4q`

The comparison supports removing this bundle under the fixed Random Forest specification. It does not establish that each omitted feature is individually uninformative.

### Validation design

Four expanding temporal folds cover validation origins from **2019 Q2 to 2023 Q1**, producing **5,220 out-of-fold modelling-row predictions**.

For each fold, training requires both the observation quarter and its future target-window end to be strictly before validation starts. This purges training outcomes that overlap the validation period.

After specification selection, the model was fitted on **9,003 mature pre-test observations**. The reserved test contains **1,369 observations** with origins from **2024 Q2 to 2025 Q1**. That test has already been evaluated; it is not an unused holdout for further tuning.

### Reported results: Reduced 29 at threshold 0.35

| Metric | Pooled temporal validation | Reserved final test |
| --- | ---: | ---: |
| Observations | 5,220 | 1,369 |
| ROC AUC | 0.6881 | 0.7906 |
| Average precision (AP) | 0.4633 | 0.6718 |
| Precision | 0.3882 | 0.4372 |
| Recall | 0.7122 | 0.7723 |
| F1 | 0.5025 | 0.5583 |
| Balanced accuracy | 0.6593 | 0.7174 |
| Brier score | 0.1852 | 0.1568 |
| Warning rate | 47.62% | 44.78% |

These results are recorded in [baseline_modelling.ipynb](notebooks/baseline_modelling.ipynb). The pooled validation row in its final comparison is retained from the completed model-selection comparison.

The test period performed better than pooled validation, but performance varies across historical quarters. Improved average performance did not establish improved temporal stability. Precision and warning volume remain important operating limitations. A lower Brier score indicates lower overall probability error, rather than proving calibration alone.

## Repository guide

| Path | Purpose |
| --- | --- |
| `notebooks/apra_data_audit.ipynb` | Data quality, entity/representation checks, SAA preparation and target construction |
| `notebooks/feature_engineering.ipynb` | Predictor construction and correlation/redundancy audit |
| `notebooks/baseline_modelling.ipynb` | Baselines, model comparisons, temporal validation and final-test evaluation |
| `etl.py` | Command-line entry point for the selected raw edition and prepared-data export |
| `transform.py` | Reusable validation, transformation, target and feature functions |
| `predict.py` | Load the fitted model, score the latest origin quarter and archive the run |
| `monitor.py` | Inspect drift and evaluate archived scores against later outcomes |
| `data/raw/apra/<edition>/` | Original CSVs for one downloaded APRA edition |
| `data/processed/hist_model_mysuper_features_<edition>.parquet` | Full prepared history for that edition |
| `models/rf_reduced29_t035_v1/model.joblib` | Selected fitted model bundle, supplied locally or through a future release |
| `models/rf_reduced29_t035_v1/specification.json` | Readable model specification |
| `outputs/final_rf_reduced29_t035/` | Historical evaluation results |
| `outputs/prediction_runs/` | Preserved prediction runs |
| `outputs/monitoring_runs/` | Preserved monitoring evaluations |
| [PREDICTION_ARCHIVE.md](PREDICTION_ARCHIVE.md) | Prediction archive instructions |
| [MONITORING.md](MONITORING.md) | Monitoring methods, reports and interpretation |

Current command-line entry points live in the project root. Additional folders such as `src/`, `config/`, `docs/` and `tests/` can support further development; the commands below use the root scripts.

The notebooks' original `hist_model_mysuper_features.parquet` remains the development snapshot. Edition-suffixed pipeline outputs are separate files; they do not replace the original notebook dataset automatically.

## Environment and model setup

Use a Python environment compatible with the saved model. Notebook metadata records **Python 3.12.4**; confirm the actual training environment when preparing a release.

Core libraries are pandas, NumPy, SciPy, scikit-learn, joblib and a Parquet engine such as PyArrow. Jupyter is used for the analysis notebooks.

Exact package versions matter when loading a persisted scikit-learn model. The saved specification records scikit-learn, pandas and NumPy versions, and predict.py checks the scikit-learn version before scoring.

### Current model availability

The trained model is excluded from Git alongside generated artefacts. A versioned **GitHub Release is planned**; no published model download is assumed by this README.

Until that release is available, running prediction requires access to the existing saved model bundle. Place it at:

```text
models/rf_reduced29_t035_v1/model.joblib
```

The bundle must contain both the fitted `model` and its `specification`. The JSON specification alone cannot generate predictions.

The modelling notebook's current save cell writes its artefacts to `outputs/final_rf_reduced29_t035/`. If using those existing artefacts, copy `model.joblib` and `specification.json` into `models/rf_reduced29_t035_v1/` before running the command-line prediction script. This does not require retraining.

Downloading the model into this ignored local folder does not require editing predict.py. When a model release is published, use the matching tagged code and the dependency versions supplied with it.

### Dependency file for the release

A pinned `requirements-inference.txt` should be committed with the first packaged release. Generate it from the environment that successfully loads the selected model, then verify it in a fresh environment:

```powershell
python --version
python -m pip freeze > requirements-inference.txt
```

Once that file is available, a fresh Windows PowerShell setup can use:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-inference.txt
```

The installation command assumes the requirements file has been supplied. Until the release setup is verified, use the existing working project environment rather than assuming that arbitrary latest package versions will load the saved model.

## Run the workflow

Run commands from the repository root using the working virtual environment. The examples use `2026-06`; replace it with an edition whose two CSVs you have downloaded.

### 1. Prepare the data

```powershell
python etl.py --edition 2026-06
```

The ETL prepares the historical data, constructs the established MySuper targets and features, and writes:

```text
data/processed/hist_model_mysuper_features_2026-06.parquet
```

All historical rows needed for lagged and rolling calculations are passed through the transformations. Inference is not restricted to rows whose future targets are already available.

A rerun replaces the generated prepared file for that edition. Use a distinct edition folder for a different APRA release.

### 2. Generate and archive predictions

```powershell
python predict.py --edition 2026-06
```

By default, predict.py loads `models/rf_reduced29_t035_v1/model.joblib`. A compatible saved bundle elsewhere can be supplied without editing the script:

```powershell
python predict.py --edition 2026-06 --model models/another_version/model.joblib
```

The script scores only the latest origin quarter in the prepared dataset. It retains product/menu representations, applies the saved feature order and threshold, and creates a new run folder on every execution.

### 3. Monitor predictions and evaluate available outcomes

```powershell
python monitor.py --edition 2026-06
```

Here, the edition identifies the prepared snapshot containing the outcomes currently known. The script reads all completed prediction archives and evaluates their original scores where labels have matured.

Monitoring can be run independently after processing newer data; another prediction run is not required just to reevaluate older forecasts.

These are separate commands. There is currently no single production orchestrator or automatic retraining process.

## Prediction archives and interpretation

Each prediction run is saved under `outputs/prediction_runs/<run_id>/`.

| File | Contents |
| --- | --- |
| `predictions.csv` | Probabilities, warnings, original identifiers, available descriptive fields and forecast dates |
| `run_metadata.json` | Run timestamp, data edition, model version, threshold and counts |
| `scored_snapshot.parquet` | Exact 29 scored inputs and their original modelling keys |
| `saa_detail.csv` | Available raw SAA components for the scored options and quarter |
| `specification.json` | Saved model specification |

Descriptive fields such as fund/product names, investment_menu_type, investment_option_type and investment_option_category are retained when supplied. SAA detail remains separate to avoid multiplying prediction rows. Domicile describes allocation components; an option can have several domicile classifications.

SAA detail is read from the matching raw edition folder. If that folder is absent, its CSV contains headers and metadata records that detail was not supplied. Prepared aggregate allocations are still retained.

For an origin of **2026 Q1**, the forecast covers **2026 Q2 to 2027 Q1**. It concerns the compounded return over that entire future window, rather than the return in the final quarter alone.

`period_end_date` is the observation quarter. `prediction_generated_at_utc` records the actual issue time. Being present in the latest reporting quarter does not necessarily mean an option is open to new members.

## Monitoring reports

Every monitoring execution creates a new folder under `outputs/monitoring_runs/<monitoring_id>/`.

| File | Purpose |
| --- | --- |
| `prediction_drift.csv` | Probability distributions, warning rates and changes from a prior run |
| `feature_drift.csv` | Predictor distributions, observed counts and missingness changes |
| `evaluated_predictions.csv` | Original issued scores, descriptive fields and matched actual outcomes |
| `performance.csv` | Per-run metrics, confusion counts and outcome-coverage counts |
| `monitoring_metadata.json` | Evaluation time, outcome edition/fingerprint and review settings |

Drift comparisons use the previous available origin quarter scored by the same model, taking its first-issued run as the reference. This version compares archived cohorts, not the original training distribution. With only one archived origin quarter, drift comparisons remain blank.

Actual outcomes are matched at the fund/option/origin-quarter level using the established option-level availability flag. Pending or unavailable outcomes are not converted into negative labels.

Use `is_first_run_for_origin` to avoid counting reruns as additional independent origin quarters. Per-run AP and ROC AUC should not be averaged and described as pooled scores.

Distribution-review flags use project-specific heuristic cutoffs. A flag is a prompt to inspect the data and outcomes; it does not prove declining model performance or trigger retraining. Full definitions and limitations are in [MONITORING.md](MONITORING.md).

## Data quality and limitations

- **Publication timing:** quarter-end observations become public later. The origin date is not the date a warning could have been issued. Historical results do not verify that every input was available at an intended real-time warning date.
- **Historical revisions:** later APRA files can revise earlier records. Development results and subsequent monitoring refer to their stated source editions, rather than a reconstructed point-in-time vintage history.
- **Repeated representations:** metrics retain the original modelling rows. Several product/menu records may represent one economic option; reported row counts are not counts of independent options.
- **Incomplete outcomes:** missing future history, gaps and discontinued options can limit target availability. Monitoring reports label coverage so these records remain visible.
- **Merger coverage:** targeted screening and selected-case reviews were completed, rather than exhaustive verification of every fund change. Original histories and identifiers were preserved; unresolved source-history limitations remain.
- **Feature definitions:** return and strategy-change features follow the established notebooks, including observed-row history definitions where used. Quarter gaps and missing coverage affect interpretation.
- **SAA composition:** selected allocation columns are not automatically normalized to sum to one. Domicile and currency-exposure classifications require their own interpretation.
- **Peer-relative objective:** bottom-quartile ranking is not a risk-adjusted performance benchmark or APRA's regulatory performance-test outcome.
- **Temporal variation:** performance differs across quarters, and neighbouring four-quarter outcomes overlap. A single held-out period does not guarantee similar future results.
- **Warning burden:** at the selected threshold, the final test flagged about 45% of modelling rows. Operational usefulness depends on the capacity and cost of reviewing warnings.

## Next development

- Validate the command-line workflow with the actual model and APRA files in a clean environment.
- Publish a versioned model release with matching code, pinned dependencies and verified download instructions.
- Consolidate prediction, monitoring and descriptive/SAA tables for Power BI.
- Build dashboard views for current warnings, historical outcomes, drift and label coverage.

Power BI export and dashboard functionality are planned and are not implemented by the commands documented above.
