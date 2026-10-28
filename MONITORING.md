# Monitor archived MySuper predictions

Place monitor.py in your project root beside transform.py and predict.py.
Use your existing project virtual environment. No workflow_common.py or model.joblib is required by monitor.py: it reads the scores already issued.

## Run

First prepare the edition whose actual outcomes are currently available. If it is already prepared, run:

    python monitor.py --edition 2026-03

The input is:

    data/processed/hist_model_mysuper_features_2026-03.parquet

The script reads every completed archive under outputs/prediction_runs/. It supports the simplified run_metadata.json format and the earlier completed manifest.json format, provided predictions.csv, scored_snapshot.parquet and specification.json are present.
An old flat predictions_2026-03.csv without run metadata is not treated as an archive.
Folders without a completed metadata marker are skipped and listed in the monitoring metadata. Malformed completed archives stop with an error rather than being silently dropped.

When a newer edition has been downloaded and processed, run monitor.py again with that edition. For example:

    python etl.py --edition 2026-06
    python predict.py --edition 2026-06
    python monitor.py --edition 2026-06

The prediction command creates the latest warnings. Monitoring also revisits the earlier archived warnings. It can be run independently, without generating another prediction run.
There is no automatic download, schedule or retraining in this script.

## What is saved

Each monitoring execution creates a new folder:

    outputs/monitoring_runs/<monitoring_id>/

| File | Contents |
| --- | --- |
| prediction_drift.csv | Probability summaries, warning rates and changes from the reference run |
| feature_drift.csv | Distribution and missingness summaries for each scored predictor |
| evaluated_predictions.csv | Original scores and descriptive fields, matched actual outcomes and evaluation status |
| performance.csv | Metrics and outcome-coverage counts for each archived prediction run |
| monitoring_metadata.json | Evaluation time, outcome edition/fingerprint, included runs and review settings |

Original prediction archives and earlier monitoring folders are not changed. Monitoring does not generate new probabilities or modify the fitted model.
If APRA revises historical outcomes, a later monitoring batch can report different actual labels. Earlier monitoring batches preserve the previous evaluation and its source fingerprint.

## Prediction drift

The reference is the first-issued run for the previous available origin quarter, using the same model version. That reference must have been issued before the run being compared.
Reruns of the same quarter do not count as new reference quarters. A different model version starts a new comparison series.

The report includes mean, median, 10th/90th percentile probability, warning rate, Wasserstein distance between probability distributions, and the change in warning rate.
Wasserstein distance is reported in probability units. It is not a change in accuracy or a statistical significance test.

With only one archived origin quarter, comparison_status is no_prior_origin_same_model and comparison/review fields are blank. Current distribution summaries are still provided.
origin_quarters_apart records gaps when the previous available archive is more than one quarter earlier.

## Input drift

The same previous-quarter reference is used for all predictors. This first version does not compare against the original training distribution.
Each predictor has observed-row counts, missing shares, medians and 10th/90th percentiles.

Value distributions are compared using only observed values, without imputation. The Wasserstein distance is divided by the reference interquartile range (IQR). If IQR is zero, reference standard deviation is used. If the reference is constant, a change from that value is flagged directly.
These calculations do not scale or transform the model inputs.

If either distribution has no observed values, value drift cannot be calculated. Missingness changes are still checked. If both are entirely missing with no missingness shift, review_required remains blank rather than indicating a completed value-distribution check.

## Review flags

The constants at the top of monitor.py are practical review guides for this project:

| Check | Flag when |
| --- | --- |
| Predicted probability distribution | Wasserstein distance is at least 0.05 |
| Warning rate | Absolute change is at least 0.10 (ten percentage points) |
| Predictor missing share | Absolute change is at least 0.10 |
| Predictor values | Scaled Wasserstein distance is at least 0.50, or a constant reference changes |

These cutoffs are not validated APRA standards, statistical tests or automatic retraining rules.
A flag means inspect the change. Market conditions, population composition, revisions and data-processing problems can all change distributions. Small observed subsets can make the comparisons noisy.
Quarter-to-quarter monitoring can miss gradual cumulative changes; use the saved summaries to inspect longer trends as more quarters accumulate.

## Actual outcomes and maturity

Actual labels come from the prepared dataset's existing fields:

- future_4q_bottom_quartile
- future_4q_target_available_option
- target_end_date

The script does not reconstruct peer targets. It validates and uses the established option-level labels.
Outcomes are matched by rse_abn, abn_investment_option_identifier and period_end_date. Product/menu names are not keys. An available label from another representation of the same option can therefore evaluate an original archived representation.
Conflicting labels between representations stop the script. Different fund/option identifiers are not stitched across mergers.

| evaluation_status | Meaning |
| --- | --- |
| pending_window | The forecast end is beyond the reporting coverage of the supplied outcome edition |
| label_available | The four-quarter window is covered and the option-level target is available |
| mature_unavailable | The window is covered, but that option-quarter has no usable target |
| option_not_in_label_snapshot | The window is covered, but that option-quarter is absent from the outcome dataset |

Unavailable outcomes remain missing. They are not counted as actual negatives.
Maturity uses the latest reporting quarter in the outcome dataset, not today's calendar date. Publication dates are not inferred.

For a 2026 Q1 origin, the forecast ends in 2027 Q1. A March or June 2026 outcome dataset cannot yet evaluate that prediction. Blank performance metrics are therefore expected while the outcome is pending.

## Performance and repeated runs

Metrics use the original product/menu modelling rows and their archived probabilities and warnings. The label join uses distinct economic option-quarters. Labels are not duplicated before matching.

The report includes ROC AUC, average precision (AP), Brier score, precision, recall, F1, accuracy, balanced accuracy and confusion counts. It also includes issued, mature, labelled, pending and unavailable counts.
label_coverage is labelled_rows divided by mature_rows. The coverage is important when missing future histories leave only part of a cohort evaluable.
Undefined metrics are blank. For example, ROC AUC needs both actual classes; precision is undefined when no labelled observations were warned.

Every run is evaluated separately. is_first_run_for_origin identifies the first-issued run for each model/origin pair. Use that flag to avoid treating reruns as additional independent origin quarters in later summaries.
AP and ROC AUC are calculated per run here. Do not average them and describe the result as pooled AP or pooled ROC AUC. This version does not create a pooled or rolling score.

An issued_after_forecast_end flag identifies retrospectively generated warnings whose forecast window had already ended when the score was issued. Other publication delays can also affect timeliness. These outputs do not certify point-in-time data availability or real-time early-warning performance.

## Power BI

The CSV files retain run/model/monitoring identifiers for later consolidation. Evaluated predictions retain the archived descriptive fields.
SAA detail remains in the prediction archive; this monitoring script does not flatten allocation components into prediction rows.
The separate Power BI export stage can later consolidate these monitoring files and the prediction archives into fixed dashboard tables.
Import ABNs and all original identifiers as text to preserve leading zeros.

## Method references

- SciPy Wasserstein distance: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html
- scikit-learn AP: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
- scikit-learn ROC AUC: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html

The reference selection, scaling and review cutoffs are project-specific implementation choices, rather than a single published monitoring protocol.
