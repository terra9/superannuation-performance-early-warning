# Superannuation Performance & Underperformance Early-Warning System

An ongoing data analytics and machine-learning project using publicly available
Australian Prudential Regulation Authority (APRA) superannuation data to
investigate investment-option performance and develop an early-warning framework
for potential future underperformance.

## Project Status

**In progress**

The project is currently focused on data integration, data-quality validation,
target construction and feature engineering.

Current processing is primarily notebook-based. Planned work includes
refactoring the workflow into reusable and reproducible data pipeline components,
followed by predictive model development, evaluation and interpretation.

## Objective

The project investigates whether historical performance, peer-relative
performance, volatility, strategic asset allocation, currency hedging and
changes in investment characteristics can provide useful early signals of
future investment-option underperformance.

The broader goal is to develop a reproducible analytical workflow that moves
from raw APRA data through validation, transformation and feature engineering
to model-ready datasets and, subsequently, predictive modelling.

## Data Source

The project uses publicly available superannuation data published by the
**Australian Prudential Regulation Authority (APRA)**.

The analysis currently incorporates information relating to:

- Historical investment performance
- Superannuation products and investment options
- Strategic asset allocation
- Asset-class allocations
- Investment volatility
- Currency hedging
- Product and investment-option characteristics

The project analyses data primarily at the **investment-option level** where
performance and investment characteristics are represented.

## Current Workflow

1. APRA source ingestion and schema inspection
2. Entity and identifier validation across source tables
3. Duplicate and conflicting-record investigation
4. Historical performance data preparation
5. Future performance target construction
6. Peer-group and peer-relative feature development
7. Historical return feature engineering
8. Volatility feature engineering
9. Strategic asset allocation feature engineering
10. Currency hedging feature engineering
11. Quarter-to-quarter allocation change features
12. Final modelling-dataset validation
13. Reproducible data and modelling pipeline development **[planned]**
14. Predictive model training and comparison **[planned]**
15. Model evaluation and interpretation **[planned]**

## Current Progress

### Completed / substantially developed

- APRA source ingestion and structural inspection
- Cross-table entity and identifier analysis
- Duplicate and conflicting-record investigation
- Historical performance preparation
- Future four-quarter performance target construction
- Peer-relative performance features
- Lagged and trailing return features
- Volatility features
- Strategic asset allocation features
- Asset-class allocation features
- Currency hedging features
- Quarter-to-quarter allocation change features
- Missing-data and feature-availability analysis

### In Progress

- Modelling-dataset preparation
- Assessment of feature distributions and data-quality edge cases

### Planned

- Refactor notebook-based processing into reusable pipeline components
- Add reproducible data-processing and modelling workflows
- Split the dataset into train and test dataset
- Train and compare predictive models
- Evaluate predictive performance and stability
- Interpret model behaviour and feature contribution
- Document final findings and limitations

## Data Quality and Validation

A major part of the project focuses on understanding and validating the
underlying data before modelling.

Examples of checks performed include:

- Duplicate record investigation
- Conflicting-value analysis across entity levels
- Cross-table identifier consistency checks
- Missing-value and historical-coverage analysis
- Quarter-to-quarter continuity validation
- Investment-option and product relationship checks
- Target-availability validation
- Feature missingness and distribution checks
- Strategic asset allocation consistency checks
- Currency hedging consistency checks
- Validation of derived historical-return features

These checks are used to reduce the risk of inconsistent, duplicated or
incomplete records propagating into downstream analysis and modelling.

## Feature Engineering

Current feature groups include:

### Historical Performance

- Quarterly return lags
- Trailing cumulative returns
- Rolling return statistics
- Positive-quarter share
- Peer-relative performance measures

### Risk and Volatility

- Historical volatility measures
- Volatility availability indicators

### Strategic Asset Allocation

- Strategic growth allocation
- Equity allocation
- Property allocation
- Fixed-income allocation
- Infrastructure allocation
- Cash allocation
- Alternatives allocation
- Credit allocation
- Quarter-to-quarter allocation changes

### Currency Hedging

- Weighted currency hedging ratio
- Hedging-applicable allocation
- Hedging configuration indicators

Feature development remains under active validation and may change as the
modelling stage progresses.

## Planned Pipeline Development

The current analytical workflow has been developed primarily through Jupyter
notebooks to support iterative investigation and validation of the APRA data.

A planned next stage is to refactor stable processing steps into reusable
pipeline components covering:

- Data ingestion
- Data validation
- Data transformation
- Target generation
- Feature engineering
- Modelling-dataset construction
- Model training
- Model evaluation

The objective is to make the workflow more reproducible and maintainable while
preserving explicit validation checks between processing stages.

This project does **not currently represent a production-deployed system**.

## Technologies

- Python
- Pandas
- NumPy
- Jupyter Notebook
- Git
- GitHub

Additional libraries and modelling tools will be documented as the modelling
stage develops.

## Repository Structure

```text
superannuation-performance-early-warning/
│
├── data/
│   └── Project data and intermediate datasets
│
├── notebooks/
│   └── Data exploration, validation, target construction
│       and feature-engineering notebooks
│
├── src/
│   └── Reusable project code and pipeline components
│
├── .gitignore
│
└── README.md