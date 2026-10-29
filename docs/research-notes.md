# Research notes

## Coursework context

This repository presents the analysis code from a three-person DG1FAID coursework project at ONCAMPUS Aston. Mohamed Kemouche contributed data cleaning and imputation analysis.

The original submission report is not included. It contains student-identifying administrative material and inconsistent results. The notes below preserve the relevant context without reproducing those details.

## Relationship between the code and report

The report discusses mean and median imputation, K-nearest-neighbour imputation and listwise deletion. The supplied script implements KNN imputation and median filling of remaining numeric values, but does not run a complete comparative experiment covering every strategy discussed in the report.

The report gives different performance values and contradictory model rankings in different sections. Consequently, this repository does not reproduce its headline scores or designate a model as the winner. The generated model-score plots show the current script's output, with the methodological limitations below.

## Current preprocessing

1. The script loads a CSV if one exists, or generates 200 synthetic records with a fixed seed.
2. It fits a five-neighbour KNN imputer on selected numeric columns before dividing the data into training and test sets.
3. Those imputer inputs include `Final_Exam_Score`, the eventual prediction target.
4. Median filling handles remaining numeric missing values. `Previous_GPA` has already been included in KNN imputation, so the later median step is not a separate GPA-imputation comparison.
5. Categorical variables are encoded and an 80/20 train/test split is created.
6. A scaler is fitted to the training predictors. The preprocessed training data are subsequently passed into five-fold cross-validation.

This order allows information from outside the relevant training split into preprocessing, and allows target values to influence predictor imputation. The reported metrics therefore do not provide an unbiased estimate of performance on unseen cases.

KNN distances are also calculated on raw numeric columns with different units. Variables with larger numerical ranges may influence neighbour selection more strongly.

## Model configuration

The held-out random forest uses 100 trees and `min_samples_split=5`. The cross-validation random forest uses 100 trees but leaves `min_samples_split` at its default. Those evaluations therefore use different configurations.

The script suppresses warnings globally. A successful execution does not establish that the methods or statistical conclusions are valid.

## Demonstration run

The original script was copied to `analysis.py` without altering its contents. It was executed with the non-interactive Matplotlib `Agg` backend using its built-in synthetic generator.

The run completed and produced all nine figures under `assets/`. The runtime was Windows with Python 3.14.6; direct package versions are recorded in `requirements.txt`. A fresh-environment dependency installation and a run with an external CSV have not been independently verified.

The synthetic generator encodes relationships between predictors and the target. Results therefore describe this constructed demonstration and should not be generalised to real student populations.

## Possible next revisions

These are proposed changes, not implemented features:

- Split predictors from the target before fitting any imputer.
- Fit imputation and scaling inside each training split or cross-validation fold.
- Use identical estimator settings when comparing evaluation methods.
- Compare imputation strategies under the same evaluation protocol.
- Produce report tables directly from the corrected pipeline.
- Evaluate variability across repeated splits and document appropriate uncertainty.

Changing the evaluation pipeline will change outputs. The existing figures should be regenerated and the documentation updated together when those revisions are made.
