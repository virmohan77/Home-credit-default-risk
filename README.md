README


# Home Credit Default Risk — Credit Risk Modeling (LogReg + LightGBM)

End-to-end solution for the Kaggle **Home Credit Default Risk** competition: build applicant-level features by aggregating auxiliary tables (previous applications, bureau history, installments, credit card balances), train baseline Logistic Regression and a stronger LightGBM model with Stratified K-Fold cross-validation, and generate Kaggle-ready submissions.

**Best Kaggle Public AUC:** **0.78213** (LightGBM)

---

## What this project demonstrates
- ✅ Practical feature engineering on a **multi-table relational dataset** (1-to-many joins → 1 row per applicant)
- ✅ Clean modeling workflow: **Stratified K-Fold CV**, early stopping, reproducible training
- ✅ Comparison of interpretable baseline (**Logistic Regression**) vs boosted trees (**LightGBM**)
- ✅ Deliverables that matter: submission files + run logs + processed feature datasets

---

## Dataset (Kaggle)
Competition: **Home Credit Default Risk**  
Goal: predict the probability that an applicant will default on a loan.

Target variable: `TARGET` (1 = default, 0 = non-default)

Key entity identifier: `SK_ID_CURR` (applicant ID)

---

## Approach (high-level)
1. **Start with application data** (`application_train.csv`, `application_test.csv`)
2. **Aggregate auxiliary tables to applicant-level features**
   - `previous_application.csv` → counts, averages, statuses, amounts
   - `bureau.csv` + `bureau_balance.csv` → credit history signals
   - `installments_payments.csv` → payment behavior / delinquencies
   - `credit_card_balance.csv` → revolving credit usage patterns
3. **Modeling**
   - Baseline: Logistic Regression (fast + interpretable)
   - Final: LightGBM (handles non-linearities + missingness well)
4. **Validation**
   - Stratified K-Fold Cross-Validation (AUC metric)
   - LightGBM early stopping on validation folds
5. **Output**
   - Submission file in Kaggle format: `SK_ID_CURR`, `TARGET`

---

## Repo structure
```text
.
├── 01_home_credit_end_to_end.ipynb        # Full pipeline notebook (data → features → models → submission)
├── README.md                              # Project overview (this file)
├── requirements.txt                       # Dependencies
├── outputs/
│   ├── submission_lgbm_final.csv          # Final Kaggle submission (best model)
│   └── results_log.csv                    # CV fold scores / run summary
└── processed/
    ├── train_features.parquet             # Engineered train features (optional / generated)
    └── test_features.parquet              # Engineered test features (optional / generated)


