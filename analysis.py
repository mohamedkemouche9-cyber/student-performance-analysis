# =============================================================================
# DG1FAID – Group Project 1
# Predicting and Explaining Student Academic Performance
# ONCAMPUS Aston | Academic Year 2025-26
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#FAFAFA",
    "axes.edgecolor":   "#CCCCCC",
    "axes.grid":        True,
    "grid.color":       "#EEEEEE",
    "grid.linestyle":   "--",
    "font.family":      "Arial",
    "font.size":        11,
})
PURPLE = "#5B2D8E"
ORANGE = "#E87722"

# =============================================================================
# 0. LOAD DATA
# =============================================================================
# ─────────────────────────────────────────────────────────────────────────────
# NOTE: Replace the path below with the actual path to your CSV/Excel file.
#       The dataset columns expected:
#       Student_ID, Age, Gender, Attendance_Rate, Study_Hours_per_Week,
#       Coursework_Avg, Sleep_Hours, Part_Time_Job, Internet_Access,
#       Socioeconomic_Score, Previous_GPA, Final_Exam_Score
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH = "student_performance_dataset.csv"   # ← change if needed

try:
    df = pd.read_csv(DATA_PATH)
    print(f"✔  Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns\n")
except FileNotFoundError:
    # ── Synthetic fallback so the code runs end-to-end for demonstration ─────
    print("⚠  Dataset file not found. Generating synthetic demonstration data.\n")
    np.random.seed(42)
    n = 200
    age        = np.random.randint(18, 31, n).astype(float)
    gender     = np.random.choice(["M", "F", "Other"], n, p=[0.5, 0.4, 0.1])
    attend     = np.clip(np.random.normal(62, 24, n), 0, 100)
    study_h    = np.clip(np.random.normal(20, 12, n), 0, 56)
    cw_avg     = np.clip(np.random.normal(63, 20, n), 20, 100)
    sleep      = np.clip(np.random.normal(6.5, 2.1, n), 3, 10)
    ptj        = np.random.choice(["Yes", "No"], n, p=[0.4, 0.6])
    internet   = np.random.choice(["Yes", "No"], n, p=[0.85, 0.15])
    ses        = np.clip(np.random.uniform(1, 10, n), 0, 10)
    gpa        = np.clip(np.random.uniform(0, 4, n), 0, 4)
    noise      = np.random.normal(0, 1.5, n)
    score      = (0.10*attend + 0.08*study_h + 0.12*cw_avg*0.3
                  + 0.15*gpa*4 - 1.0*(ptj=="Yes") + noise + 5)
    score      = np.clip(score, 8, 28)

    df = pd.DataFrame({
        "Student_ID":           [f"S{1000+i}" for i in range(n)],
        "Age":                  age, "Gender": gender,
        "Attendance_Rate":      attend, "Study_Hours_per_Week": study_h,
        "Coursework_Avg":       cw_avg, "Sleep_Hours": sleep,
        "Part_Time_Job":        ptj, "Internet_Access": internet,
        "Socioeconomic_Score":  ses, "Previous_GPA": gpa,
        "Final_Exam_Score":     score,
    })

    # Inject realistic missingness
    for col, pct in [("Age",0.04),("Study_Hours_per_Week",0.28),
                     ("Coursework_Avg",0.30),("Sleep_Hours",0.25),
                     ("Socioeconomic_Score",0.28),("Previous_GPA",0.06)]:
        mask = np.random.rand(n) < pct
        df.loc[mask, col] = np.nan

# =============================================================================
# 1. DATA AUDIT
# =============================================================================
print("=" * 65)
print("TASK 1 — DATA AUDIT")
print("=" * 65)

# 1a. Variable types
type_map = {
    "Student_ID":           "Nominal (Identifier)",
    "Age":                  "Discrete Numeric",
    "Gender":               "Nominal Categorical",
    "Attendance_Rate":      "Continuous (Ratio %)",
    "Study_Hours_per_Week": "Continuous (Ratio)",
    "Coursework_Avg":       "Continuous (Ratio %)",
    "Sleep_Hours":          "Continuous (Ratio)",
    "Part_Time_Job":        "Binary Categorical",
    "Internet_Access":      "Binary Categorical",
    "Socioeconomic_Score":  "Continuous (Interval)",
    "Previous_GPA":         "Continuous (Ratio)",
    "Final_Exam_Score":     "Continuous (Ratio) — TARGET",
}
print("\nVariable Classification:")
print(f"{'Variable':<26} {'Type'}")
print("-" * 60)
for k, v in type_map.items():
    print(f"  {k:<24} {v}")

# 1b. Missing data
print("\nMissing Data Summary:")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
miss_df = pd.DataFrame({"Missing (n)": missing, "Missing (%)": missing_pct})
miss_df = miss_df[miss_df["Missing (n)"] > 0].sort_values("Missing (%)", ascending=False)
print(miss_df.to_string())

# 1c. Outlier detection via IQR
numeric_cols = ["Attendance_Rate", "Study_Hours_per_Week", "Coursework_Avg",
                "Sleep_Hours", "Socioeconomic_Score", "Previous_GPA",
                "Final_Exam_Score"]
print("\nOutlier Detection (IQR Method):")
print(f"{'Variable':<26} {'Q1':>8} {'Q3':>8} {'IQR':>8} {'Lower':>8} {'Upper':>8} {'Outliers':>8}")
print("-" * 80)
for col in numeric_cols:
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    n_out = ((df[col] < lo) | (df[col] > hi)).sum()
    print(f"  {col:<24} {Q1:>8.2f} {Q3:>8.2f} {IQR:>8.2f} {lo:>8.2f} {hi:>8.2f} {n_out:>8}")

# =============================================================================
# 2. MISSING DATA STRATEGY
# =============================================================================
print("\n" + "=" * 65)
print("TASK 2 — MISSING DATA MECHANISM & IMPUTATION")
print("=" * 65)

# Identify high-missingness continuous columns
high_miss_cols = ["Study_Hours_per_Week", "Coursework_Avg",
                  "Sleep_Hours", "Socioeconomic_Score"]
low_miss_cols  = ["Age", "Previous_GPA"]

# ── Before/After distribution plots ─────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
fig.suptitle("Before vs After Imputation — Distribution Comparison",
             fontsize=14, fontweight="bold", color=PURPLE, y=1.01)

imputer_knn = KNNImputer(n_neighbors=5)
df_knn = df.copy()
df_knn[numeric_cols] = imputer_knn.fit_transform(df[numeric_cols])

for i, col in enumerate(high_miss_cols):
    ax_before = axes[0, i]
    ax_after  = axes[1, i]
    df[col].dropna().plot(kind="hist", bins=20, color=PURPLE, alpha=0.6,
                          edgecolor="white", ax=ax_before)
    df_knn[col].plot(kind="hist", bins=20, color=ORANGE, alpha=0.6,
                     edgecolor="white", ax=ax_after)
    ax_before.set_title(f"{col}\n(Before — n={df[col].notna().sum()})",
                        fontsize=9, fontweight="bold")
    ax_after.set_title(f"{col}\n(After KNN — n={len(df_knn)})",
                       fontsize=9, fontweight="bold", color=ORANGE)
    for ax in [ax_before, ax_after]:
        ax.set_xlabel("")
        ax.set_ylabel("Frequency", fontsize=9)

axes[0, 0].set_ylabel("Before\nFrequency", fontsize=10, fontweight="bold")
axes[1, 0].set_ylabel("After KNN\nFrequency", fontsize=10, fontweight="bold",
                       color=ORANGE)
plt.tight_layout()
plt.savefig("fig1_imputation_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("✔  Figure saved: fig1_imputation_comparison.png")

# KS test: compare distributions before vs after
print("\nKolmogorov-Smirnov Test (original vs KNN-imputed):")
print(f"{'Variable':<26} {'KS Statistic':>14} {'p-value':>12} {'Similar?':>10}")
print("-" * 65)
for col in high_miss_cols:
    ks_stat, ks_p = stats.ks_2samp(
        df[col].dropna().values, df_knn[col].values)
    print(f"  {col:<24} {ks_stat:>14.4f} {ks_p:>12.4f} "
          f"{'YES' if ks_p > 0.05 else 'NO':>10}")

# Also apply median imputation to low-missingness columns
for col in low_miss_cols:
    df_knn[col].fillna(df_knn[col].median(), inplace=True)

print("\n✔  Imputation complete. Working dataset: df_knn")

# =============================================================================
# 3. DESCRIPTIVE STATISTICS
# =============================================================================
print("\n" + "=" * 65)
print("TASK 3 — DESCRIPTIVE STATISTICS & DISTRIBUTION ANALYSIS")
print("=" * 65)

desc_cols = numeric_cols + ["Age"]
stats_rows = []
for col in desc_cols:
    s = df_knn[col]
    stats_rows.append({
        "Variable":  col,
        "Mean":      round(s.mean(), 3),
        "Median":    round(s.median(), 3),
        "Std Dev":   round(s.std(), 3),
        "IQR":       round(s.quantile(0.75) - s.quantile(0.25), 3),
        "Skewness":  round(s.skew(), 3),
        "Min":       round(s.min(), 3),
        "Max":       round(s.max(), 3),
    })
stats_df = pd.DataFrame(stats_rows).set_index("Variable")
print("\nDescriptive Statistics (post-imputation):")
print(stats_df.to_string())

print("\nSkewness Interpretation:")
for _, row in stats_df.iterrows():
    sk = row["Skewness"]
    direction = ("approximately symmetric" if abs(sk) < 0.5
                 else ("positively skewed (right tail)" if sk > 0
                       else "negatively skewed (left tail)"))
    print(f"  {_:<26}: {sk:+.3f}  →  {direction}")

# =============================================================================
# 4. EXPLORATORY DATA ANALYSIS
# =============================================================================
print("\n" + "=" * 65)
print("TASK 4 — EXPLORATORY DATA ANALYSIS")
print("=" * 65)

# ── Fig 2: Histogram of Final_Exam_Score ────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df_knn["Final_Exam_Score"], bins=25, color=PURPLE, alpha=0.75,
        edgecolor="white")
ax.axvline(df_knn["Final_Exam_Score"].mean(),   color=ORANGE,  lw=2,
           linestyle="--", label=f"Mean = {df_knn['Final_Exam_Score'].mean():.2f}")
ax.axvline(df_knn["Final_Exam_Score"].median(), color="#2E8B57", lw=2,
           linestyle="-.",  label=f"Median = {df_knn['Final_Exam_Score'].median():.2f}")
ax.set_title("Distribution of Final Exam Score", fontsize=14,
             fontweight="bold", color=PURPLE)
ax.set_xlabel("Final Exam Score", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig("fig2_final_exam_histogram.png", dpi=150, bbox_inches="tight")
plt.close()
print("✔  Figure saved: fig2_final_exam_histogram.png")

# ── Fig 3: Scatter plots ─────────────────────────────────────────────────────
scatter_vars = [
    ("Attendance_Rate",      "Attendance Rate (%)"),
    ("Study_Hours_per_Week", "Study Hours per Week"),
    ("Coursework_Avg",       "Coursework Average (%)"),
]
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Scatter Plots: Predictors vs Final Exam Score",
             fontsize=13, fontweight="bold", color=PURPLE)

for ax, (xcol, xlabel) in zip(axes, scatter_vars):
    ax.scatter(df_knn[xcol], df_knn["Final_Exam_Score"],
               color=PURPLE, alpha=0.45, s=30, edgecolors="none")
    m, b, r, p, _ = stats.linregress(df_knn[xcol], df_knn["Final_Exam_Score"])
    xline = np.linspace(df_knn[xcol].min(), df_knn[xcol].max(), 100)
    ax.plot(xline, m*xline + b, color=ORANGE, lw=2,
            label=f"r = {r:.3f}  (p={'<0.001' if p < 0.001 else f'{p:.3f}'})")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Final Exam Score", fontsize=11)
    ax.set_title(f"{xlabel}\nvs Final Exam Score", fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    print(f"  Correlation {xcol:<26}: r = {r:.4f}, p = {p:.4e}")

plt.tight_layout()
plt.savefig("fig3_scatter_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("✔  Figure saved: fig3_scatter_plots.png")

# ── Fig 4: Boxplot – Part-Time Job vs Final Score ────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
groups = [
    df_knn.loc[df_knn["Part_Time_Job"] == "No",  "Final_Exam_Score"].values,
    df_knn.loc[df_knn["Part_Time_Job"] == "Yes", "Final_Exam_Score"].values,
]
bp_plot = ax.boxplot(groups, patch_artist=True, notch=False,
                     widths=0.45, medianprops=dict(color="white", lw=2))
for patch, color in zip(bp_plot["boxes"], [PURPLE, ORANGE]):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
ax.set_xticks([1, 2])
ax.set_xticklabels(["No Part-Time Job", "Part-Time Job"], fontsize=12)
ax.set_ylabel("Final Exam Score", fontsize=12)
ax.set_title("Final Exam Score by Part-Time Employment Status",
             fontsize=13, fontweight="bold", color=PURPLE)

no_med  = np.median(groups[0])
yes_med = np.median(groups[1])
ax.annotate(f"Median = {no_med:.2f}",  xy=(1, no_med),  xytext=(1.15, no_med+0.3),  fontsize=10, color=PURPLE)
ax.annotate(f"Median = {yes_med:.2f}", xy=(2, yes_med), xytext=(2.05, yes_med+0.3), fontsize=10, color=ORANGE)

t_stat, t_p = stats.ttest_ind(groups[0], groups[1])
ax.text(0.5, 0.02, f"Independent t-test: t = {t_stat:.3f}, p = {t_p:.4f}",
        transform=ax.transAxes, ha="center", fontsize=10, color="#555555",
        style="italic")
plt.tight_layout()
plt.savefig("fig4_boxplot_parttime.png", dpi=150, bbox_inches="tight")
plt.close()
print("✔  Figure saved: fig4_boxplot_parttime.png")

# ── Fig 5: Correlation heatmap ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
corr_cols = ["Attendance_Rate", "Study_Hours_per_Week", "Coursework_Avg",
             "Sleep_Hours", "Socioeconomic_Score", "Previous_GPA",
             "Final_Exam_Score"]
corr_matrix = df_knn[corr_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdYlGn",
            mask=mask, ax=ax, linewidths=0.5, vmin=-1, vmax=1,
            annot_kws={"size": 9})
ax.set_title("Pearson Correlation Matrix (Numeric Variables)",
             fontsize=13, fontweight="bold", color=PURPLE, pad=15)
plt.tight_layout()
plt.savefig("fig5_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("✔  Figure saved: fig5_correlation_heatmap.png")

# =============================================================================
# 5. PREPROCESSING FOR MODELLING
# =============================================================================
print("\n" + "=" * 65)
print("TASK 5 — MODEL BUILDING")
print("=" * 65)

df_model = df_knn.copy()

# Drop identifier
df_model.drop(columns=["Student_ID"], inplace=True, errors="ignore")

# Encode binary
df_model["Part_Time_Job"]   = (df_model["Part_Time_Job"]   == "Yes").astype(int)
df_model["Internet_Access"] = (df_model["Internet_Access"] == "Yes").astype(int)

# One-hot encode Gender
df_model = pd.get_dummies(df_model, columns=["Gender"], drop_first=False)
# Drop one dummy to avoid perfect multicollinearity
if "Gender_Other" in df_model.columns:
    df_model.drop(columns=["Gender_Other"], inplace=True)

# Fill any remaining NaNs (e.g. from boolean dummy cols)
df_model.fillna(df_model.median(numeric_only=True), inplace=True)

# Define X / y
TARGET = "Final_Exam_Score"
X = df_model.drop(columns=[TARGET])
y = df_model[TARGET]

print(f"\nFeatures used: {list(X.columns)}")
print(f"Target:        {TARGET}")

# Train/test split (80/20, stratified on Gender group not possible for
# continuous target — using random_state for reproducibility)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)
print(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}")

# Standardise
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Linear Regression ────────────────────────────────────────────────────────
lr = LinearRegression()
lr.fit(X_train_sc, y_train)
y_pred_lr = lr.predict(X_test_sc)

lr_r2   = r2_score(y_test, y_pred_lr)
lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
lr_mae  = mean_absolute_error(y_test, y_pred_lr)

print("\n── Linear Regression ──────────────────────────────────────")
print(f"  R²   = {lr_r2:.4f}")
print(f"  RMSE = {lr_rmse:.4f}")
print(f"  MAE  = {lr_mae:.4f}")

# Coefficients
coef_df = pd.DataFrame({
    "Feature":     X.columns,
    "Coefficient": lr.coef_
}).sort_values("Coefficient", key=abs, ascending=False)
print("\n  Top Coefficients (standardised):")
print(coef_df.head(8).to_string(index=False))

# ── Random Forest ────────────────────────────────────────────────────────────
rf = RandomForestRegressor(n_estimators=100, min_samples_split=5,
                           random_state=42, n_jobs=-1)
rf.fit(X_train_sc, y_train)
y_pred_rf = rf.predict(X_test_sc)

rf_r2   = r2_score(y_test, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
rf_mae  = mean_absolute_error(y_test, y_pred_rf)

print("\n── Random Forest ───────────────────────────────────────────")
print(f"  R²   = {rf_r2:.4f}")
print(f"  RMSE = {rf_rmse:.4f}")
print(f"  MAE  = {rf_mae:.4f}")

# Feature importances
fi_df = pd.DataFrame({
    "Feature":    X.columns,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False)
print("\n  Feature Importances:")
print(fi_df.to_string(index=False))

# Cross-validation
cv_lr = cross_val_score(LinearRegression(), X_train_sc, y_train,
                        cv=5, scoring="r2")
cv_rf = cross_val_score(RandomForestRegressor(n_estimators=100,
                        random_state=42, n_jobs=-1), X_train_sc, y_train,
                        cv=5, scoring="r2")
print(f"\n  5-fold CV R² — Linear Regression: {cv_lr.mean():.4f} ± {cv_lr.std():.4f}")
print(f"  5-fold CV R² — Random Forest:      {cv_rf.mean():.4f} ± {cv_rf.std():.4f}")

# ── Model Comparison Summary ─────────────────────────────────────────────────
print("\n── Model Comparison ────────────────────────────────────────")
print(f"  {'Metric':<10} {'Linear Reg':>14} {'Random Forest':>14}")
print(f"  {'-'*40}")
print(f"  {'R²':<10} {lr_r2:>14.4f} {rf_r2:>14.4f}")
print(f"  {'RMSE':<10} {lr_rmse:>14.4f} {rf_rmse:>14.4f}")
print(f"  {'MAE':<10} {lr_mae:>14.4f} {rf_mae:>14.4f}")

# ── Fig 6: Model comparison bar chart ───────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
fig.suptitle("Model Performance Comparison: Linear Regression vs Random Forest",
             fontsize=13, fontweight="bold", color=PURPLE)

metrics  = ["R²", "RMSE", "MAE"]
lr_vals  = [lr_r2, lr_rmse, lr_mae]
rf_vals  = [rf_r2, rf_rmse, rf_mae]

for ax, metric, lv, rv in zip(axes, metrics, lr_vals, rf_vals):
    bars = ax.bar(["Linear\nRegression", "Random\nForest"],
                  [lv, rv], color=[PURPLE, ORANGE], alpha=0.80, width=0.5,
                  edgecolor="white")
    for bar, val in zip(bars, [lv, rv]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(lv,rv),
                f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title(metric, fontsize=13, fontweight="bold")
    ax.set_ylabel(metric, fontsize=11)
    ax.set_ylim(0, max(lv, rv) * 1.25)

plt.tight_layout()
plt.savefig("fig6_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("✔  Figure saved: fig6_model_comparison.png")

# ── Fig 7: Actual vs Predicted (both models) ─────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Actual vs Predicted — Final Exam Score",
             fontsize=13, fontweight="bold", color=PURPLE)

lim = [y_test.min() - 1, y_test.max() + 1]
for ax, preds, title, color in [
    (ax1, y_pred_lr, f"Linear Regression  (R²={lr_r2:.3f})", PURPLE),
    (ax2, y_pred_rf, f"Random Forest  (R²={rf_r2:.3f})",     ORANGE)
]:
    ax.scatter(y_test, preds, color=color, alpha=0.55, s=40, edgecolors="none")
    ax.plot(lim, lim, "k--", lw=1.5, label="Perfect prediction")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Actual Score", fontsize=11)
    ax.set_ylabel("Predicted Score", fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig("fig7_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.close()
print("✔  Figure saved: fig7_actual_vs_predicted.png")

# ── Fig 8: Feature Importance (Random Forest) ────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
fi_plot = fi_df.head(10).sort_values("Importance")
colors  = [ORANGE if v == fi_plot["Importance"].max() else PURPLE
           for v in fi_plot["Importance"]]
bars = ax.barh(fi_plot["Feature"], fi_plot["Importance"],
               color=colors, alpha=0.80, edgecolor="white")
for bar, val in zip(bars, fi_plot["Importance"]):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=9)
ax.set_xlabel("Mean Decrease in Impurity (Feature Importance)", fontsize=11)
ax.set_title("Random Forest — Feature Importance Rankings",
             fontsize=13, fontweight="bold", color=PURPLE)
plt.tight_layout()
plt.savefig("fig8_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("✔  Figure saved: fig8_feature_importance.png")

# ── Fig 9: Residual plots ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Residual Analysis", fontsize=13, fontweight="bold", color=PURPLE)

for ax, preds, title, color in [
    (axes[0], y_pred_lr, "Linear Regression", PURPLE),
    (axes[1], y_pred_rf, "Random Forest",     ORANGE),
]:
    residuals = y_test.values - preds
    ax.scatter(preds, residuals, color=color, alpha=0.55, s=35, edgecolors="none")
    ax.axhline(0, color="black", lw=1.5, linestyle="--")
    ax.set_xlabel("Predicted Score", fontsize=11)
    ax.set_ylabel("Residual (Actual − Predicted)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.text(0.05, 0.95, f"RMSE = {np.sqrt(np.mean(residuals**2)):.3f}",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.savefig("fig9_residual_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("✔  Figure saved: fig9_residual_plots.png")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 65)
print("PIPELINE COMPLETE — SUMMARY")
print("=" * 65)
print(f"\n  Dataset:          {len(df_knn)} students × {len(X.columns)} features")
print(f"  Imputation:       KNN (k=5) for high-missingness columns")
print(f"\n  Linear Regression  →  R²={lr_r2:.4f}  RMSE={lr_rmse:.4f}  MAE={lr_mae:.4f}")
print(f"  Random Forest      →  R²={rf_r2:.4f}  RMSE={rf_rmse:.4f}  MAE={rf_mae:.4f}")
print(f"\n  Best model:        {'Random Forest' if rf_r2 > lr_r2 else 'Linear Regression'}")
print(f"\n  Figures saved:     fig1 – fig9 (PNG, 150 dpi)")
print("\n  ✔  Done. Use these metric values in your report.")
print("=" * 65)
