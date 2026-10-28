"""
================================================================================
TITANIC DATASET - DATA CLEANING & EXPLORATORY DATA ANALYSIS (EDA)
================================================================================
Task: Prodigy InfoTech - Data Science Task 2
Dataset: Titanic (train.csv, test.csv, gender_submission.csv)
Source: https://github.com/Prodigy-InfoTech/data-science-datasets/tree/main/Task%202

This script performs:
  1. Data loading & inspection
  2. Data cleaning (missing values, duplicates, data types, outliers)
  3. Feature engineering
  4. Exploratory Data Analysis (univariate, bivariate, multivariate)
  5. Saves cleaned datasets, plots, and an EDA summary report
================================================================================
"""

import os
import warnings
import re
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "eda_output")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

TRAIN_PATH = os.path.join(BASE_DIR, "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "test.csv")
SUBMISSION_PATH = os.path.join(BASE_DIR, "gender_submission.csv")

os.makedirs(PLOTS_DIR, exist_ok=True)

# Consistent styling
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11


def save_fig(fig, name):
    """Save a matplotlib figure to the plots directory."""
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ===========================================================================
# 1. DATA LOADING & INITIAL INSPECTION
# ===========================================================================
print("=" * 80)
print("1. DATA LOADING & INITIAL INSPECTION")
print("=" * 80)

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
gender_submission = pd.read_csv(SUBMISSION_PATH)

# Combine train and test for unified cleaning (keep a marker to split later)
train["source"] = "train"
test["source"] = "test"
test["Survived"] = np.nan  # add placeholder so columns align

df = pd.concat([train, test], axis=0, ignore_index=True)
print(f"\nCombined dataset shape: {df.shape}")

print("\n--- train.csv info ---")
print(train.info())
print("\n--- test.csv info ---")
print(test.info())

print("\n--- First 5 rows of train ---")
print(train.head())

print("\n--- Missing values per column (train) ---")
print(train.isnull().sum())

print("\n--- Duplicate rows in train:", train.duplicated().sum(), "---")
print("--- Duplicate rows in test:", test.duplicated().sum(), "---")

print("\n--- Descriptive statistics (train, numeric) ---")
print(train.describe())

print("\n--- Descriptive statistics (train, categorical) ---")
print(train.describe(include=["O"]))


# ===========================================================================
# 2. DATA CLEANING
# ===========================================================================
print("\n" + "=" * 80)
print("2. DATA CLEANING")
print("=" * 80)

# --- 2.1 Remove exact duplicates (none expected, but verify) ---
before = len(df)
df = df.drop_duplicates()
print(f"\nRemoved {before - len(df)} duplicate row(s). Remaining: {len(df)}")

# --- 2.2 Missing values summary (combined) ---
print("\n--- Missing values per column (combined) ---")
print(df.isnull().sum())

# --- 2.3 Handle missing 'Embarked' (2 missing in train) -> mode imputation ---
embarked_mode = df["Embarked"].mode()[0]
df["Embarked"] = df["Embarked"].fillna(embarked_mode)
print(f"\nFilled missing 'Embarked' with mode: '{embarked_mode}'")

# --- 2.4 Handle missing 'Fare' (1 missing in test) -> median by Pclass ---
fare_median_by_class = df.groupby("Pclass")["Fare"].transform("median")
df["Fare"] = df["Fare"].fillna(fare_median_by_class)
print("Filled missing 'Fare' with median Fare by Pclass.")

# --- 2.5 Handle missing 'Age' -> median by Sex & Pclass ---
age_median_by_sex_pclass = df.groupby(["Sex", "Pclass"])["Age"].transform("median")
df["Age"] = df["Age"].fillna(age_median_by_sex_pclass)
print("Filled missing 'Age' with median Age by Sex & Pclass.")

# --- 2.6 Handle 'Cabin' -> extract Deck letter; missing -> 'Unknown' ---
df["Cabin"] = df["Cabin"].fillna("Unknown")
df["Deck"] = df["Cabin"].astype(str).str[0]
df["Deck"] = df["Deck"].replace({"Unknown": "Unknown"})
print("\nExtracted 'Deck' from 'Cabin'. Missing cabins marked 'Unknown'.")

# --- 2.7 Data type conversions ---
df["Survived"] = df["Survived"].astype("Int64")  # nullable integer
df["Pclass"] = df["Pclass"].astype("category")
df["Sex"] = df["Sex"].astype("category")
df["Embarked"] = df["Embarked"].astype("category")
df["Deck"] = df["Deck"].astype("category")

print("\n--- Missing values after cleaning ---")
print(df.isnull().sum()[df.isnull().sum() > 0] if df.isnull().sum().any() else "None")

# --- 2.8 Outlier detection on 'Fare' (combined) ---
Q1 = df["Fare"].quantile(0.25)
Q3 = df["Fare"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df["Fare"] < lower_bound) | (df["Fare"] > upper_bound)]
print(f"\nFare outliers (IQR method): {len(outliers)} rows")

# ===========================================================================
# 3. FEATURE ENGINEERING
# ===========================================================================
print("\n" + "=" * 80)
print("3. FEATURE ENGINEERING")
print("=" * 80)

# --- 3.1 Extract Title from Name ---
def get_title(name):
    title_search = re.search(r" ([A-Za-z]+)\.", name)
    return title_search.group(1) if title_search else "Unknown"

df["Title"] = df["Name"].apply(get_title)

# Group rare titles
title_counts = df["Title"].value_counts()
rare_titles = title_counts[title_counts < 10].index.tolist()
df["Title"] = df["Title"].replace(rare_titles, "Rare")
df["Title"] = df["Title"].astype("category")
print("\nTitle value counts (after grouping rare titles):")
print(df["Title"].value_counts())

# --- 3.2 Family size & IsAlone ---
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

# --- 3.3 Age bins ---
df["AgeBin"] = pd.cut(
    df["Age"],
    bins=[0, 12, 18, 35, 60, 100],
    labels=["Child", "Teen", "YoungAdult", "Adult", "Senior"],
)
df["AgeBin"] = df["AgeBin"].astype("category")

# --- 3.4 Fare bins ---
df["FareBin"] = pd.qcut(df["Fare"], q=4, labels=["Low", "Medium", "High", "VeryHigh"])
df["FareBin"] = df["FareBin"].astype("category")

# --- 3.5 Split back into train & test ---
train_clean = df[df["source"] == "train"].copy().drop(columns=["source"])
test_clean = df[df["source"] == "test"].copy().drop(columns=["source", "Survived"])

# Reattach ground truth Survived to test for EDA comparison (optional)

# ===========================================================================
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# ===========================================================================
print("\n" + "=" * 80)
print("4. EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 80)

eda_summary = []
eda_summary.append("TITANIC DATASET - EDA SUMMARY REPORT")
eda_summary.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
eda_summary.append("=" * 80)


def record(section, text):
    eda_summary.append(f"\n[{section}] {text}")

# ---------------------------------------------------------------------------
# 4.1 UNIVARIATE ANALYSIS
# ---------------------------------------------------------------------------
print("\n--- 4.1 Univariate Analysis ---")

# Survival distribution
surv_counts = train_clean["Survived"].value_counts()
surv_rate = train_clean["Survived"].mean() * 100
record("Univariate - Survival",
       f"Total passengers: {len(train_clean)} | Survived: {surv_counts.get(1, 0)} "
       f"({surv_counts.get(1,0)/len(train_clean)*100:.1f}%) | "
       f"Did not survive: {surv_counts.get(0, 0)} "
       f"({surv_counts.get(0,0)/len(train_clean)*100:.1f}%)")
print(f"Survival rate: {surv_rate:.2f}%")

fig, ax = plt.subplots()
sns.countplot(x="Survived", data=train_clean, ax=ax, palette="Set2")
ax.set_title("Survival Count (0 = No, 1 = Yes)")
ax.set_xticklabels(["Did Not Survive", "Survived"])
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=11, fontweight="bold")
save_fig(fig, "01_survival_count.png")

# Pclass
pclass_dist = train_clean["Pclass"].value_counts(normalize=True).round(3) * 100
record("Univariate - Pclass", f"Class distribution (%): {pclass_dist.to_dict()}")
fig, ax = plt.subplots()
sns.countplot(x="Pclass", data=train_clean, ax=ax, palette="Set2", order=[1, 2, 3])
ax.set_title("Passenger Class Distribution")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=11, fontweight="bold")
save_fig(fig, "02_pclass_distribution.png")

# Sex
sex_dist = train_clean["Sex"].value_counts(normalize=True).round(3) * 100
record("Univariate - Sex", f"Sex distribution (%): {sex_dist.to_dict()}")
fig, ax = plt.subplots()
sns.countplot(x="Sex", data=train_clean, ax=ax, palette="Set2")
ax.set_title("Sex Distribution")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=11, fontweight="bold")
save_fig(fig, "03_sex_distribution.png")

# Age distribution
fig, ax = plt.subplots()
sns.histplot(train_clean["Age"], bins=30, kde=True, ax=ax, color="steelblue")
ax.set_title("Age Distribution")
ax.set_xlabel("Age")
save_fig(fig, "04_age_distribution.png")

# Fare distribution (log-scaled due to skew)
fig, ax = plt.subplots()
sns.histplot(np.log1p(train_clean["Fare"]), bins=30, kde=True, ax=ax, color="darkorange")
ax.set_title("Fare Distribution (log1p scale)")
ax.set_xlabel("log(Fare + 1)")
save_fig(fig, "05_fare_distribution.png")

# Embarked
embarked_dist = train_clean["Embarked"].value_counts(normalize=True).round(3) * 100
record("Univariate - Embarked", f"Embarked distribution (%): {embarked_dist.to_dict()}")
fig, ax = plt.subplots()
sns.countplot(x="Embarked", data=train_clean, ax=ax, palette="Set2", order=["C", "Q", "S"])
ax.set_title("Port of Embarkation Distribution")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=11, fontweight="bold")
save_fig(fig, "06_embarked_distribution.png")

# SibSp / Parch
fig, axes = plt.subplots(1, 2)
sns.countplot(x="SibSp", data=train_clean, ax=axes[0], palette="Set2")
axes[0].set_title("Siblings/Spouses Aboard (SibSp)")
sns.countplot(x="Parch", data=train_clean, ax=axes[1], palette="Set2")
axes[1].set_title("Parents/Children Aboard (Parch)")
plt.tight_layout()
save_fig(plt.gcf(), "07_sibsp_parch_distribution.png")

test_with_truth = test_clean.merge(gender_submission, on="PassengerId", how="left")

print(f"\ntrain_clean shape: {train_clean.shape}")
print(f"test_clean  shape: {test_clean.shape}")

# Save cleaned datasets
train_clean.to_csv(os.path.join(OUTPUT_DIR, "train_cleaned.csv"), index=False)
test_clean.to_csv(os.path.join(OUTPUT_DIR, "test_cleaned.csv"), index=False)
print("\nSaved cleaned datasets to eda_output/")

print(f"  Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}")
# Note: We keep outliers (do not drop) since they may be legitimate (e.g., high fare)

gender_submission = pd.read_csv(SUBMISSION_PATH)

print(f"\ntrain.csv  shape: {train.shape}")
print(f"test.csv   shape: {test.shape}")
print(f"gender_submission.csv shape: {gender_submission.shape}")

# Combine train and test for unified cleaning (keep a marker to split later)
train["source"] = "train"
test["source"] = "test"
test["Survived"] = np.nan  # add placeholder so columns align

df = pd.concat([train, test], axis=0, ignore_index=True)
print(f"\nCombined dataset shape: {df.shape}")

print("\n--- train.csv info ---")
print(train.info())
print("\n--- test.csv info ---")
print(test.info())

print("\n--- First 5 rows of train ---")
print(train.head())

print("\n--- Missing values per column (train) ---")
print(train.isnull().sum())

print("\n--- Duplicate rows in train:", train.duplicated().sum(), "---")
print("--- Duplicate rows in test:", test.duplicated().sum(), "---")


# ---------------------------------------------------------------------------
# 4.2 BIVARIATE ANALYSIS (Survival vs each variable)
# ---------------------------------------------------------------------------
print("\n--- 4.2 Bivariate Analysis (vs Survived) ---")

# Survival by Pclass
fig, ax = plt.subplots()
sns.barplot(x="Pclass", y="Survived", data=train_clean, ax=ax, order=[1, 2, 3], palette="Set2")
ax.set_title("Survival Rate by Passenger Class")
ax.set_ylabel("Survival Rate")
ax.set_ylim(0, 1)
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2%}", (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=11)
save_fig(fig, "08_survival_by_pclass.png")
record("Bivariate - Pclass",
       f"Survival rate by class: {train_clean.groupby('Pclass')['Survived'].mean().round(3).to_dict()}")

# Survival by Sex
fig, ax = plt.subplots()
sns.barplot(x="Sex", y="Survived", data=train_clean, ax=ax, palette="Set2")
ax.set_title("Survival Rate by Sex")
ax.set_ylabel("Survival Rate")
ax.set_ylim(0, 1)
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2%}", (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=11)
save_fig(fig, "09_survival_by_sex.png")
record("Bivariate - Sex",
       f"Survival rate by sex: {train_clean.groupby('Sex')['Survived'].mean().round(3).to_dict()}")

# Survival by AgeBin
fig, ax = plt.subplots()
order_age = ["Child", "Teen", "YoungAdult", "Adult", "Senior"]
sns.barplot(x="AgeBin", y="Survived", data=train_clean, ax=ax, order=order_age, palette="Set2")
ax.set_title("Survival Rate by Age Group")
ax.set_ylabel("Survival Rate")
ax.set_ylim(0, 1)
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2%}", (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=11)
save_fig(fig, "10_survival_by_agebin.png")
record("Bivariate - AgeBin",
       f"Survival rate by age group: {train_clean.groupby('AgeBin')['Survived'].mean().round(3).reindex(order_age).to_dict()}")

# Survival by Embarked
fig, ax = plt.subplots()
sns.barplot(x="Embarked", y="Survived", data=train_clean, ax=ax, order=["C", "Q", "S"], palette="Set2")
ax.set_title("Survival Rate by Embarkation Port")
ax.set_ylabel("Survival Rate")
ax.set_ylim(0, 1)
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2%}", (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=11)
save_fig(fig, "11_survival_by_embarked.png")
record("Bivariate - Embarked",
       f"Survival rate by embarked: {train_clean.groupby('Embarked')['Survived'].mean().round(3).to_dict()}")

# Survival by Sex & Pclass (grouped)
fig, ax = plt.subplots()
sns.barplot(x="Pclass", y="Survived", hue="Sex", data=train_clean, ax=ax, order=[1, 2, 3], palette="Set2")
ax.set_title("Survival Rate by Class & Sex")
ax.set_ylabel("Survival Rate")
ax.set_ylim(0, 1)
ax.legend(title="Sex")
save_fig(fig, "12_survival_by_class_sex.png")

# Survival by FamilySize
fig, ax = plt.subplots()
sns.barplot(x="FamilySize", y="Survived", data=train_clean, ax=ax, palette="Set2")
ax.set_title("Survival Rate by Family Size")
ax.set_ylabel("Survival Rate")
ax.set_ylim(0, 1)
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2%}", (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=10)
save_fig(fig, "13_survival_by_familysize.png")
record("Bivariate - FamilySize",
       f"Survival rate by family size: {train_clean.groupby('FamilySize')['Survived'].mean().round(3).to_dict()}")

# Survival by IsAlone
fig, ax = plt.subplots()
sns.barplot(x="IsAlone", y="Survived", data=train_clean, ax=ax, palette="Set2")
ax.set_title("Survival Rate: Alone vs With Family")
ax.set_ylabel("Survival Rate")
ax.set_xticklabels(["With Family", "Alone"])
ax.set_ylim(0, 1)
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2%}", (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=11)
save_fig(fig, "14_survival_by_isalone.png")
record("Bivariate - IsAlone",
       f"Survival rate by IsAlone: {train_clean.groupby('IsAlone')['Survived'].mean().round(3).to_dict()}")

# Survival by Title
fig, ax = plt.subplots()
title_order = train_clean.groupby("Title")["Survived"].mean().sort_values(ascending=False).index
sns.barplot(x="Title", y="Survived", data=train_clean, ax=ax, order=title_order, palette="Set2")
ax.set_title("Survival Rate by Title")
ax.set_ylabel("Survival Rate")
ax.set_ylim(0, 1)
ax.tick_params(axis="x", rotation=45)
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2%}", (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=10)
save_fig(fig, "15_survival_by_title.png")
record("Bivariate - Title",
       f"Survival rate by title: {train_clean.groupby('Title')['Survived'].mean().round(3).sort_values(ascending=False).to_dict()}")

# Age vs Survival (boxplot)
fig, ax = plt.subplots()
sns.boxplot(x="Survived", y="Age", data=train_clean, ax=ax, palette="Set2")
ax.set_title("Age Distribution by Survival")
ax.set_xticklabels(["Did Not Survive", "Survived"])
save_fig(fig, "16_age_by_survival_boxplot.png")

# Fare vs Survival (boxplot)
fig, ax = plt.subplots()
sns.boxplot(x="Survived", y="Fare", data=train_clean, ax=ax, palette="Set2")
ax.set_title("Fare Distribution by Survival")
ax.set_xticklabels(["Did Not Survive", "Survived"])
save_fig(fig, "17_fare_by_survival_boxplot.png")

print("\n--- Descriptive statistics (train, numeric) ---")
print(train.describe())

print("\n--- Descriptive statistics (train, categorical) ---")
print(train.describe(include=["O"]))


# ---------------------------------------------------------------------------
# 4.3 MULTIVARIATE ANALYSIS
# ---------------------------------------------------------------------------
print("\n--- 4.3 Multivariate Analysis ---")

# Correlation heatmap (numeric)
numeric_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"]
corr = train_clean[numeric_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0, ax=ax, fmt=".2f",
            square=True, linewidths=0.5)
ax.set_title("Correlation Heatmap (Numeric Variables)")
save_fig(fig, "18_correlation_heatmap.png")

record("Multivariate - Correlation",
       f"Top correlations with Survived:\n{corr['Survived'].sort_values(ascending=False).to_string()}")

# Crosstab: Pclass x Sex x Survived (counts)
ct = pd.crosstab([train_clean["Pclass"], train_clean["Sex"]], train_clean["Survived"])
record("Multivariate - Crosstab",
       f"Crosstab (Pclass, Sex) x Survived:\n{ct.to_string()}")

# Pivot: survival rate by Pclass and Embarked
pvt = train_clean.pivot_table(values="Survived", index="Pclass", columns="Embarked", aggfunc="mean")
record("Multivariate - Pivot",
       f"Survival rate pivot (Pclass x Embarked):\n{pvt.round(3).to_string()}")

# Pairplot subset (PairGrid object needs special handling)
pair_cols = ["Survived", "Pclass", "Age", "Fare", "FamilySize"]
pair_grid = sns.pairplot(train_clean[pair_cols], hue="Survived", palette="Set2", diag_kind="hist")
pair_grid.fig.suptitle("Pairplot of Numeric Variables (hue = Survived)", y=1.02)
pair_path = os.path.join(PLOTS_DIR, "19_pairplot.png")
pair_grid.fig.savefig(pair_path, dpi=150, bbox_inches="tight")
plt.close(pair_grid.fig)
print(f"Saved pairplot to: {pair_path}")

# ---------------------------------------------------------------------------
# 5. KEY FINDINGS / PATTERNS
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("5. KEY FINDINGS & PATTERNS IDENTIFIED")
print("=" * 80)

findings = [
    "1. Gender is the strongest predictor: females had a much higher survival rate than males.",
    "2. Passenger class matters: 1st class passengers had the highest survival rate, 3rd class the lowest.",
    "3. Age matters: children had a higher survival rate than adults/seniors.",
    "4. Embarkation port: passengers from Cherbourg (C) had higher survival than Queenstown (Q) or Southampton (S).",
    "5. Family size: passengers with small families (2-4) tended to survive more than those alone or in very large groups.",
    "6. Fare: higher fare (proxy for class/wealth) correlates with higher survival.",
    "7. Title: titles like 'Mrs', 'Miss', and rare/noble titles show higher survival; 'Mr' shows lower survival.",
    "8. Missing data handled: Age (median by Sex+Pclass), Embarked (mode), Fare (median by Pclass), Cabin (deck extracted).",
    "9. Fare is right-skewed; log transformation recommended for modeling.",
    "10. No duplicate rows were found; outliers in Fare retained as they may reflect genuine high fares.",
]
for f in findings:
    print(f)
    eda_summary.append(f)

# ---------------------------------------------------------------------------
# 6. SAVE REPORT
# ---------------------------------------------------------------------------
report_path = os.path.join(OUTPUT_DIR, "eda_summary_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(eda_summary))

print(f"\nEDA summary report saved to: {report_path}")
print(f"Plots saved to: {PLOTS_DIR}")
print(f"Cleaned data saved to: {OUTPUT_DIR}")
print("\n================================================================================")
print("ANALYSIS COMPLETE")
print("================================================================================")
