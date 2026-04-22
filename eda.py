"""
eda.py — Exploratory Data Analysis for Heart Disease Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving figures
import seaborn as sns
import os

sns.set_theme(style="whitegrid", palette="muted")
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 1. Load & Basic Info
# ─────────────────────────────────────────────
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def basic_info(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("DATASET SHAPE:", df.shape)
    print("=" * 60)
    print("\n--- dtypes ---")
    print(df.dtypes)
    print("\n--- describe ---")
    print(df.describe())


# ─────────────────────────────────────────────
# 2. Missing Values
# ─────────────────────────────────────────────
def missing_value_analysis(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({"Missing Count": missing, "Missing %": pct})
    report = report[report["Missing Count"] > 0]
    if report.empty:
        print("\n✅  No missing values found.")
    else:
        print("\n⚠️  Missing Values:\n", report)

    # Save figure only if there are missing values
    mv = df.isnull().sum()
    mv_nonzero = mv[mv > 0]
    if not mv_nonzero.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        mv_nonzero.plot(kind="bar", ax=ax, color="salmon")
        ax.set_title("Missing Values per Column")
        ax.set_ylabel("Count")
        plt.tight_layout()
        fig.savefig(f"{FIGURES_DIR}/missing_values.png", dpi=120)
        plt.close(fig)
        print(f"   → Figure saved: {FIGURES_DIR}/missing_values.png")
    else:
        print("   → No missing value chart needed (dataset is complete).")
    return report


# ─────────────────────────────────────────────
# 3. Target Distribution
# ─────────────────────────────────────────────
def target_distribution(df: pd.DataFrame, target_col: str) -> None:
    counts = df[target_col].value_counts()
    pct = df[target_col].value_counts(normalize=True) * 100
    print(f"\nTarget column: '{target_col}'")
    print(counts)
    print(pct.round(2))

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(x=target_col, data=df, ax=ax, palette="pastel")
    ax.set_title(f"Target Distribution  (0 = No Disease, 1 = Disease)")
    ax.set_xlabel(target_col)
    ax.set_ylabel("Count")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/target_distribution.png", dpi=120)
    plt.close(fig)
    print(f"   → Figure saved: {FIGURES_DIR}/target_distribution.png")

    # Insight
    majority_pct = pct.max()
    if majority_pct > 60:
        print(f"\n💡 Insight: Dataset is imbalanced ({majority_pct:.1f}% majority). "
              "SMOTE will be applied during preprocessing.")
    else:
        print(f"\n💡 Insight: Dataset is relatively balanced ({majority_pct:.1f}% majority).")


# ─────────────────────────────────────────────
# 4. Correlation Heatmap
# ─────────────────────────────────────────────
def correlation_heatmap(df: pd.DataFrame) -> None:
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", annot=True,
                fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/correlation_heatmap.png", dpi=120)
    plt.close(fig)
    print(f"   → Figure saved: {FIGURES_DIR}/correlation_heatmap.png")

    # Insight: top correlations with target
    if "target" in corr.columns:
        top = corr["target"].drop("target").abs().sort_values(ascending=False).head(5)
        print("\n💡 Top features correlated with target:")
        print(top.to_string())


# ─────────────────────────────────────────────
# 5. Feature Distributions (Histograms)
# ─────────────────────────────────────────────
def feature_distributions(df: pd.DataFrame, target_col: str) -> None:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != target_col]

    n = len(num_cols)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        axes[i].hist(df[col].dropna(), bins=25, color="steelblue", edgecolor="white")
        axes[i].set_title(col)
        axes[i].set_xlabel("")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Feature Distributions", y=1.01, fontsize=14)
    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/feature_distributions.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"   → Figure saved: {FIGURES_DIR}/feature_distributions.png")


# ─────────────────────────────────────────────
# 6. Boxplots by Target
# ─────────────────────────────────────────────
def boxplots_by_target(df: pd.DataFrame, target_col: str) -> None:
    # Separate continuous vs categorical
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target_col]

    # Heuristic: categorical = low unique values
    categorical_cols = [c for c in numeric_cols if df[c].nunique() <= 10]
    continuous_cols  = [c for c in numeric_cols if c not in categorical_cols]

    # ─────────────────────────────
    # 1. BOXPLOTS (continuous only)
    # ─────────────────────────────
    if continuous_cols:
        n = len(continuous_cols)
        cols = 4
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        axes = axes.flatten()

        for i, col in enumerate(continuous_cols):
            sns.boxplot(x=target_col, y=col, data=df, ax=axes[i], palette="pastel")
            axes[i].set_title(col)

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.suptitle("Continuous Features by Target", y=1.01, fontsize=14)
        plt.tight_layout()
        fig.savefig(f"{FIGURES_DIR}/boxplots_by_target.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"   → Figure saved: {FIGURES_DIR}/boxplots_by_target.png")

    # ─────────────────────────────
    # 2. COUNTPLOTS (categorical)
    # ─────────────────────────────
    if categorical_cols:
        n = len(categorical_cols)
        cols = 4
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        axes = axes.flatten()

        for i, col in enumerate(categorical_cols):
            sns.countplot(x=col, hue=target_col, data=df, ax=axes[i], palette="pastel")
            axes[i].set_title(col)

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.suptitle("Categorical Features by Target", y=1.01, fontsize=14)
        plt.tight_layout()
        fig.savefig(f"{FIGURES_DIR}/countplots_by_target.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"   → Figure saved: {FIGURES_DIR}/countplots_by_target.png")
# ─────────────────────────────────────────────
# 7. Main EDA Runner
# ─────────────────────────────────────────────
def run_eda(csv_path: str, target_col: str = "target") -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("       HEART DISEASE — EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    df = load_data(csv_path)

    basic_info(df)
    missing_value_analysis(df)
    target_distribution(df, target_col)
    correlation_heatmap(df)
    feature_distributions(df, target_col)
    boxplots_by_target(df, target_col)

    print("\n✅  EDA complete. Figures saved to:", FIGURES_DIR)
    return df


if __name__ == "__main__":
    run_eda("heart.csv")
