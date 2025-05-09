import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def stratify_risk(df):
    # Convert all risk-related columns to numeric (if they aren't already)
    cols = [
        "BH_SPMI", "DX_CNT_PSYCHOSOCIAL_STRESSORS", "DX_CNT_MENTAL_HEALTH",
        "DX_CNT_SUBSTANCE_ABUSE", "DX_CNT_COMPLEX_TRAUMA",
        "DX_IND_MEDICALLY_FRAGILE", "DX_CNT_MEDICALLY_FRAGILE",
        "DX_IND_ESRD", "DX_IND_CKD_ESRD", "CANCER_ACTIVE_IND",
        "_ASTHMA_IND", "_ASCVD_IND", "_CHF_IND", "_CKD_IND", "_COPD_IND"
    ]

    # Safely convert to numeric, coercing errors (e.g., empty strings) to NaN and then fill with 0
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # Risk score formula
    df["risk_score"] = (
        df["BH_SPMI"] * 3 +
        df["DX_CNT_PSYCHOSOCIAL_STRESSORS"] * 2 +
        df["DX_CNT_MENTAL_HEALTH"] * 2 +
        df["DX_CNT_SUBSTANCE_ABUSE"] * 3 +
        df["DX_CNT_COMPLEX_TRAUMA"] * 2 +
        df["DX_IND_MEDICALLY_FRAGILE"] * 4 +
        df["DX_CNT_MEDICALLY_FRAGILE"] * 2 +
        df["DX_IND_ESRD"] * 4 +
        df["DX_IND_CKD_ESRD"] * 3 +
        df["CANCER_ACTIVE_IND"] * 3 +
        df["_ASTHMA_IND"] * 1 +
        df["_ASCVD_IND"] * 2 +
        df["_CHF_IND"] * 3 +
        df["_CKD_IND"] * 3 +
        df["_COPD_IND"] * 2
    )

    # Categorize into risk levels
    df["Risk Category"] = pd.cut(
        df["risk_score"],
        bins=[-1, 10, 20, float("inf")],
        labels=["Low", "Medium", "High"]
    )

    return df


# def filter_members(df, age_range, gender, risk_category):
#     return df[
#         (df["age"].between(age_range[0], age_range[1])) &
#         (df["gender"].isin(gender)) &
#         (df["Risk Category"].isin(risk_category))
#     ]
def filter_members(df, age_range, gender, risk_category):
    return df[
        (df["Age"].between(age_range[0], age_range[1])) &
        (df["Gender"].isin(gender)) &
        (df["Risk Category"].isin(risk_category))
    ]


def plot_risk_distribution(df):
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="Risk Category", palette="coolwarm", ax=ax)
    ax.set_title("Risk Category Distribution")
    return fig
 
