"""
data/preprocess.py
Data cleaning, validation, and aggregation helpers.
"""

import pandas as pd
import numpy as np
from data.generate_data import AGE_GROUPS, PLATFORMS, CONTENT, MONTHS_SHORT


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove outliers and ensure data quality."""
    df = df.copy()
    df = df[df["Views"] > 0]
    df = df[df["Engagement_Rate"] >= 0]
    df = df[df["Engagement_Rate"] <= 100]

    # Cap extreme values at 99th percentile
    for col in ["Likes", "Comments", "Shares", "Views", "Impressions"]:
        cap = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=cap)
    return df.reset_index(drop=True)


# ── Aggregations ──────────────────────────────────────────────────────────
def agg_by_age(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("Age_Group").agg(
        Records          = ("Engagement_Rate", "count"),
        Avg_Eng_Rate     = ("Engagement_Rate", "mean"),
        Median_Eng_Rate  = ("Engagement_Rate", "median"),
        Avg_Likes        = ("Likes",           "mean"),
        Avg_Comments     = ("Comments",        "mean"),
        Avg_Shares       = ("Shares",          "mean"),
        Avg_Saves        = ("Saves",           "mean"),
        Avg_Views        = ("Views",           "mean"),
        Avg_Click_Rate   = ("Click_Rate",      "mean"),
        Avg_Virality     = ("Virality_Score",  "mean"),
        Total_Likes      = ("Likes",           "sum"),
        Total_Views      = ("Views",           "sum"),
    ).round(3).reindex(AGE_GROUPS).reset_index()


def agg_age_platform(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["Age_Group","Platform"])["Engagement_Rate"]\
             .mean().unstack().reindex(AGE_GROUPS).round(3)


def agg_age_content(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["Age_Group","Content_Type"])["Likes"]\
             .mean().unstack().reindex(AGE_GROUPS).round(1)


def agg_monthly(df: pd.DataFrame) -> pd.DataFrame:
    data = df.groupby(["Month","Age_Group"])["Engagement_Rate"]\
             .mean().reset_index()
    data["Month"] = pd.Categorical(data["Month"], categories=MONTHS_SHORT, ordered=True)
    return data.sort_values("Month")


def agg_hourly(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["Hour","Age_Group"])["Engagement_Rate"]\
             .mean().reset_index()


def agg_platform_dist(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["Age_Group","Platform"]).size()\
             .reset_index(name="Count")


def agg_gender(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["Age_Group","Gender"])["Engagement_Rate"]\
             .mean().reset_index()


def agg_region(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["Region","Age_Group"])["Engagement_Rate"]\
             .mean().reset_index()


def agg_content_eng(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("Content_Type").agg(
        Avg_Eng    = ("Engagement_Rate","mean"),
        Avg_Likes  = ("Likes",         "mean"),
        Avg_Views  = ("Views",         "mean"),
        Count      = ("Likes",         "count"),
    ).round(3).reset_index()


def top_posts(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    cols = ["Platform","Age_Group","Content_Type","Region","Gender",
            "Likes","Comments","Shares","Views","Engagement_Rate",
            "Click_Rate","Virality_Score"]
    return df.nlargest(n, "Engagement_Rate")[cols].round(3).reset_index(drop=True)


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["Likes","Comments","Shares","Saves","Clicks",
            "Views","Impressions","Reach","Followers",
            "Engagement_Rate","Click_Rate","Virality_Score"]
    return df[cols].corr().round(3)


def kpi_summary(df: pd.DataFrame) -> dict:
    return {
        "total_records":  len(df),
        "avg_eng":        round(df["Engagement_Rate"].mean(), 3),
        "avg_likes":      round(df["Likes"].mean(), 1),
        "avg_views":      round(df["Views"].mean(), 1),
        "avg_virality":   round(df["Virality_Score"].mean(), 4),
        "top_age":        df.groupby("Age_Group")["Engagement_Rate"].mean().idxmax(),
        "top_platform":   df.groupby("Platform")["Engagement_Rate"].mean().idxmax(),
        "top_content":    df.groupby("Content_Type")["Engagement_Rate"].mean().idxmax(),
        "top_region":     df.groupby("Region")["Engagement_Rate"].mean().idxmax(),
        "total_likes":    int(df["Likes"].sum()),
        "total_views":    int(df["Views"].sum()),
    }
