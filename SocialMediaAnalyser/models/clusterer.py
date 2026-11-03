"""
models/clusterer.py
K-Means Persona Clustering for social media users.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

PERSONA_MAP = {
    0: {"name": "🔥 Viral Creators",     "color": "#7C3AED", "desc": "High likes, shares & views"},
    1: {"name": "💬 Active Commenters",  "color": "#10B981", "desc": "High comment-to-view ratio"},
    2: {"name": "👁️ Passive Scrollers",  "color": "#F59E0B", "desc": "High views, low interactions"},
    3: {"name": "📤 Share Champions",    "color": "#3B82F6", "desc": "Above-average share rate"},
    4: {"name": "⭐ Niche Enthusiasts",  "color": "#EF4444", "desc": "Small reach, high engagement"},
}

CLUSTER_FEATURES = ["Likes","Comments","Shares","Saves","Clicks",
                    "Views","Engagement_Rate","Virality_Score","Click_Rate"]


class PersonaClusterer:
    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.kmeans     = KMeans(n_clusters=n_clusters, random_state=42, n_init=15)
        self.scaler     = StandardScaler()
        self.pca        = PCA(n_components=2, random_state=42)
        self.is_trained = False
        self.sil_score  = None

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df[CLUSTER_FEATURES].dropna().copy()
        scaled = self.scaler.fit_transform(data)
        labels = self.kmeans.fit_predict(scaled)
        pca_coords = self.pca.fit_transform(scaled)

        # Silhouette score
        try:
            self.sil_score = round(silhouette_score(scaled, labels, sample_size=1000), 4)
        except Exception:
            self.sil_score = None

        df_out = df.loc[data.index].copy()
        df_out["Cluster"]   = labels
        df_out["Persona"]   = df_out["Cluster"].map(lambda c: PERSONA_MAP.get(c, {}).get("name", f"Cluster {c}"))
        df_out["PCA_X"]     = pca_coords[:, 0]
        df_out["PCA_Y"]     = pca_coords[:, 1]
        self.is_trained = True
        return df_out

    def summary(self, df_c: pd.DataFrame) -> pd.DataFrame:
        return df_c.groupby("Persona").agg(
            Count        = ("Persona",         "count"),
            Avg_Likes    = ("Likes",            "mean"),
            Avg_Comments = ("Comments",         "mean"),
            Avg_Shares   = ("Shares",           "mean"),
            Avg_Eng_Rate = ("Engagement_Rate",  "mean"),
            Avg_Views    = ("Views",            "mean"),
            Avg_Virality = ("Virality_Score",   "mean"),
            Avg_Click    = ("Click_Rate",        "mean"),
        ).round(3).reset_index()

    def age_distribution(self, df_c: pd.DataFrame) -> pd.DataFrame:
        return df_c.groupby(["Age_Group","Persona"]).size().reset_index(name="Count")

    def platform_distribution(self, df_c: pd.DataFrame) -> pd.DataFrame:
        return df_c.groupby(["Platform","Persona"]).size().reset_index(name="Count")

    @staticmethod
    def persona_colors():
        return {v["name"]: v["color"] for v in PERSONA_MAP.values()}

    @staticmethod
    def persona_info():
        return PERSONA_MAP
