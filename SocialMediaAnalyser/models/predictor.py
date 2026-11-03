"""
models/predictor.py
Random Forest Engagement Rate Predictor with full evaluation metrics.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

FEATURES = ["Platform","Age_Group","Content_Type","Gender",
            "Region","DayOfWeek","Hour","Followers","Quarter"]
TARGET   = "Engagement_Rate"


class EngagementPredictor:
    def __init__(self):
        self.model      = RandomForestRegressor(
            n_estimators=200, max_depth=12,
            min_samples_split=4, random_state=42, n_jobs=-1
        )
        self.encoders   = {}
        self.is_trained = False
        self.metrics    = {}
        self.feature_names = FEATURES

    # ── Encoding ─────────────────────────────────────────────────────────
    def _encode(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        cat_cols = ["Platform","Age_Group","Content_Type","Gender",
                    "Region","DayOfWeek","Quarter"]
        df = df.copy()
        for col in cat_cols:
            if col not in df.columns:
                continue
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
            else:
                le = self.encoders.get(col)
                if le is not None:
                    df[col] = df[col].astype(str).apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else 0
                    )
        return df

    # ── Train ─────────────────────────────────────────────────────────────
    def train(self, df: pd.DataFrame) -> dict:
        data = df[FEATURES + [TARGET]].dropna().copy()
        data = self._encode(data, fit=True)

        X, y = data[FEATURES], data[TARGET]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model.fit(X_tr, y_tr)
        preds = self.model.predict(X_te)
        cv    = cross_val_score(self.model, X, y, cv=5, scoring="r2")

        self.metrics = {
            "MAE":      round(mean_absolute_error(y_te, preds), 4),
            "RMSE":     round(np.sqrt(mean_squared_error(y_te, preds)), 4),
            "R2":       round(r2_score(y_te, preds), 4),
            "CV_R2":    round(cv.mean(), 4),
            "CV_Std":   round(cv.std(), 4),
            "Accuracy": round(max(0, r2_score(y_te, preds)) * 100, 2),
            "TrainSize":len(X_tr),
            "TestSize": len(X_te),
        }
        self.is_trained = True
        return self.metrics

    # ── Predict ───────────────────────────────────────────────────────────
    def predict(self, platform, age_group, content_type, gender,
                region, day_of_week, hour, followers, quarter="Q2") -> float:
        if not self.is_trained:
            raise ValueError("Model not trained.")
        row = pd.DataFrame([{
            "Platform": platform, "Age_Group": age_group,
            "Content_Type": content_type, "Gender": gender,
            "Region": region, "DayOfWeek": day_of_week,
            "Hour": hour, "Followers": followers, "Quarter": quarter,
        }])
        row = self._encode(row, fit=False)
        return round(float(self.model.predict(row[FEATURES])[0]), 4)

    # ── Feature Importance ────────────────────────────────────────────────
    def feature_importance(self) -> pd.DataFrame:
        if not self.is_trained:
            return pd.DataFrame(columns=["Feature","Importance"])
        return pd.DataFrame({
            "Feature":    FEATURES,
            "Importance": self.model.feature_importances_,
        }).sort_values("Importance", ascending=False).reset_index(drop=True)

    # ── Rating label ─────────────────────────────────────────────────────
    @staticmethod
    def rating(val: float) -> tuple:
        if val >= 6:   return "🔥 Excellent",  "#10B981", "High Engagement"
        elif val >= 4: return "✅ Good",        "#7C3AED", "Above Average"
        elif val >= 2: return "⚠️ Average",    "#F59E0B", "Moderate"
        else:          return "❌ Low",         "#EF4444", "Below Average"
