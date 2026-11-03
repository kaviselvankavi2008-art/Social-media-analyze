"""
data/generate_data.py
Generates realistic synthetic social media engagement dataset.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# ── Constants ──────────────────────────────────────────────────────────────
PLATFORMS    = ["Instagram", "Facebook", "Twitter/X", "TikTok", "YouTube", "LinkedIn"]
AGE_GROUPS   = ["13-17", "18-24", "25-34", "35-44", "45-54", "55+"]
CONTENT      = ["Video", "Image", "Story", "Reel", "Article", "Live"]
GENDERS      = ["Male", "Female", "Non-binary"]
REGIONS      = ["North America", "Europe", "Asia", "South America", "Middle East", "Africa"]
DAYS         = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
MONTHS_SHORT = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

AGE_BIAS = {
    "13-17": 120, "18-24": 200, "25-34": 180,
    "35-44": 140, "45-54":  90, "55+":   60,
}
PLATFORM_MULT = {
    "Instagram": 1.4, "Facebook": 1.0, "Twitter/X": 0.9,
    "TikTok":    1.8, "YouTube":  1.3, "LinkedIn":  0.7,
}
CONTENT_MULT = {
    "Video": 1.5, "Reel": 1.6, "Live": 1.3,
    "Image": 1.0, "Story": 0.8, "Article": 0.7,
}
REGION_MULT = {
    "North America": 1.3, "Europe": 1.1, "Asia": 1.5,
    "South America": 0.9, "Middle East": 1.0, "Africa": 0.8,
}


def generate_dataset(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate n rows of synthetic social media engagement data."""
    np.random.seed(seed)
    random.seed(seed)
    records = []
    base_date = datetime(2024, 1, 1)

    for _ in range(n):
        age    = random.choice(AGE_GROUPS)
        plat   = random.choice(PLATFORMS)
        cont   = random.choice(CONTENT)
        gender = random.choice(GENDERS)
        region = random.choice(REGIONS)
        days_offset = random.randint(0, 364)
        hour   = random.randint(0, 23)
        date   = base_date + timedelta(days=days_offset, hours=hour)

        mult   = PLATFORM_MULT[plat] * CONTENT_MULT[cont] * REGION_MULT[region]
        base   = AGE_BIAS[age] * mult

        likes    = max(0, int(np.random.normal(base,          base * 0.40)))
        comments = max(0, int(np.random.normal(base * 0.15,   base * 0.10)))
        shares   = max(0, int(np.random.normal(base * 0.08,   base * 0.06)))
        saves    = max(0, int(np.random.normal(base * 0.05,   base * 0.04)))
        clicks   = max(0, int(np.random.normal(base * 0.20,   base * 0.10)))
        views    = max(likes + 1, int(np.random.normal(base * 6, base * 2)))
        followers= random.randint(100, 500_000)
        impressions = max(views, int(views * np.random.uniform(1.2, 2.5)))
        reach    = max(views, int(views * np.random.uniform(0.9, 1.5)))

        eng_rate      = round(((likes + comments + shares + saves) / views * 100), 4)
        click_rate    = round((clicks / impressions * 100), 4) if impressions > 0 else 0
        reach_rate    = round((reach / followers * 100), 4)    if followers > 0 else 0
        virality      = round((shares / views * 100), 4)       if views > 0 else 0

        records.append({
            "Date":           date,
            "Month":          date.strftime("%b"),
            "MonthNum":       date.month,
            "Week":           date.isocalendar()[1],
            "Hour":           hour,
            "DayOfWeek":      date.strftime("%A"),
            "Quarter":        f"Q{((date.month - 1) // 3) + 1}",
            "Platform":       plat,
            "Age_Group":      age,
            "Gender":         gender,
            "Region":         region,
            "Content_Type":   cont,
            "Likes":          likes,
            "Comments":       comments,
            "Shares":         shares,
            "Saves":          saves,
            "Clicks":         clicks,
            "Views":          views,
            "Impressions":    impressions,
            "Reach":          reach,
            "Followers":      followers,
            "Engagement_Rate":eng_rate,
            "Click_Rate":     click_rate,
            "Reach_Rate":     reach_rate,
            "Virality_Score": virality,
        })

    df = pd.DataFrame(records)
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def simulate_new_batch(n: int = 60) -> pd.DataFrame:
    """Simulate a real-time batch of new incoming posts."""
    records = []
    now = datetime.now()
    for _ in range(n):
        age   = random.choice(AGE_GROUPS)
        plat  = random.choice(PLATFORMS)
        cont  = random.choice(CONTENT)
        region= random.choice(REGIONS)
        mult  = PLATFORM_MULT[plat] * CONTENT_MULT[cont] * REGION_MULT[region]
        base  = AGE_BIAS[age] * mult
        views = max(500, int(np.random.normal(base * 6, base * 2)))
        likes = max(0,   int(np.random.normal(base,     base * 0.4)))
        cmt   = max(0,   int(np.random.normal(base*0.15,base * 0.1)))
        shr   = max(0,   int(np.random.normal(base*0.08,base * 0.06)))
        sav   = max(0,   int(np.random.normal(base*0.05,base * 0.04)))
        clk   = max(0,   int(np.random.normal(base*0.20,base * 0.10)))
        impr  = max(views, int(views * random.uniform(1.2, 2.5)))
        foll  = random.randint(100, 500_000)
        reach = max(views, int(views * random.uniform(0.9, 1.5)))
        er    = round(((likes + cmt + shr + sav) / views * 100), 4) if views > 0 else 0
        cr    = round((clk / impr * 100), 4) if impr > 0 else 0
        rr    = round((reach / foll * 100), 4) if foll > 0 else 0
        vir   = round((shr / views * 100), 4) if views > 0 else 0

        records.append({
            "Date": now, "Month": now.strftime("%b"),
            "MonthNum": now.month, "Week": now.isocalendar()[1],
            "Hour": now.hour, "DayOfWeek": now.strftime("%A"),
            "Quarter": f"Q{((now.month-1)//3)+1}",
            "Platform": plat, "Age_Group": age,
            "Gender": random.choice(GENDERS),
            "Region": region, "Content_Type": cont,
            "Likes": likes, "Comments": cmt, "Shares": shr,
            "Saves": sav, "Clicks": clk,
            "Views": views, "Impressions": impr,
            "Reach": reach, "Followers": foll,
            "Engagement_Rate": er, "Click_Rate": cr,
            "Reach_Rate": rr, "Virality_Score": vir,
        })
    return pd.DataFrame(records)
