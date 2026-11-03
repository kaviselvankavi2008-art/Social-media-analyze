"""
utils/charts.py
20 interactive Plotly charts — Light Modern White/Purple theme.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# ── Design Tokens ─────────────────────────────────────────────────────────
PRIMARY       = "#7C3AED"
PRIMARY_L     = "#A78BFA"
PRIMARY_XL    = "#EDE9FE"
ACCENT_GREEN  = "#10B981"
ACCENT_AMBER  = "#F59E0B"
ACCENT_BLUE   = "#3B82F6"
ACCENT_RED    = "#EF4444"
ACCENT_PINK   = "#EC4899"
BG            = "#FFFFFF"
SURFACE       = "#FAFAFA"
BORDER        = "#E5E7EB"
TEXT          = "#111827"
TEXT_MUTED    = "#6B7280"

PALETTE = [PRIMARY, ACCENT_GREEN, ACCENT_AMBER, ACCENT_BLUE,
           ACCENT_RED, ACCENT_PINK, "#14B8A6", "#8B5CF6",
           "#F97316", "#06B6D4"]

AGE_GROUPS  = ["13-17","18-24","25-34","35-44","45-54","55+"]
MONTHS      = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

BASE_LAYOUT = dict(
    paper_bgcolor = BG,
    plot_bgcolor  = SURFACE,
    font = dict(family="DM Sans, sans-serif", color=TEXT, size=12),
    margin = dict(l=44, r=20, t=52, b=44),
    legend = dict(
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor=BORDER, borderwidth=1,
        font=dict(size=11),
    ),
    colorway = PALETTE,
    hoverlabel = dict(
        bgcolor="white", bordercolor=BORDER,
        font=dict(size=12, color=TEXT, family="DM Sans"),
    ),
)

def _fmt(fig, title="", xt="", yt=""):
    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text=title, font=dict(size=15, color=TEXT, family="Syne, sans-serif"), x=0),
        xaxis_title=xt, yaxis_title=yt,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#F3F4F6", linecolor=BORDER, zeroline=False,
                     tickfont=dict(size=11))
    fig.update_yaxes(showgrid=True, gridcolor="#F3F4F6", linecolor=BORDER, zeroline=False,
                     tickfont=dict(size=11))
    return fig

def _hex_rgba(h, a):
    r,g,b = int(h[1:3],16), int(h[3:5],16), int(h[5:7],16)
    return f"rgba({r},{g},{b},{a})"


# ── 1. Gradient Bar — Avg Engagement by Age ───────────────────────────────
def bar_eng_by_age(df):
    d = df.groupby("Age_Group")["Engagement_Rate"].mean().reindex(AGE_GROUPS).reset_index()
    d.columns = ["Age_Group","ER"]
    colors = [_hex_rgba(PRIMARY, 0.35 + i*0.13) for i in range(len(d))]
    fig = go.Figure(go.Bar(
        x=d["Age_Group"], y=d["ER"],
        marker_color=colors, marker_line_color=PRIMARY, marker_line_width=1.5,
        text=d["ER"].round(2).astype(str)+"%", textposition="outside",
        textfont=dict(size=11, color=PRIMARY),
        hovertemplate="<b>%{x}</b><br>Avg Eng Rate: %{y:.3f}%<extra></extra>",
    ))
    return _fmt(fig, "📊 Avg Engagement Rate by Age Group", "Age Group", "Engagement Rate (%)")


# ── 2. Heatmap — Platform × Age Group ────────────────────────────────────
def heatmap_platform_age(df):
    pivot = df.groupby(["Age_Group","Platform"])["Engagement_Rate"]\
              .mean().unstack().reindex(AGE_GROUPS)
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale=[[0,PRIMARY_XL],[0.5,PRIMARY_L],[1,PRIMARY]],
        text=np.round(pivot.values,2),
        texttemplate="%{text}%", textfont=dict(size=11),
        hovertemplate="<b>%{y} × %{x}</b><br>Eng Rate: %{z:.3f}%<extra></extra>",
        colorbar=dict(title="Eng%", tickfont=dict(size=10)),
    ))
    return _fmt(fig, "🔥 Engagement Heatmap: Age Group × Platform","Platform","Age Group")


# ── 3. Donut — Age Distribution ───────────────────────────────────────────
def donut_age(df):
    counts = df["Age_Group"].value_counts().reindex(AGE_GROUPS)
    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values, hole=0.58,
        marker=dict(colors=PALETTE[:len(AGE_GROUPS)], line=dict(color=BG, width=3)),
        textinfo="percent+label", textfont=dict(size=12),
        hovertemplate="<b>%{label}</b><br>Posts: %{value:,}<br>%{percent}<extra></extra>",
    ))
    fig.add_annotation(text="<b>Age</b><br>Groups", x=0.5, y=0.5, showarrow=False,
                       font=dict(size=13, color=TEXT, family="Syne"))
    return _fmt(fig, "👥 Post Distribution by Age Group")


# ── 4. Grouped Bar — Content × Age ────────────────────────────────────────
def grouped_content_age(df):
    d = df.groupby(["Age_Group","Content_Type"])["Likes"].mean().reset_index()
    fig = px.bar(d, x="Age_Group", y="Likes", color="Content_Type",
                 barmode="group", color_discrete_sequence=PALETTE,
                 labels={"Likes":"Avg Likes","Age_Group":"Age Group"})
    fig.update_traces(marker_line_width=0.5, marker_line_color="white")
    return _fmt(fig, "📱 Avg Likes by Content Type & Age Group","Age Group","Avg Likes")


# ── 5. Box Plot — Engagement Distribution ────────────────────────────────
def box_eng_age(df):
    fig = go.Figure()
    for i, age in enumerate(AGE_GROUPS):
        sub = df[df["Age_Group"]==age]["Engagement_Rate"]
        c = PALETTE[i % len(PALETTE)]
        fig.add_trace(go.Box(
            y=sub, name=age, marker_color=c, line_color=c,
            fillcolor=_hex_rgba(c, 0.2), boxmean="sd",
            hovertemplate=f"<b>{age}</b><br>%{{y:.3f}}%<extra></extra>",
        ))
    return _fmt(fig, "📦 Engagement Rate Distribution by Age Group","Age Group","Engagement Rate (%)")


# ── 6. Area — Monthly Trend ────────────────────────────────────────────────
def area_monthly(df):
    d = df.groupby(["Month","Age_Group"])["Engagement_Rate"].mean().reset_index()
    d["Month"] = pd.Categorical(d["Month"], categories=MONTHS, ordered=True)
    d = d.sort_values("Month")
    fig = go.Figure()
    for i, age in enumerate(AGE_GROUPS):
        sub = d[d["Age_Group"]==age]
        c = PALETTE[i % len(PALETTE)]
        fig.add_trace(go.Scatter(
            x=sub["Month"], y=sub["Engagement_Rate"], name=age,
            mode="lines+markers",
            line=dict(color=c, width=2.5),
            marker=dict(size=7, color=c),
            fill="tozeroy", fillcolor=_hex_rgba(c, 0.07),
            hovertemplate=f"<b>{age}</b><br>%{{x}}: %{{y:.3f}}%<extra></extra>",
        ))
    fig.update_layout(hovermode="x unified")
    return _fmt(fig, "📅 Monthly Engagement Trend by Age Group","Month","Avg Engagement Rate (%)")


# ── 7. Scatter — Likes vs Comments ───────────────────────────────────────
def scatter_likes_comments(df):
    s = df.sample(min(1000, len(df)), random_state=42)
    fig = px.scatter(s, x="Likes", y="Comments", color="Age_Group",
                     size="Views", opacity=0.65, size_max=18,
                     color_discrete_sequence=PALETTE,
                     hover_data={"Platform":True,"Content_Type":True,"Engagement_Rate":":.3f"})
    return _fmt(fig, "🔗 Likes vs Comments (sized by Views)","Likes","Comments")


# ── 8. Radar — Platform per Age Group ─────────────────────────────────────
def radar_platform_age(df):
    plats = df["Platform"].unique().tolist()
    fig = go.Figure()
    for i, age in enumerate(AGE_GROUPS):
        sub = df[df["Age_Group"]==age].groupby("Platform")["Engagement_Rate"].mean()
        vals = [sub.get(p,0) for p in plats] + [sub.get(plats[0],0)]
        c = PALETTE[i%len(PALETTE)]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=plats+[plats[0]], name=age, fill="toself",
            line_color=c, fillcolor=_hex_rgba(c, 0.12),
            hovertemplate=f"<b>{age}</b><br>%{{theta}}: %{{r:.3f}}%<extra></extra>",
        ))
    fig.update_layout(**BASE_LAYOUT,
        title=dict(text="🕸️ Platform Engagement Radar by Age Group",
                   font=dict(size=15, color=TEXT, family="Syne"), x=0),
        polar=dict(bgcolor=SURFACE,
                   radialaxis=dict(visible=True, gridcolor=BORDER),
                   angularaxis=dict(gridcolor=BORDER)),
    )
    return fig


# ── 9. Stacked Bar — Platform Usage ──────────────────────────────────────
def stacked_platform_usage(df):
    d = df.groupby(["Age_Group","Platform"]).size().reset_index(name="Count")
    fig = px.bar(d, x="Age_Group", y="Count", color="Platform",
                 color_discrete_sequence=PALETTE,
                 labels={"Count":"Posts","Age_Group":"Age Group"})
    fig.update_layout(barmode="stack")
    fig.update_traces(marker_line_width=0.5, marker_line_color="white")
    return _fmt(fig, "📡 Platform Usage Distribution by Age Group","Age Group","Number of Posts")


# ── 10. Heatmap — Gender × Age ────────────────────────────────────────────
def heatmap_gender_age(df):
    pivot = df.groupby(["Age_Group","Gender"])["Engagement_Rate"]\
              .mean().unstack().reindex(AGE_GROUPS)
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale=[[0,PRIMARY_XL],[0.5,PRIMARY_L],[1,PRIMARY]],
        text=np.round(pivot.values,2), texttemplate="%{text}%",
        textfont=dict(size=12),
        hovertemplate="<b>%{y} — %{x}</b><br>Eng Rate: %{z:.3f}%<extra></extra>",
        colorbar=dict(title="Eng%"),
    ))
    return _fmt(fig, "♀️♂️ Engagement Heatmap: Age Group × Gender","Gender","Age Group")


# ── 11. Bar — Region × Age Group ─────────────────────────────────────────
def bar_region_age(df):
    d = df.groupby(["Region","Age_Group"])["Engagement_Rate"].mean().reset_index()
    fig = px.bar(d, x="Region", y="Engagement_Rate", color="Age_Group",
                 barmode="group", color_discrete_sequence=PALETTE,
                 labels={"Engagement_Rate":"Avg Engagement Rate (%)","Region":"Region"})
    fig.update_traces(marker_line_width=0.5, marker_line_color="white")
    return _fmt(fig, "🌍 Engagement Rate by Region & Age Group","Region","Avg Engagement Rate (%)")


# ── 12. Violin — Views by Age ──────────────────────────────────────────────
def violin_views_age(df):
    fig = go.Figure()
    for i, age in enumerate(AGE_GROUPS):
        c = PALETTE[i % len(PALETTE)]
        fig.add_trace(go.Violin(
            y=df[df["Age_Group"]==age]["Views"], name=age,
            line_color=c, fillcolor=_hex_rgba(c, 0.25),
            meanline_visible=True, box_visible=True,
            hovertemplate=f"<b>{age}</b><br>%{{y:,.0f}} views<extra></extra>",
        ))
    return _fmt(fig, "🎻 View Count Distribution by Age Group","Age Group","Views")


# ── 13. Hourly Heatmap ────────────────────────────────────────────────────
def heatmap_hourly(df):
    pivot = df.groupby(["Hour","Age_Group"])["Engagement_Rate"]\
              .mean().unstack().reindex(range(24))
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(),
        y=[f"{h:02d}:00" for h in range(24)],
        colorscale=[[0,PRIMARY_XL],[0.5,PRIMARY_L],[1,PRIMARY]],
        hovertemplate="<b>%{x}</b><br>Hour %{y}<br>Eng: %{z:.3f}%<extra></extra>",
        colorbar=dict(title="Eng%"),
    ))
    return _fmt(fig, "⏰ Hourly Engagement Heatmap by Age Group","Age Group","Hour of Day")


# ── 14. Correlation Heatmap ────────────────────────────────────────────────
def heatmap_correlation(df):
    cols = ["Likes","Comments","Shares","Saves","Clicks",
            "Views","Engagement_Rate","Virality_Score","Click_Rate"]
    corr = df[cols].corr().round(3)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale="RdBu", zmid=0,
        text=corr.values.round(2), texttemplate="%{text}",
        textfont=dict(size=9),
        hovertemplate="<b>%{x} × %{y}</b><br>r = %{z:.3f}<extra></extra>",
        colorbar=dict(title="r"),
    ))
    return _fmt(fig, "🔢 Metric Correlation Matrix")


# ── 15. Content Type — Engagement Comparison ─────────────────────────────
def bar_content_eng(df):
    d = df.groupby("Content_Type")["Engagement_Rate"].mean().reset_index().sort_values("Engagement_Rate", ascending=True)
    fig = go.Figure(go.Bar(
        x=d["Engagement_Rate"], y=d["Content_Type"], orientation="h",
        marker_color=[_hex_rgba(PRIMARY, 0.4+i*0.12) for i in range(len(d))],
        marker_line_color=PRIMARY, marker_line_width=1,
        text=d["Engagement_Rate"].round(3).astype(str)+"%",
        textposition="outside", textfont=dict(size=11, color=PRIMARY),
        hovertemplate="<b>%{y}</b><br>Avg Eng: %{x:.3f}%<extra></extra>",
    ))
    return _fmt(fig, "🎬 Avg Engagement Rate by Content Type","Engagement Rate (%)","")


# ── 16. Bubble — Age Group Performance Matrix ─────────────────────────────
def bubble_age_matrix(df):
    d = df.groupby("Age_Group").agg(
        Avg_Likes        = ("Likes",           "mean"),
        Avg_Eng_Rate     = ("Engagement_Rate", "mean"),
        Avg_Views        = ("Views",           "mean"),
    ).reindex(AGE_GROUPS).reset_index()
    fig = go.Figure(go.Scatter(
        x=d["Avg_Likes"], y=d["Avg_Eng_Rate"],
        mode="markers+text",
        text=d["Age_Group"], textposition="top center",
        textfont=dict(size=11, color=TEXT),
        marker=dict(
            size=d["Avg_Views"] / d["Avg_Views"].max() * 60 + 15,
            color=PALETTE[:len(d)],
            line=dict(color=BG, width=2), opacity=0.85,
        ),
        hovertemplate="<b>%{text}</b><br>Avg Likes: %{x:,.0f}<br>Eng Rate: %{y:.3f}%<extra></extra>",
    ))
    return _fmt(fig, "🫧 Age Group Performance Bubble Chart","Avg Likes","Avg Engagement Rate (%)")


# ── 17. Funnel — Avg Interaction Metrics by Age ───────────────────────────
def funnel_interactions(df):
    d = df.groupby("Age_Group").agg(
        Views    = ("Views",    "mean"),
        Likes    = ("Likes",    "mean"),
        Comments = ("Comments", "mean"),
        Shares   = ("Shares",   "mean"),
        Saves    = ("Saves",    "mean"),
    ).reindex(AGE_GROUPS)

    fig = go.Figure()
    stages = ["Views","Likes","Comments","Shares","Saves"]
    for i, age in enumerate(AGE_GROUPS):
        fig.add_trace(go.Funnel(
            name=age, y=stages,
            x=[d.loc[age,s] for s in stages],
            textinfo="value+percent initial",
            marker=dict(color=_hex_rgba(PALETTE[i%len(PALETTE)], 0.75)),
            visible=(i==0),
        ))
    buttons = [dict(
        label=age, method="update",
        args=[{"visible": [j==i for j in range(len(AGE_GROUPS))]},
              {"title": f"🔻 Interaction Funnel — {age}"}]
    ) for i, age in enumerate(AGE_GROUPS)]
    fig.update_layout(**BASE_LAYOUT,
        title=dict(text=f"🔻 Interaction Funnel — {AGE_GROUPS[0]}",
                   font=dict(size=15, color=TEXT, family="Syne"), x=0),
        updatemenus=[dict(buttons=buttons, direction="down",
                          showactive=True, x=1.0, y=1.15)],
    )
    return fig


# ── 18. Feature Importance ────────────────────────────────────────────────
def bar_feature_importance(imp_df):
    fig = go.Figure(go.Bar(
        x=imp_df["Importance"], y=imp_df["Feature"], orientation="h",
        marker_color=[_hex_rgba(PRIMARY, 0.35 + i*0.09) for i in range(len(imp_df))],
        marker_line_color=PRIMARY, marker_line_width=1,
        text=imp_df["Importance"].round(4), textposition="outside",
        textfont=dict(size=10, color=PRIMARY),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return _fmt(fig, "🌲 Feature Importance — Random Forest","Importance Score","")


# ── 19. Cluster Scatter (PCA) ─────────────────────────────────────────────
def scatter_clusters_pca(df_c):
    s = df_c.sample(min(800, len(df_c)), random_state=42)
    colors = {p["name"]: p["color"] for p in __import__("models.clusterer",fromlist=["PERSONA_MAP"]).PERSONA_MAP.values()}
    fig = px.scatter(s, x="PCA_X", y="PCA_Y", color="Persona",
                     size="Engagement_Rate", opacity=0.7, size_max=16,
                     color_discrete_map=colors,
                     hover_data={"Age_Group":True,"Platform":True,"Content_Type":True,
                                 "Engagement_Rate":":.3f"})
    return _fmt(fig, "🔮 K-Means Persona Clusters (PCA 2D)","PCA Component 1","PCA Component 2")


# ── 20. Realtime Bar ──────────────────────────────────────────────────────
def realtime_bar(live_df, tick):
    d = live_df.groupby("Age_Group")["Engagement_Rate"].mean().reindex(AGE_GROUPS).reset_index()
    d.columns = ["Age_Group","ER"]
    fig = go.Figure(go.Bar(
        x=d["Age_Group"], y=d["ER"],
        marker_color=PALETTE[:len(d)],
        marker_line_color="white", marker_line_width=0.8,
        text=d["ER"].round(3).astype(str)+"%", textposition="outside",
        textfont=dict(size=11, color=PRIMARY),
        hovertemplate="<b>%{x}</b><br>Live Eng: %{y:.3f}%<extra></extra>",
    ))
    ymax = d["ER"].max() * 1.35 if not d["ER"].isna().all() else 5
    return _fmt(fig,
        f"🔴 Live Engagement Feed — Update #{tick}  |  Records: {len(live_df):,}",
        "Age Group","Engagement Rate (%)")
