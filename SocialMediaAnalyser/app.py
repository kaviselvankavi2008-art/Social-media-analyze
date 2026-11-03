"""
app.py — Social Media Engagement Analyser
Streamlit + Plotly | Python Data Science | Light Modern White/Purple
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import time, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.generate_data import (
    generate_dataset, simulate_new_batch,
    AGE_GROUPS, PLATFORMS, CONTENT, GENDERS, REGIONS, DAYS,
)
from data.preprocess import (
    clean, agg_by_age, kpi_summary, top_posts, correlation_matrix,
    agg_monthly, agg_hourly, agg_platform_dist, agg_content_eng,
)
from models.predictor  import EngagementPredictor
from models.clusterer  import PersonaClusterer
from utils.charts import (
    bar_eng_by_age, heatmap_platform_age, donut_age,
    grouped_content_age, box_eng_age, area_monthly,
    scatter_likes_comments, radar_platform_age, stacked_platform_usage,
    heatmap_gender_age, bar_region_age, violin_views_age,
    heatmap_hourly, heatmap_correlation, bar_content_eng,
    bubble_age_matrix, funnel_interactions, bar_feature_importance,
    scatter_clusters_pca, realtime_bar,
)

# ────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Social Media Engagement Analyser",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Generating dataset…")
def load_data(seed):
    df = generate_dataset(n=5000, seed=seed)
    return clean(df)

if "seed"       not in st.session_state: st.session_state.seed       = 42
if "df"         not in st.session_state: st.session_state.df         = load_data(42)
if "predictor"  not in st.session_state:
    p = EngagementPredictor()
    st.session_state.pred_metrics = p.train(st.session_state.df)
    st.session_state.predictor    = p
if "clusterer"  not in st.session_state:
    c = PersonaClusterer(5)
    st.session_state.df_cluster   = c.fit(st.session_state.df)
    st.session_state.clusterer    = c
if "live_df"    not in st.session_state: st.session_state.live_df    = st.session_state.df.copy()
if "rt_tick"    not in st.session_state: st.session_state.rt_tick    = 0

# ────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand">📊 EngageIQ<br>'
        '<span>Social Media Analyser</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section">🎛️ Data Filters</div>', unsafe_allow_html=True)
    sel_platforms = st.multiselect("Platforms",     PLATFORMS,   default=PLATFORMS)
    sel_ages      = st.multiselect("Age Groups",    AGE_GROUPS,  default=AGE_GROUPS)
    sel_content   = st.multiselect("Content Types", CONTENT,     default=CONTENT)
    sel_regions   = st.multiselect("Regions",       REGIONS,     default=REGIONS)
    eng_range     = st.slider("Engagement Rate (%)", 0.0, 30.0, (0.0, 30.0), 0.1)

    st.markdown('<div class="sidebar-section">⚙️ Actions</div>', unsafe_allow_html=True)
    if st.button("🔄 Regenerate Dataset", use_container_width=True):
        st.session_state.seed = np.random.randint(1, 99999)
        st.session_state.df   = load_data(st.session_state.seed)
        p = EngagementPredictor()
        st.session_state.pred_metrics = p.train(st.session_state.df)
        st.session_state.predictor    = p
        c = PersonaClusterer(5)
        st.session_state.df_cluster   = c.fit(st.session_state.df)
        st.session_state.clusterer    = c
        st.session_state.live_df      = st.session_state.df.copy()
        st.session_state.rt_tick      = 0
        st.rerun()

    # CSV download
    csv_bytes = st.session_state.df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Full Dataset", csv_bytes,
                       "engagement_data.csv", "text/csv",
                       use_container_width=True)

    st.markdown('<div class="sidebar-section">🤖 Model Metrics</div>', unsafe_allow_html=True)
    m = st.session_state.pred_metrics
    st.metric("R² Score",     m["R2"])
    st.metric("RMSE",         m["RMSE"])
    st.metric("CV R² Score",  m["CV_R2"])
    st.metric("Accuracy",     f"{m['Accuracy']}%")

    st.markdown("---")
    st.markdown(
        "<small style='color:rgba(255,255,255,0.45);'>Built with Python · Streamlit · Plotly<br>"
        "Pandas · Scikit-learn · NumPy<br>© 2024 EngageIQ Analytics</small>",
        unsafe_allow_html=True,
    )

# ────────────────────────────────────────────────────────────────────────────
#  FILTER DATA
# ────────────────────────────────────────────────────────────────────────────
df = st.session_state.df.copy()
df = df[
    df["Platform"].isin(sel_platforms) &
    df["Age_Group"].isin(sel_ages) &
    df["Content_Type"].isin(sel_content) &
    df["Region"].isin(sel_regions) &
    df["Engagement_Rate"].between(*eng_range)
]
df_c = st.session_state.df_cluster.copy()
df_c = df_c[
    df_c["Platform"].isin(sel_platforms) &
    df_c["Age_Group"].isin(sel_ages)
]
kpi = kpi_summary(df)

# ────────────────────────────────────────────────────────────────────────────
#  HERO
# ────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <div class="hero-tag">📊 Data Science Dashboard</div>
  <h1>Social Media Engagement Analyser</h1>
  <p>Real-time insights across age groups, platforms, regions & content types</p>
  <div class="hero-badges">
    <span class="hero-badge">🔢 {kpi['total_records']:,} Records</span>
    <span class="hero-badge">📈 {kpi['avg_eng']:.2f}% Avg Engagement</span>
    <span class="hero-badge">🏆 Top Age: {kpi['top_age']}</span>
    <span class="hero-badge">🥇 Top Platform: {kpi['top_platform']}</span>
    <span class="hero-badge">🎬 Top Content: {kpi['top_content']}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
#  KPI ROW
# ────────────────────────────────────────────────────────────────────────────
k1,k2,k3,k4,k5,k6 = st.columns(6)
kpi_defs = [
    (k1, "purple", "📋", f"{kpi['total_records']:,}",  "Total Records",   ""),
    (k2, "green",  "📈", f"{kpi['avg_eng']:.2f}%",     "Avg Eng. Rate",   ""),
    (k3, "amber",  "❤️",  f"{kpi['avg_likes']:,.0f}",  "Avg Likes",       ""),
    (k4, "blue",   "👁️",  f"{kpi['avg_views']:,.0f}",  "Avg Views",       ""),
    (k5, "red",    "🔥", f"{kpi['avg_virality']:.3f}%","Avg Virality",    ""),
    (k6, "purple", "📤", f"{kpi['total_views']/1e6:.1f}M","Total Views",  ""),
]
for col, color, icon, val, label, sub in kpi_defs:
    with col:
        st.markdown(f"""
        <div class="kpi-card {color}">
          <div class="kpi-icon">{icon}</div>
          <div class="kpi-value">{val}</div>
          <div class="kpi-label">{label}</div>
          {"<div class='kpi-sub'>"+sub+"</div>" if sub else ""}
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ────────────────────────────────────────────────────────────────────────────
#  TABS
# ────────────────────────────────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
    "📊 Overview",
    "🔬 Deep Analysis",
    "🌍 Demographics",
    "🤖 ML Prediction",
    "👥 Persona Clusters",
    "🔴 Live Feed",
])

# ═══════════════════════════════════════════════════════════════════════════
#  TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">📌 Engagement Overview</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(bar_eng_by_age(df), use_container_width=True)
    with c2: st.plotly_chart(donut_age(df),       use_container_width=True)

    st.markdown('<div class="section-title">📡 Platform Intelligence</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3: st.plotly_chart(heatmap_platform_age(df),    use_container_width=True)
    with c4: st.plotly_chart(stacked_platform_usage(df),  use_container_width=True)

    st.markdown('<div class="section-title">🎬 Content Performance</div>', unsafe_allow_html=True)
    c5, c6 = st.columns([2, 1])
    with c5: st.plotly_chart(grouped_content_age(df),  use_container_width=True)
    with c6: st.plotly_chart(bar_content_eng(df),      use_container_width=True)

    st.markdown('<div class="section-title">📅 Trend Over Time</div>', unsafe_allow_html=True)
    st.plotly_chart(area_monthly(df), use_container_width=True)

    st.markdown('<div class="section-title">🫧 Age Group Performance Matrix</div>', unsafe_allow_html=True)
    st.plotly_chart(bubble_age_matrix(df), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 2 — DEEP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">📦 Distributions</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(box_eng_age(df),       use_container_width=True)
    with c2: st.plotly_chart(violin_views_age(df),  use_container_width=True)

    st.markdown('<div class="section-title">🕸️ Platform Radar & Scatter</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3: st.plotly_chart(radar_platform_age(df),        use_container_width=True)
    with c4: st.plotly_chart(scatter_likes_comments(df),    use_container_width=True)

    st.markdown('<div class="section-title">⏰ Temporal Patterns</div>', unsafe_allow_html=True)
    st.plotly_chart(heatmap_hourly(df), use_container_width=True)

    st.markdown('<div class="section-title">🔢 Correlation Matrix</div>', unsafe_allow_html=True)
    st.plotly_chart(heatmap_correlation(df), use_container_width=True)

    st.markdown('<div class="section-title">🔻 Interaction Funnel by Age Group</div>', unsafe_allow_html=True)
    st.plotly_chart(funnel_interactions(df), use_container_width=True)

    st.markdown('<div class="section-title">🏆 Top 20 High-Engagement Posts</div>', unsafe_allow_html=True)
    st.dataframe(top_posts(df, 20), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">📊 Summary Statistics by Age Group</div>', unsafe_allow_html=True)
    st.dataframe(agg_by_age(df), use_container_width=True, hide_index=True)

    top_csv = top_posts(df, 20).to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Top Posts", top_csv, "top_posts.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 3 — DEMOGRAPHICS
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">♀️♂️ Gender Analysis</div>', unsafe_allow_html=True)
    st.plotly_chart(heatmap_gender_age(df), use_container_width=True)

    st.markdown('<div class="section-title">🌍 Regional Breakdown</div>', unsafe_allow_html=True)
    st.plotly_chart(bar_region_age(df), use_container_width=True)

    # Gender breakdown table
    st.markdown('<div class="section-title">📋 Gender × Age Engagement Table</div>', unsafe_allow_html=True)
    g_table = df.groupby(["Age_Group","Gender"])["Engagement_Rate"].mean()\
                .unstack().reindex(AGE_GROUPS).round(3)
    st.dataframe(g_table, use_container_width=True)

    # Region table
    st.markdown('<div class="section-title">📋 Region × Age Engagement Table</div>', unsafe_allow_html=True)
    r_table = df.groupby(["Region","Age_Group"])["Engagement_Rate"].mean()\
                .unstack().round(3)
    st.dataframe(r_table, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 4 — ML PREDICTION
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">🤖 Predict Engagement Rate — Random Forest</div>',
                unsafe_allow_html=True)
    st.markdown("Enter post details below and our **Random Forest model** predicts the expected engagement rate.")

    col_form, col_chart = st.columns([1, 1])
    with col_form:
        st.markdown("#### 📝 Post Configuration")
        p_plat   = st.selectbox("Platform",       PLATFORMS)
        p_age    = st.selectbox("Target Age Group", AGE_GROUPS)
        p_cont   = st.selectbox("Content Type",   CONTENT)
        p_gender = st.selectbox("Target Gender",  GENDERS)
        p_region = st.selectbox("Region",         REGIONS)
        p_day    = st.selectbox("Day of Week",    DAYS)
        p_qtr    = st.selectbox("Quarter",        ["Q1","Q2","Q3","Q4"])
        p_hour   = st.slider("Posting Hour (0–23)", 0, 23, 12)
        p_foll   = st.number_input("Follower Count", 100, 1_000_000, 15_000, step=1000)
        predict_btn = st.button("🚀 Predict Engagement Rate", use_container_width=True)

    with col_chart:
        st.markdown("#### 🌲 Feature Importance")
        st.plotly_chart(bar_feature_importance(st.session_state.predictor.feature_importance()),
                        use_container_width=True)
        m = st.session_state.pred_metrics
        mc1, mc2 = st.columns(2)
        mc1.metric("R² Score", m["R2"])
        mc2.metric("CV R²",    m["CV_R2"])
        mc3, mc4 = st.columns(2)
        mc3.metric("RMSE",     m["RMSE"])
        mc4.metric("Accuracy", f"{m['Accuracy']}%")

    if predict_btn:
        pred = st.session_state.predictor.predict(
            p_plat, p_age, p_cont, p_gender, p_region, p_day, p_hour, p_foll, p_qtr
        )
        rating, color, badge = st.session_state.predictor.rating(pred)
        st.markdown(f"""
        <div class="pred-box">
          <div class="pred-label">Predicted Engagement Rate</div>
          <div class="pred-rate">{pred}%</div>
          <div class="pred-badge">{rating} · {badge}</div>
          <div class="pred-meta">{p_plat} · {p_age} · {p_cont} · {p_day} {p_hour:02d}:00 · {p_region}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 💡 Smart Optimisation Tips")
        t1, t2 = st.columns(2)
        best_cont = df[df["Age_Group"]==p_age].groupby("Content_Type")["Engagement_Rate"].mean().idxmax()
        best_plat = df[df["Age_Group"]==p_age].groupby("Platform")["Engagement_Rate"].mean().idxmax()
        best_hour = int(df[df["Age_Group"]==p_age].groupby("Hour")["Engagement_Rate"].mean().idxmax())
        best_day  = df[df["Age_Group"]==p_age].groupby("DayOfWeek")["Engagement_Rate"].mean().idxmax()
        best_reg  = df[df["Age_Group"]==p_age].groupby("Region")["Engagement_Rate"].mean().idxmax()
        with t1:
            st.markdown(f'<div class="tip-box">🎬 Best content for <b>{p_age}</b>: <b>{best_cont}</b></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="tip-box">📡 Best platform for <b>{p_age}</b>: <b>{best_plat}</b></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="tip-box">🌍 Top region for <b>{p_age}</b>: <b>{best_reg}</b></div>', unsafe_allow_html=True)
        with t2:
            st.markdown(f'<div class="tip-box">⏰ Peak hour for <b>{p_age}</b>: <b>{best_hour:02d}:00</b></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="tip-box">📅 Best day for <b>{p_age}</b>: <b>{best_day}</b></div>', unsafe_allow_html=True)
            hour_status = "✅ Optimal timing!" if p_hour == best_hour else f"⚠️ Try {best_hour:02d}:00 instead"
            st.markdown(f'<div class="tip-box">Your hour vs peak: <b>{hour_status}</b></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 5 — PERSONA CLUSTERS
# ═══════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">👥 K-Means Persona Clustering</div>', unsafe_allow_html=True)
    st.markdown("Users automatically clustered into **5 behavioural personas** using K-Means on engagement features.")

    sil = st.session_state.clusterer.sil_score
    if sil:
        st.info(f"🎯 Silhouette Score: **{sil}** — measures cluster quality (closer to 1.0 = better)")

    summary = st.session_state.clusterer.summary(df_c)
    pcolors = st.session_state.clusterer.persona_colors()

    # Persona cards
    cols = st.columns(5)
    for i, row in summary.iterrows():
        clr = pcolors.get(row["Persona"], "#7C3AED")
        with cols[i % 5]:
            st.markdown(f"""
            <div class="persona-card" style="border-top:3px solid {clr};">
              <div class="persona-name" style="color:{clr};">{row['Persona']}</div>
              <div class="persona-count">{int(row['Count']):,} users</div>
              <div class="persona-stat">
                ❤️ {row['Avg_Likes']:.0f} avg likes<br>
                💬 {row['Avg_Comments']:.0f} comments<br>
                📤 {row['Avg_Shares']:.0f} shares<br>
                📈 {row['Avg_Eng_Rate']:.2f}% eng rate<br>
                🔥 {row['Avg_Virality']:.3f}% virality
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(scatter_clusters_pca(df_c), use_container_width=True)
    with c2:
        from utils.charts import _fmt, go, PALETTE, _hex_rgba, AGE_GROUPS as AG
        import numpy as _np
        dist = st.session_state.clusterer.age_distribution(df_c)
        pivot = dist.pivot(index="Age_Group", columns="Persona", values="Count").fillna(0)
        pivot = pivot.reindex([a for a in AG if a in pivot.index])
        fig_h = go.Figure(go.Heatmap(
            z=pivot.values, x=[c.split(" ",1)[-1] for c in pivot.columns],
            y=pivot.index.tolist(),
            colorscale=[[0,"#EDE9FE"],[0.5,"#A78BFA"],[1,"#7C3AED"]],
            text=pivot.values.astype(int), texttemplate="%{text}",
            textfont=dict(size=11),
            hovertemplate="<b>%{y} — %{x}</b><br>Count: %{z}<extra></extra>",
            colorbar=dict(title="Count"),
        ))
        _fmt(fig_h, "🗺️ Persona Distribution across Age Groups","Persona","Age Group")
        st.plotly_chart(fig_h, use_container_width=True)

    st.markdown('<div class="section-title">📊 Full Persona Summary Table</div>', unsafe_allow_html=True)
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">🔎 Age × Persona Pivot</div>', unsafe_allow_html=True)
    dist2 = st.session_state.clusterer.age_distribution(df_c)
    pivot2 = dist2.pivot(index="Age_Group", columns="Persona", values="Count").fillna(0).astype(int)
    pivot2 = pivot2.reindex([a for a in AGE_GROUPS if a in pivot2.index])
    st.dataframe(pivot2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 6 — LIVE FEED
# ═══════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-title">🔴 Real-Time Engagement Simulation</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="live-banner">
      🟢 Live Simulation Mode — New posts ingested every update cycle
    </div>
    """, unsafe_allow_html=True)

    ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 2])
    with ctrl1: batch_sz = st.slider("Batch Size",    10, 150, 50, step=10)
    with ctrl2: delay    = st.slider("Delay (sec)",   0.3, 3.0, 0.8, step=0.1)
    with ctrl3: updates  = st.slider("No. of Updates", 3, 20, 8)

    b1, b2, _ = st.columns([1, 1, 4])
    with b1: start_btn = st.button("▶️ Start Live Feed", use_container_width=True)
    with b2: reset_btn = st.button("🔁 Reset",           use_container_width=True)

    if reset_btn:
        st.session_state.live_df = st.session_state.df.copy()
        st.session_state.rt_tick = 0
        st.rerun()

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("📦 Total Records", f"{len(st.session_state.live_df):,}")
    m2.metric("🔄 Updates",       st.session_state.rt_tick)
    m3.metric("📈 Live Avg Eng",  f"{st.session_state.live_df['Engagement_Rate'].mean():.3f}%")
    m4.metric("❤️ Live Avg Likes",f"{st.session_state.live_df['Likes'].mean():,.0f}")

    chart_slot = st.empty()
    chart_slot.plotly_chart(
        realtime_bar(st.session_state.live_df, st.session_state.rt_tick),
        use_container_width=True,
    )

    if start_btn:
        for _ in range(updates):
            batch = simulate_new_batch(n=batch_sz)
            st.session_state.live_df = pd.concat(
                [st.session_state.live_df, batch], ignore_index=True
            )
            st.session_state.rt_tick += 1
            chart_slot.plotly_chart(
                realtime_bar(st.session_state.live_df, st.session_state.rt_tick),
                use_container_width=True,
            )
            time.sleep(delay)

    st.markdown('<div class="section-title">📋 Latest Incoming Posts</div>', unsafe_allow_html=True)
    recent = st.session_state.live_df.tail(25)[
        ["Platform","Age_Group","Content_Type","Region","Gender",
         "Likes","Comments","Shares","Views","Engagement_Rate","Virality_Score"]
    ].round(3).iloc[::-1]
    st.dataframe(recent, use_container_width=True, hide_index=True)

    live_csv = st.session_state.live_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Live Dataset", live_csv, "live_data.csv", "text/csv")

# ────────────────────────────────────────────────────────────────────────────
#  FOOTER
# ────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small style='color:#9CA3AF;font-family:DM Sans,sans-serif;'>"
    "📊 Social Media Engagement Analyser · Python · Streamlit · Plotly · "
    "Pandas · Scikit-learn · NumPy</small></center>",
    unsafe_allow_html=True,
)
