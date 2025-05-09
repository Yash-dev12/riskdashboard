import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils import stratify_risk, filter_members, plot_risk_distribution
import os

st.set_page_config(page_title="Risk Stratification Dashboard", layout="wide", page_icon="📊")
sns.set_theme(style="whitegrid")

@st.cache_data
def load_data():
    return pd.read_csv("data/members.csv")

with st.spinner("🔍 Loading and analyzing data..."):
    df = load_data()
    df = stratify_risk(df)

with st.container():
    col1, col2 = st.columns([1, 4])
    # with col1:
    #     st.markdown("<h2 style='font-size:50px;'>💓</h2>", unsafe_allow_html=True)
    with col2:
        st.markdown("<h1 style='text-align: left; color: teal;'>🩺 Risk Stratification Dashboard</h1>", unsafe_allow_html=True)

st.markdown("---")

# Sidebar filters
st.sidebar.header("🎛 Filter Controls")
age_range = st.sidebar.slider("Age Range", int(df["Age"].min()), int(df["Age"].max()), (30, 60))
gender = st.sidebar.multiselect("Gender", df["Gender"].unique(), default=list(df["Gender"].unique()))
county = st.sidebar.multiselect("Home County", df["HOME_COUNTY"].unique(), default=list(df["HOME_COUNTY"].unique()))
cancer = st.sidebar.multiselect("Cancer Type", df["CANCER_TYPE"].unique(), default=list(df["CANCER_TYPE"].unique()))
new_member = st.sidebar.selectbox("New Member?", ["All", 0, 1])
risk_levels = st.sidebar.multiselect("Risk Category", ["Low", "Medium", "High"], default=["Low", "Medium", "High"])

filtered_df = df[
    (df["Age"].between(age_range[0], age_range[1])) &
    (df["Gender"].isin(gender)) &
    (df["HOME_COUNTY"].isin(county)) &
    (df["CANCER_TYPE"].isin(cancer)) &
    (df["Risk Category"].isin(risk_levels))
]
if new_member != "All":
    filtered_df = filtered_df[filtered_df["NEW_MEMBER"] == int(new_member)]

# Metrics
st.markdown("### 🔢 Key Metrics")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Members", len(filtered_df))
col2.metric("% High Risk", f"{(filtered_df['Risk Category'] == 'High').mean() * 100:.1f}%")
col3.metric("Avg Risk Score", round(filtered_df["risk_score"].mean(), 1))
col4.metric("Avg Age", round(filtered_df["Age"].mean(), 1))
col5.metric("Medically Fragile %", f"{filtered_df['DX_IND_MEDICALLY_FRAGILE'].mean() * 100:.1f}%")

st.markdown("### 📋 Member Data (Filtered)")
st.dataframe(filtered_df, use_container_width=True, height=300)

# Gauges
st.markdown("### 🎯 Gauge Indicators")
g1, g2, g3 = st.columns(3)
with g1:
    fig = go.Figure(go.Indicator(mode="gauge+number", value=filtered_df["BH_SPMI"].mean(), title={"text": "Avg BH_SPMI"}, gauge={"axis": {"range": [0, 5]}, "bar": {"color": "orange"}}))
    st.plotly_chart(fig, use_container_width=True)
with g2:
    fig = go.Figure(go.Indicator(mode="gauge+number", value=filtered_df["DX_CNT_SUBSTANCE_ABUSE"].mean(), title={"text": "Substance Abuse"}, gauge={"axis": {"range": [0, 5]}, "bar": {"color": "crimson"}}))
    st.plotly_chart(fig, use_container_width=True)
with g3:
    fig = go.Figure(go.Indicator(mode="gauge+number", value=filtered_df["DX_CNT_COMPLEX_TRAUMA"].mean(), title={"text": "Complex Trauma"}, gauge={"axis": {"range": [0, 5]}, "bar": {"color": "green"}}))
    st.plotly_chart(fig, use_container_width=True)

# Trend line
st.markdown("### 📈 Risk Score Trend Over Months")
trend_df = filtered_df.copy()
trend_df["Month"] = np.random.choice(["Jan", "Feb", "Mar", "Apr", "May", "Jun"], size=len(trend_df))
monthly_avg = trend_df.groupby("Month")["risk_score"].mean().reindex(["Jan", "Feb", "Mar", "Apr", "May", "Jun"])
fig, ax = plt.subplots(figsize=(8, 4))
monthly_avg.plot(kind="line", marker="o", color="teal", ax=ax)
ax.set_title("Average Risk Score Over Months")
st.pyplot(fig)

# Pie chart
st.markdown("### 🧩 Risk Category Breakdown")
fig = px.pie(filtered_df, names='Risk Category', title='Proportion of Risk Categories', hole=0.4)
st.plotly_chart(fig, use_container_width=True)

# Risk by County
st.markdown("### 🌍 Risk Stratification by County")
fig, ax = plt.subplots(figsize=(8, 4))
pd.crosstab(filtered_df["HOME_COUNTY"], filtered_df["Risk Category"]).plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
ax.set_title("Risk by County")
plt.xticks(rotation=45)
st.pyplot(fig)

# Risk by Gender
st.markdown("### 🧑‍🤝‍🧑 Gender vs Risk")
gender_risk = pd.crosstab(filtered_df['Gender'], filtered_df['Risk Category'])
fig, ax = plt.subplots(figsize=(6, 4))
gender_risk.plot(kind='bar', stacked=True, ax=ax, colormap='coolwarm')
ax.set_title("Risk Stratification by Gender")
st.pyplot(fig)

# Heatmap
st.markdown("### 🧠 Chronic Conditions Correlation")
chronic_cols = ["_ASTHMA_IND", "_ASCVD_IND", "_CHF_IND", "_CKD_IND", "_COPD_IND"]
heat_data = filtered_df[chronic_cols].corr()
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(heat_data, annot=True, cmap="YlGnBu", ax=ax)
st.pyplot(fig)

# Footer
st.markdown("<hr style='border-top: 1px solid lightgrey;' />", unsafe_allow_html=True)
# st.markdown("<h4 style='text-align: center; color: grey;'>Made with ❤️ by Yashaswini Guntupalli</h4>", unsafe_allow_html=True)
