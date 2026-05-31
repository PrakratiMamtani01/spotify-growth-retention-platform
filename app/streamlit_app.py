import pandas as pd
import numpy as np
import streamlit as st
import joblib
from pathlib import Path
import plotly.express as px


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Spotify Growth & Retention Intelligence",
    page_icon="🎧",
    layout="wide"
)


# -----------------------------
# Paths
# -----------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"


# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_churn_data():
    return pd.read_csv(DATA_DIR / "spotify_user_churn_clean.csv")


@st.cache_data
def load_segmented_users():
    return pd.read_csv(DATA_DIR / "spotify_users_segmented.csv")


@st.cache_data
def load_regional_features():
    df = pd.read_parquet(DATA_DIR / "weekly_regional_features.parquet")
    df["week"] = pd.to_datetime(df["week"])
    return df


@st.cache_data
def load_forecast_results():
    df = pd.read_csv(DATA_DIR / "regional_forecast_test_results.csv")
    df["week"] = pd.to_datetime(df["week"])
    return df


@st.cache_data
def load_breakout_recommendations():
    df = pd.read_csv(DATA_DIR / "latest_breakout_recommendations.csv")
    df["week"] = pd.to_datetime(df["week"])
    return df


# -----------------------------
# Load models
# -----------------------------
@st.cache_resource
def load_churn_model():
    return joblib.load(MODELS_DIR / "best_churn_model.pkl")


@st.cache_resource
def load_segmentation_model():
    return joblib.load(MODELS_DIR / "user_segmentation_model.pkl")


# -----------------------------
# Helper functions
# -----------------------------
def risk_label(probability):
    if probability >= 0.70:
        return "High Risk"
    elif probability >= 0.40:
        return "Medium Risk"
    else:
        return "Low Risk"


def retention_recommendation(probability, subscription_type, avg_daily_minutes, support_tickets, days_since_last_login):
    if probability >= 0.70:
        if support_tickets > 0:
            return "Prioritize support follow-up and issue resolution. This user shows high churn risk with platform friction."
        elif avg_daily_minutes < 60:
            return "Send a reactivation campaign with personalized playlists and new-release recommendations."
        elif days_since_last_login > 14:
            return "Send a win-back notification highlighting recent releases and saved playlist activity."
        else:
            return "Offer a targeted retention incentive or personalized content campaign."

    elif probability >= 0.40:
        if subscription_type == "Free":
            return "Encourage deeper engagement with personalized discovery and a Premium trial offer."
        else:
            return "Recommend personalized playlists and monitor engagement over the next week."

    else:
        return "User appears stable. Maintain engagement through regular personalization and new content discovery."


def segment_recommendation(segment_name):
    if segment_name == "Loyal High-Engagement Users":
        return "Use loyalty campaigns, early feature access, and premium engagement experiences."
    elif segment_name == "Low-Engagement At-Risk Users":
        return "Use reactivation campaigns, personalized playlists, and habit-building recommendations."
    elif segment_name == "High-Friction Engaged Users":
        return "Prioritize support resolution, customer service follow-up, and friction reduction."
    else:
        return "Monitor behavior and personalize engagement strategy."


# -----------------------------
# Load all required objects
# -----------------------------
churn_df = load_churn_data()
segmented_df = load_segmented_users()
regional_df = load_regional_features()
forecast_df = load_forecast_results()
breakout_df = load_breakout_recommendations()

churn_model = load_churn_model()
segmentation_artifacts = load_segmentation_model()


# -----------------------------
# Sidebar navigation
# -----------------------------
st.sidebar.title("Spotify Intelligence Platform")

page = st.sidebar.radio(
    "Choose a page",
    [
        "Project Overview",
        "Churn Prediction",
        "User Segmentation",
        "Regional Growth Forecasting",
        "Breakout Track Recommendations"
    ]
)


# -----------------------------
# Project Overview
# -----------------------------
if page == "Project Overview":
    st.title("🎧 Spotify Growth & Retention Intelligence Platform")

    st.markdown(
        """
        This project combines regional Spotify chart data and user-level churn data to support
        growth, retention, and content promotion strategy.

        The platform includes four main components:

        1. **Churn Prediction**: predicts whether a user is likely to leave Spotify.
        2. **User Segmentation**: groups users into behavior-based retention segments.
        3. **Regional Growth Forecasting**: forecasts next-week regional streaming demand.
        4. **Breakout Track Detection**: identifies Viral 50 tracks likely to enter Top 200.
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Churn Users Dataset", f"{len(churn_df):,} users")
    col2.metric("Regional Feature Rows", f"{len(regional_df):,}")
    col3.metric("Regions", f"{regional_df['region'].nunique():,}")
    col4.metric("Breakout Candidates", f"{len(breakout_df):,}")

    st.subheader("Project Data Sources")

    st.markdown(
        """
        - **Spotify chart dataset**: regional Top 200 and Viral 50 chart records.
        - **Spotify churn dataset**: user-level subscription, engagement, and churn behavior.
        """
    )

    st.subheader("Business Goal")

    st.markdown(
        """
        The goal is to identify growth opportunities by understanding which regions are growing,
        which songs are breaking out, which users are likely to churn, and which retention actions
        may be most appropriate for different user groups.
        """
    )


# -----------------------------
# Churn Prediction Page
# -----------------------------
elif page == "Churn Prediction":
    st.title("📉 Churn Prediction")

    st.markdown(
        """
        Enter a user's behavior and profile details to estimate churn probability.
        The model uses subscription type, listening activity, playlists, skips,
        support tickets, inactivity, country, and top genre.
        """
    )

    countries = sorted(churn_df["country"].unique())
    genres = sorted(churn_df["top_genre"].unique())
    subscription_types = sorted(churn_df["subscription_type"].unique())

    col1, col2 = st.columns(2)

    with col1:
        subscription_type = st.selectbox("Subscription Type", subscription_types)
        country = st.selectbox("Country", countries)
        top_genre = st.selectbox("Top Genre", genres)
        avg_daily_minutes = st.slider("Average Daily Minutes", 0.0, 250.0, 90.0, 1.0)

    with col2:
        number_of_playlists = st.slider("Number of Playlists", 0, 10, 3)
        skips_per_day = st.slider("Skips Per Day", 0, 20, 5)
        support_tickets = st.slider("Support Tickets", 0, 5, 0)
        days_since_last_login = st.slider("Days Since Last Login", 0, 60, 7)

    input_df = pd.DataFrame(
        {
            "subscription_type": [subscription_type],
            "country": [country],
            "avg_daily_minutes": [avg_daily_minutes],
            "number_of_playlists": [number_of_playlists],
            "top_genre": [top_genre],
            "skips_per_day": [skips_per_day],
            "support_tickets": [support_tickets],
            "days_since_last_login": [days_since_last_login],
        }
    )

    if st.button("Predict Churn Risk"):
        churn_probability = churn_model.predict_proba(input_df)[0, 1]
        label = risk_label(churn_probability)
        recommendation = retention_recommendation(
            churn_probability,
            subscription_type,
            avg_daily_minutes,
            support_tickets,
            days_since_last_login
        )

        col1, col2 = st.columns(2)

        col1.metric("Churn Probability", f"{churn_probability:.1%}")
        col2.metric("Risk Level", label)

        st.subheader("Recommended Retention Action")
        st.write(recommendation)

    st.subheader("Churn Dataset Summary")

    churn_rate = churn_df["churned"].mean()
    st.metric("Overall Churn Rate", f"{churn_rate:.1%}")

    subscription_churn = (
        churn_df.groupby("subscription_type")["churned"]
        .mean()
        .reset_index()
        .rename(columns={"churned": "churn_rate"})
    )

    fig = px.bar(
        subscription_churn,
        x="subscription_type",
        y="churn_rate",
        title="Churn Rate by Subscription Type",
        labels={"subscription_type": "Subscription Type", "churn_rate": "Churn Rate"}
    )
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# User Segmentation Page
# -----------------------------
elif page == "User Segmentation":
    st.title("👥 User Segmentation")

    st.markdown(
        """
        Users were clustered into behavior-based segments using listening activity,
        playlists, support tickets, skips, inactivity, subscription type, country, and genre.
        The churn label was not used during clustering.
        """
    )

    segment_profile = (
        segmented_df.groupby("segment_name")
        .agg(
            user_count=("user_id", "count"),
            churn_rate=("churned", "mean"),
            avg_daily_minutes=("avg_daily_minutes", "mean"),
            number_of_playlists=("number_of_playlists", "mean"),
            support_tickets=("support_tickets", "mean"),
            days_since_last_login=("days_since_last_login", "mean")
        )
        .reset_index()
    )

    st.subheader("Segment Profiles")
    st.dataframe(segment_profile, use_container_width=True)

    fig = px.bar(
        segment_profile,
        x="segment_name",
        y="churn_rate",
        title="Churn Rate by User Segment",
        labels={"segment_name": "User Segment", "churn_rate": "Churn Rate"}
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Segment Strategy")

    selected_segment = st.selectbox(
        "Choose a segment",
        sorted(segmented_df["segment_name"].unique())
    )

    st.write(segment_recommendation(selected_segment))

    selected_profile = segment_profile[segment_profile["segment_name"] == selected_segment]
    st.dataframe(selected_profile, use_container_width=True)


# -----------------------------
# Regional Growth Forecasting Page
# -----------------------------
elif page == "Regional Growth Forecasting":
    st.title("🌍 Regional Growth Forecasting")

    st.markdown(
        """
        This section shows regional streaming trends and next-week forecast results.
        The forecasting model was benchmarked against a simple persistence baseline.
        """
    )

    available_regions = sorted(
        [region for region in regional_df["region"].unique() if region != "Global"]
    )

    selected_region = st.selectbox("Choose a region", available_regions)

    region_history = regional_df[regional_df["region"] == selected_region].copy()
    region_forecast = forecast_df[forecast_df["region"] == selected_region].copy()

    st.subheader(f"Weekly Total Streams: {selected_region}")

    fig = px.line(
        region_history,
        x="week",
        y="total_streams",
        title=f"Weekly Top 200 Streams in {selected_region}",
        labels={"week": "Week", "total_streams": "Total Streams"}
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast Test Results")

    if len(region_forecast) > 0:
        fig2 = px.line(
            region_forecast,
            x="week",
            y=["next_week_streams", "rf_prediction", "baseline_prediction"],
            title=f"Actual vs Predicted Next-Week Streams: {selected_region}",
            labels={"week": "Week", "value": "Streams", "variable": "Series"}
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(
            region_forecast[
                [
                    "region",
                    "week",
                    "total_streams",
                    "next_week_streams",
                    "baseline_prediction",
                    "rf_prediction",
                    "rf_absolute_error"
                ]
            ].sort_values("week", ascending=False),
            use_container_width=True
        )
    else:
        st.info("No forecast test results available for this region.")

    st.subheader("Top Regions by Total Streams")

    region_summary = (
        regional_df[regional_df["region"] != "Global"]
        .groupby("region", as_index=False)
        .agg(total_streams=("total_streams", "sum"))
        .sort_values("total_streams", ascending=False)
        .head(15)
    )

    fig3 = px.bar(
        region_summary,
        x="region",
        y="total_streams",
        title="Top 15 Regions by Total Streams",
        labels={"region": "Region", "total_streams": "Total Streams"}
    )
    st.plotly_chart(fig3, use_container_width=True)


# -----------------------------
# Breakout Recommendations Page
# -----------------------------
elif page == "Breakout Track Recommendations":
    st.title("🚀 Breakout Track Recommendations")

    st.markdown(
        """
        This section ranks Viral 50 tracks by their predicted probability of entering
        the same region's Top 200 chart within the next 14 days.
        """
    )

    breakout_regions = sorted(breakout_df["region"].unique())
    selected_region = st.selectbox("Choose a region", breakout_regions)

    region_recs = breakout_df[breakout_df["region"] == selected_region].copy()

    top_n = st.slider("Number of recommendations", 5, 50, 20)

    st.subheader(f"Top Breakout Candidates: {selected_region}")

    display_cols = [
        "region",
        "week",
        "title",
        "artist",
        "best_viral_rank",
        "avg_viral_rank",
        "viral_days_on_chart",
        "breakout_probability"
    ]

    st.dataframe(
        region_recs[display_cols]
        .sort_values("breakout_probability", ascending=False)
        .head(top_n),
        use_container_width=True
    )

    fig = px.bar(
        region_recs.sort_values("breakout_probability", ascending=False).head(top_n),
        x="title",
        y="breakout_probability",
        color="artist",
        title=f"Top {top_n} Breakout Candidates in {selected_region}",
        labels={"title": "Track", "breakout_probability": "Breakout Probability"}
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        **Interpretation:** These recommendations should be treated as a ranking tool,
        not a perfect yes/no prediction system. The model is designed to surface promising
        regional Viral 50 tracks for further review or promotional testing.
        """
    )