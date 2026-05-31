# Spotify Growth & Retention Intelligence Platform

## Overview

This project is a data science portfolio project that combines Spotify regional chart data and user-level churn data to support growth, retention, and content strategy.

The platform includes:

- Churn prediction
- User segmentation
- Regional stream forecasting
- Breakout track recommendation

## Business Problem

Spotify needs to understand which users are likely to churn, which user groups require different retention strategies, which regions are growing, and which Viral 50 tracks may break into the Top 200.

## Datasets

This project uses two Kaggle-style datasets:

1. Spotify regional chart dataset containing Top 200 and Viral 50 chart records from 2017 to 2021.
2. Spotify user churn dataset containing subscription type, country, usage behavior, support tickets, inactivity, and churn labels.

Raw datasets are not included in the repository because of size. Processed files required for the deployed app are included.

## Project Components

### 1. Churn Prediction

A Random Forest classifier was selected as the final churn model. It predicted churn using subscription type, average daily minutes, playlists, skips, support tickets, inactivity, country, and top genre.

### 2. User Segmentation

KMeans clustering was used to create three user segments:

- Loyal High-Engagement Users
- Low-Engagement At-Risk Users
- High-Friction Engaged Users

### 3. Regional Forecasting

A Random Forest regressor was used to predict next-week regional Top 200 streams. The model was benchmarked against a persistence baseline using current-week streams.

### 4. Breakout Track Detection

A Logistic Regression model was used as a breakout ranking model to identify Viral 50 tracks likely to enter the same region's Top 200 chart within 14 days.

## App Features

The Streamlit app includes:

- Interactive churn risk prediction
- User segment profiles and retention strategies
- Regional stream trend visualization
- Forecast results by region
- Breakout track recommendations by region

## Tech Stack

- Python
- pandas
- scikit-learn
- Streamlit
- Plotly
- DuckDB
- Parquet
- Joblib

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py