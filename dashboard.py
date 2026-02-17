import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import joblib
import shap
from lime.lime_tabular import LimeTabularExplainer

st.set_page_config(page_title="Stock & News Sentiment Dashboard", layout="wide")

# -----------------------------------------------
# Load data and models
# -----------------------------------------------
DATA_PATH = "data/all_stocks_features.csv"
MODEL_PATH = "models/best_xgb_model.pkl"
FEATURES_PATH = "models/features_lagged.pkl"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=['Date'])
    return df

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, features

all_stocks = load_data()
model, features_lagged = load_model()

# -----------------------------------------------
# Sidebar
# -----------------------------------------------
st.sidebar.header("Dashboard Controls")
stocks_list = all_stocks['Stock'].unique().tolist()
selected_stock = st.sidebar.selectbox("Select Stock", stocks_list)

# Filter stock-specific data first
stock_data = all_stocks[all_stocks['Stock'] == selected_stock].copy()

# Ensure the stock has at least some valid data
if stock_data.empty:
    st.warning(f"No data available for {selected_stock}.")
    st.stop()

# Sidebar date input based on this stock only
date_range = st.sidebar.date_input(
    "Select Date Range",
    [stock_data['Date'].min(), stock_data['Date'].max()],
    min_value=stock_data['Date'].min(),
    max_value=stock_data['Date'].max()
)

# Filter data for selected date range
df = stock_data[
    (stock_data['Date'] >= pd.to_datetime(date_range[0])) &
    (stock_data['Date'] <= pd.to_datetime(date_range[1]))
].copy()

# Remove rows with missing critical values
df.dropna(subset=features_lagged + ['Returns', 'Sentiment', 'Target'], inplace=True)

# Stop if no data after filtering
if df.empty:
    st.warning("No data available for the selected stock and date range.")
    st.stop()

# Features for SHAP/LIME
X_selected = df[features_lagged].copy()

# -----------------------------------------------
# Overview Metrics
# -----------------------------------------------
st.title(f"{selected_stock} — Stock & News Sentiment Dashboard")
st.subheader("Overview Metrics")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Average Return", f"{df['Returns'].mean():.4f}")
col2.metric("Average Volatility", f"{df['Volatility'].mean():.4f}")
col3.metric("Average Sentiment", f"{df['Sentiment'].mean():.4f}")
col4.metric("Sharpe Ratio", f"{(df['Returns']*df['Target']).mean()/((df['Returns']*df['Target']).std() + 1e-9) * np.sqrt(252):.2f}")

# -----------------------------------------------
# Time Series Plots
# -----------------------------------------------
st.subheader("Time Series: Returns vs Sentiment")
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df['Date'], y=df['Returns'], name='Daily Returns', line=dict(color='blue'),
    hovertemplate='Date: %{x}<br>Return: %{y:.4f}<extra></extra>'
))
fig.add_trace(go.Scatter(
    x=df['Date'], y=df['Sentiment'], name='Daily Sentiment', line=dict(color='orange'), yaxis='y2',
    hovertemplate='Date: %{x}<br>Sentiment: %{y:.4f}<extra></extra>'
))

fig.update_layout(
    title=f"{selected_stock}: Daily Returns vs Sentiment",
    xaxis_title='Date',
    yaxis=dict(title='Returns', side='left'),
    yaxis2=dict(title='Sentiment', overlaying='y', side='right'),
    legend=dict(x=0.01, y=0.99)
)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------
# Scatter Plot: Returns vs Sentiment
# -----------------------------------------------
st.subheader("Scatter Plot: Daily Returns vs Sentiment")
fig2 = px.scatter(
    df,
    x='Sentiment',
    y='Returns',
    color=df['Target'].map({0:'Down', 1:'Up'}),
    title=f"{selected_stock}: Returns vs Sentiment",
    trendline="ols",
    opacity=0.6,
    marginal_x='histogram',
    marginal_y='histogram',
    labels={'Sentiment':'Daily Sentiment','Returns':'Daily Returns'}
)
st.plotly_chart(fig2, use_container_width=True)

# -----------------------------------------------
# Rolling Correlation
# -----------------------------------------------
st.subheader("Rolling 14-Day Correlation: Returns & Sentiment")
df['rolling_corr'] = df['Returns'].rolling(14).corr(df['Sentiment'])
fig3 = px.line(
    df,
    x='Date',
    y='rolling_corr',
    title=f"{selected_stock}: 14-Day Rolling Correlation",
    labels={'rolling_corr':'Rolling 14-Day Corr'}
)
fig3.add_hline(y=0, line_dash="dash", line_color="black")
st.plotly_chart(fig3, use_container_width=True)

# -----------------------------------------------
# Distribution Plots
# -----------------------------------------------
st.subheader("Distribution: Returns & Sentiment")
fig4 = go.Figure()
fig4.add_trace(go.Histogram(x=df['Returns'], name="Returns", opacity=0.6, nbinsx=50))
fig4.add_trace(go.Histogram(x=df['Sentiment'], name="Sentiment", opacity=0.6, nbinsx=50))
fig4.update_layout(barmode='overlay',
                   title=f"{selected_stock}: Distribution of Returns & Sentiment",
                   xaxis_title='Value', yaxis_title='Count')
st.plotly_chart(fig4, use_container_width=True)

# -----------------------------------------------
# SHAP Explainability
# -----------------------------------------------
st.subheader("SHAP Feature Importance (XGBoost)")
st.write("Shows the contribution of features to model predictions.")

@st.cache_resource
def compute_shap(_model, _X):
    explainer = shap.TreeExplainer(_model)
    shap_values = explainer.shap_values(_X)
    return explainer, shap_values

explainer, shap_values = compute_shap(model, X_selected)

shap.summary_plot(shap_values, X_selected, feature_names=X_selected.columns, plot_type="bar", show=False)
st.pyplot(plt.gcf())
plt.clf()

# -----------------------------------------------
# LIME Explainability
# -----------------------------------------------
st.subheader("LIME Explanation (First Instance)")

if len(X_selected) > 0:
    explainer_lime = LimeTabularExplainer(
        training_data=X_selected.values,
        feature_names=X_selected.columns.tolist(),
        class_names=['Down', 'Up'],
        mode='classification'
    )

    exp = explainer_lime.explain_instance(X_selected.values[0], model.predict_proba, num_features=5)

    # Visualize LIME
    lime_data = exp.as_list()
    features_lime, weights = zip(*lime_data)
    colors = ['red' if w < 0 else 'green' for w in weights]

    fig5, ax = plt.subplots(figsize=(8, 5))
    ax.barh(features_lime, weights, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Contribution to Prediction")
    ax.set_title("LIME Explanation — First Instance")
    st.pyplot(fig5)
else:
    st.warning("Not enough data for LIME explanation.")

# -----------------------------------------------
# End
st.write("Dashboard complete. You can explore different stocks and date ranges.")
