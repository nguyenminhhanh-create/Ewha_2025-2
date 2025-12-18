import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import plotly.graph_objects as go
from datetime import datetime, timedelta
import openai
import os

# --- Page Config ---
st.set_page_config(layout="wide", page_title="YouTube Trends Korea 2026 Predictor")

# --- Constants ---
# The user's file is named daily_trending_videos.xlsb.csv
DATA_FILE = "daily_trending_videos.xlsb.csv"

# --- Helper Functions ---

@st.cache_data
def load_data():
    """
    Filters for Korea region.
    """
    if os.path.exists(DATA_FILE):
        try:
            df = pd.read_csv(DATA_FILE)
            
            # Column mapping based on inspection:
            # video_id,channel,country,views,likes,comments,published_at,fetch_date
            
            # Normalize column names to lowercase
            df.columns = [c.lower() for c in df.columns]
            
            # Filter for Korea
            if 'country' in df.columns:
                df = df[df['country'] == 'KR']
            
            # Rename columns to standard internal names
            # views -> view_count
            # comments -> comment_count
            # fetch_date -> trending_date
            
            rename_map = {
                'views': 'view_count',
                'comments': 'comment_count',
                'fetch_date': 'trending_date'
            }
            df = df.rename(columns=rename_map)
            
            # Ensure proper dates
            df['trending_date'] = pd.to_datetime(df['trending_date'], format='mixed', utc=True).dt.date
            
            # Group by date to get daily totals
            daily_df = df.groupby('trending_date')[['view_count', 'likes', 'comment_count']].sum().reset_index()
            return daily_d

    
    # Trend with some seasonality and noise
    t = np.arange(n)
    except
    
    # Views: Growing trend + random noise
    base_views = 1_000_000 + (t * 2000) + (np.sin(t / 30) * 100_000)
    view_counts = base_views + np.random.normal(0, 150_000, n)
    view_counts = np.maximum(view_counts, 0).astype(int)
    
    # Likes: Correlated with views
    likes = (view_counts * 0.04) + np.random.normal(0, 5000, n)
    likes = np.maximum(likes, 0).astype(int)
    
    # Comments: Correlated with views
    comments = (view_counts * 0.005) + np.random.normal(0, 1000, n)
    comments = np.maximum(comments, 0).astype(int)
    
    df = pd.DataFrame({
        'trending_date': dates.date,
        'view_count': view_counts,
        'likes': likes,
        'comment_count': comments,
    })
    return df

def run_regression(df, target_col):
    """
    Runs linear regression on the dataframe for a specific target column.
    Returns the model, metrics, and X/y used.
    """
    df = df.sort_values('trending_date')
    
    # Create ordinal date for regression
    df['date_ordinal'] = pd.to_datetime(df['trending_date']).map(datetime.toordinal)
    
    X = df[['date_ordinal']]
    y = df[target_col]
    
    model = LinearRegression()
    model.fit(X, y)
    
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    
    return model, r2, X, y

def predict_future(model, start_date, end_date):
    """Predicts values for a future date range."""
    future_dates = pd.date_range(start_date, end_date)
    future_dates_ordinal = future_dates.map(datetime.toordinal).values.reshape(-1, 1)
    
    predictions = model.predict(future_dates_ordinal)
    predictions = np.maximum(predictions, 0) # No negative values
    
    return future_dates, predictions

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    st.info("Please enter your OpenAI API key to enable the AI analysis assistant.")
    
    st.divider()
    st.subheader("About")
    st.markdown("""
    **YouTube Trending Predictor (Korea)**
    
    This app analyzes historical trending video statistics and predicts future performance for the first half of 2026.
    
    **Features:**
    - Time-series Regression
    - 6-month forecasting
    - AI-powered interpretation
    """)

# --- Main Page ---
st.title("🇰🇷 YouTube Trending Analysis & Prediction (Korea)")
st.write("Historical analysis of 2023-2025 and predictions for Jan-Jun 2026.")

# Load Data
df = load_data()

st.divider()

# --- Analysis & Visualization Section ---
col1, col2 = st.columns([2, 1])

# Run Regressions
targets = ['view_count', 'likes', 'comment_count']
models = {}
predictions_2026 = {}
future_dates_2026 = None

# Date range for prediction: Jan 1 2026 to Jun 30 2026
pred_start = datetime(2026, 1, 1)
pred_end = datetime(2026, 6, 30)

analysis_summary = "### Regression Analysis Summary\n\n"

with col1:
    st.subheader("Data Visualization & Forecast")
    
    tab1, tab2, tab3 = st.tabs(["Views", "Likes", "Comments"])
    
    tabs = {
        'view_count': tab1,
        'likes': tab2,
        'comment_count': tab3
    }
    
    for target in targets:
        # Analysis
        model, r2, X, y = run_regression(df, target)
        models[target] = model
        
        # Prediction
        f_dates, f_preds = predict_future(model, pred_start, pred_end)
        predictions_2026[target] = f_preds
        if future_dates_2026 is None:
            future_dates_2026 = f_dates

        # Construct Summary for AI
        latest_historical = y.iloc[-1]
        predicted_start = f_preds[0]
        predicted_end = f_preds[-1]
        trend_direction = "increasing" if predicted_end > predicted_start else "decreasing"
        
        analysis_summary += f"**{target.replace('_', ' ').title()}**:\n"
        analysis_summary += f"- R² Score: {r2:.3f} (Model Fit)\n"
        analysis_summary += f"- Last Observed ({df['trending_date'].iloc[-1]}): {latest_historical:,.0f}\n"
        analysis_summary += f"- Pred. Jan 1 2026: {predicted_start:,.0f}\n"
        analysis_summary += f"- Pred. Jun 30 2026: {predicted_end:,.0f}\n"
        analysis_summary += f"- Trend: {trend_direction}\n\n"

        # Visualization
        with tabs[target]:
            fig = go.Figure()
            
            # Historical Data
            fig.add_trace(go.Scatter(
                x=df['trending_date'], 
                y=y, 
                mode='lines', 
                name='Historical',
                line=dict(color='#FF0000') # YouTube Red
            ))
            
            # Regression Line (Historical) -> basically the model on X
            y_fit = model.predict(X)
            fig.add_trace(go.Scatter(
                x=df['trending_date'],
                y=y_fit,
                mode='lines',
                name='Trend (Fit)',
                line=dict(color='blue', dash='dash')
            ))
            
            # Future Prediction
            fig.add_trace(go.Scatter(
                x=f_dates,
                y=f_preds,
                mode='lines',
                name='Prediction (2026)',
                line=dict(color='green', width=3)
            ))
            
            fig.update_layout(
                title=f"Korea Trending {target.replace('_', ' ').title()} (2023 - 2026)",
                xaxis_title="Date",
                yaxis_title="Count",
                hovermode="x unified",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Metric Summary")
    for target in targets:
        avg_hist = df[target].mean()
        avg_pred = predictions_2026[target].mean()
        growth = ((avg_pred - avg_hist) / avg_hist) * 100
        
        st.metric(
            label=f"Avg {target.replace('_', ' ').title()} (2026 H1)",
            value=f"{avg_pred:,.0f}",
            delta=f"{growth:+.1f}% vs Hist. Avg"
        )
    
    st.markdown("---")
    st.write("### Data Stats")
    st.write(f"Start Date: {df['trending_date'].min()}")
    st.write(f"End Date: {df['trending_date'].max()}")
    st.write(f"Total Days: {len(df)}")


# --- Chat Interface ---
st.divider()
st.subheader("💬 AI Analysis Assistant")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! I've analyzed the YouTube trending data for Korea. Ask me anything about the predictions for 2026!"}]

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

prompt = st.chat_input("Ask about the trends (e.g., 'Will views increase in 2026?')")

if prompt:
    if not openai_api_key:
        st.error("Please enter your OpenAI API key in the sidebar to chat.")
    else:
        st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Prepare context for OpenAI
        system_instruction = f"""
        You are an expert data analyst specializing in Social Media Trends for Korea.
        You have access to the following regression analysis results for YouTube Trending videos in Korea:
        
        DATA CONTEXT:
        - Region: Korea (South)
        - Historical Data: {df['trending_date'].min()} to {df['trending_date'].max()}
        - Prediction Period: Jan 1, 2026 to Jun 30, 2026 (Next 6 months)
        
        ANALYSIS RESULTS:
        {analysis_summary}
        
        USER QUESTION: {prompt}
        
        INSTRUCTIONS:
        - Answer the user's question based strictly on the data provided above.
        - Be professional, concise, and insightful.
        - If the user asks about specific numbers, quote the predictions.
        - Mention that these are statistical projections based on linear regression.
        """
        
        client = openai.OpenAI(api_key=openai_api_key)
        
        try:
            stream = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ],
                stream=True
            )
            
            with st.chat_message("assistant"):
                response = st.write_stream(stream)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            
    





