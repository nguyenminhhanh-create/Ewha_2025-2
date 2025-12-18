import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import openai

# Page Configuration
st.set_page_config(page_title="Korea YouTube Trends 2026", layout="wide")

# Sidebar
st.sidebar.title="Settings"
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

st.title("🇰🇷 Korea YouTube Trends Analysis & Prediction (2026)")
st.markdown("Analysis of YouTube trending videos in Korea with predictions for the first half of 2026.")

# Data Loading Function
@st.cache_data
def load_data():
    file_path = "daily_trending_videos.csv"
    try:
        df = pd.read_csv(file_path)
        # Ensure dates are datetime
        df['trending_date'] = pd.to_datetime(df['trending_date'])
        
        # Filter for Korea
        # Check if country column exists. If not, warn user or assume data is already specific.
        if 'country' in df.columns:
            df_kr = df[df['country'] == 'KR'].copy()
        else:
            st.warning("No 'country' column found. Using all data.")
            df_kr = df.copy()
            
        return df_kr
    except FileNotFoundError:
        return None

data = load_data()

if data is None:
    st.error("Dataset 'daily_trending_videos.csv' not found. Please ensure the file is in the application directory.")
    st.info("You can run 'python create_mock_data.py' to generate a sample dataset for testing.")
    st.stop()

if data.empty:
    st.error("No data found for Korea (KR). Please check the dataset.")
    st.stop()

# Preprocessing for Regression
# We want to predict daily global stats (total views, likes, comments) or average? 
# Let's do Daily Average per video to normalize for number of trending videos captured.
daily_stats = data.groupby(data['trending_date'].dt.date).agg({
    'view_count': 'mean',
    'likes': 'mean',
    'comment_count': 'mean',
    'video_id': 'count' # Number of videos trending that day
}).reset_index()
daily_stats['trending_date'] = pd.to_datetime(daily_stats['trending_date'])
daily_stats['ordinal_date'] = daily_stats['trending_date'].map(datetime.toordinal)

# Analysis & Prediction Function
def predict_metric(df, metric_name, target_dates):
    X = df[['ordinal_date']]
    y = df[metric_name]
    
    model = LinearRegression()
    model.fit(X, y)
    
    target_ordinals = target_dates.map(datetime.toordinal).values.reshape(-1, 1)
    predictions = model.predict(target_ordinals)
    
    return predictions, model

# Future Dates (Jan 2026 - Jun 2026)
future_start = datetime(2026, 1, 1)
future_end = datetime(2026, 6, 30)
days_count = (future_end - future_start).days + 1
future_dates = [future_start + timedelta(days=x) for x in range(days_count)]
future_dates_pd = pd.to_datetime(future_dates)

# Perform Predictions
metrics = ['view_count', 'likes', 'comment_count']
predictions = {}

for metric in metrics:
    pred_vals, _ = predict_metric(daily_stats, metric, future_dates_pd)
    predictions[metric] = pred_vals

# Visualization Section
st.header("Trend Visualizations & Predictions")

tab1, tab2, tab3 = st.tabs(["Views", "Likes", "Comments"])

def plot_trend(metric, label):
    fig = go.Figure()
    
    # Historical
    fig.add_trace(go.Scatter(
        x=daily_stats['trending_date'], 
        y=daily_stats[metric],
        mode='lines',
        name=f'Historical {label}'
    ))
    
    # Prediction
    fig.add_trace(go.Scatter(
        x=future_dates_pd,
        y=predictions[metric],
        mode='lines',
        name=f'Predicted {label} (2026)',
        line=dict(dash='dash')
    ))
    
    fig.update_layout(title=f"Average Daily {label} Trend", xaxis_title="Date", yaxis_title=label)
    return fig

with tab1:
    st.plotly_chart(plot_trend('view_count', 'Views'), use_container_width=True)
with tab2:
    st.plotly_chart(plot_trend('likes', 'Likes'), use_container_width=True)
with tab3:
    st.plotly_chart(plot_trend('comment_count', 'Comments'), use_container_width=True)

# Chat Interface Section
st.header("AI Analyst Chat")
st.markdown("Ask questions about the trends and predictions.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about the 2026 predictions..."):
    if not api_key:
        st.error("Please enter your OpenAI API Key in the sidebar first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Prepare context for AI
        # Summarize recent history and predictions
        latest_hist = daily_stats.iloc[-1]
        summary_context = f"""
        Data Context: YouTube Trending Videos in Korea.
        Historical Data (Latest {latest_hist['trending_date'].date()}):
        - Avg Views: {latest_hist['view_count']:.0f}
        - Avg Likes: {latest_hist['likes']:.0f}
        - Avg Comments: {latest_hist['comment_count']:.0f}
        
        Predictions for 2026 (First 6 Months):
        - Jan 1, 2026 Predicted: Views ~{predictions['view_count'][0]:.0f}
        - Jun 30, 2026 Predicted: Views ~{predictions['view_count'][-1]:.0f}
        
        General Trend:
        Views are {'increasing' if predictions['view_count'][-1] > predictions['view_count'][0] else 'decreasing'}.
        """
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                client = openai.OpenAI(api_key=api_key)
                
                stream = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": f"You are a data analyst assistant. Interpret the following regression analysis results for Korean YouTube trends.\n{summary_context}"},
                        {"role": "user", "content": prompt}
                    ],
                    stream=True
                )
                
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

