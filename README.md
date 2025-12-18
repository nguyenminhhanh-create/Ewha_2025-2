# 🇰🇷 Korea YouTube Trends Analysis & Prediction (2026)

This project is a Streamlit application designed to analyze daily trending YouTube videos in Korea. It processes historical data to perform regression analysis and predicts viewing trends (Views, Likes, Comments) for the first half of 2026. Additionally, it features an OpenAI-powered AI Analyst Chatbot that allows users to interactively query the data and predictions.

## 🚀 Features

*   **Trend Analysis**: Aggregates daily trending statistics for Korea.
*   **Future Predictions**: Uses Linear Regression to predict trends for **January - June 2026**.
*   **Interactive Visualizations**: Dynamic line charts for Views, Likes, and Comments using Plotly.
*   **AI Chat Interface**:  Integrated with OpenAI's GPT models to answer natural language questions about the trends.
*   **Streaming Responses**: Real-time streaming of AI responses for a smooth user experience.

## 🛠️ Tech Stack

*   **Frontend/App Framework**: [Streamlit](https://streamlit.io/)
*   **Data Manipulation**: Pandas, NumPy
*   **Machine Learning**: Scikit-learn (Linear Regression)
*   **Visualization**: Plotly
*   **AI Integration**: OpenAI API

## 📂 Dataset

The application expects a dataset named `daily_trending_videos.csv`.
*   **Source**: [YouTube Trending Video Dataset (Kaggle)](https://www.kaggle.com/datasets/sebastianbesinski/youtube-trending-videos-2025-updated-daily?select=daily_trending_videos.csv)
*   **Note**: The app filters for entries where `country == 'KR'`.

## 📦 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/korea-youtube-trends.git
    cd korea-youtube-trends
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

### 1. Prepare Data
If you do not have the Kaggle dataset yet, you can generate a synthetic dataset for testing purposes:

```bash
python create_mock_data.py
```
*This will create a `daily_trending_videos.csv` file in your directory.*

### 2. Run the App
Execute the Streamlit application:

```bash
streamlit run app.py
```

### 3. Open in Browser
The app will open automatically in your default browser (usually at `http://localhost:8501`).

### 4. Configure AI
Enter your **OpenAI API Key** in the sidebar to enable the chat functionality.

## 📊 Project Structure

*   `app.py`: Main Streamlit application source code.
*   `create_mock_data.py`: Utility script to generate mock data for testing.
*   `requirements.txt`: Python dependencies.
*   `README.md`: Project documentation.

## 📝 License

This project is open source and available under the [MIT License](LICENSE).
