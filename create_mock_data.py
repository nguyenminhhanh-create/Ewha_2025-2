import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def create_mock_dataset():
    """
    Creates a mock 'daily_trending_videos.csv' mimicking the structure of the Kaggle dataset.
    Includes data for multiple countries, specifically 'KR' (Korea).
    """
    print("Generating mock dataset...")
    
    # Date range: 2024-01-01 to today (late 2025 based on prompt context, 
    # but let's generate from 2024-01-01 to 2025-12-31 to have enough data)
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 12, 19) # Current date per system
    days = (end_date - start_date).days
    
    data = []
    
    # Generate random trends
    # specific trend for Korea: generally increasing views
    
    countries = ['US', 'KR', 'JP', 'IN']
    
    for day in range(days + 1):
        current_date = start_date + timedelta(days=day)
        date_str = current_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # specific daily trends
        # Make Korea trends vary with some seasonality/noise
        base_views = 500000 + (day * 1000) # Increasing trend
        
        # Generate 50 videos per day per country
        for country in countries:
            for i in range(20): # 20 videos per country per day
                video_id = f"vid_{country}_{day}_{i}"
                
                # Randomize views based on trend
                views = int(abs(np.random.normal(base_views, 200000)))
                if country == 'KR':
                    views = int(views * 1.2) # Korea has slightly more views in this mock
                
                likes = int(views * np.random.uniform(0.01, 0.05))
                comment_count = int(likes * np.random.uniform(0.05, 0.15))
                
                row = {
                    'video_id': video_id,
                    'title': f"Video Title {i}",
                    'publishedAt': (current_date - timedelta(days=random.randint(0, 5))).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    'channelId': f"ch_{random.randint(1000, 9999)}",
                    'channelTitle': f"Channel {random.randint(100, 999)}",
                    'categoryId': random.randint(1, 30),
                    'trending_date': date_str,
                    'view_count': views,
                    'likes': likes,
                    'comment_count': comment_count,
                    'country': country
                }
                data.append(row)
                
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_file = 'daily_trending_videos.csv'
    df.to_csv(output_file, index=False)
    print(f"Mock dataset created: {output_file}")
    print(f"Total rows: {len(df)}")
    print(f"Korea (KR) rows: {len(df[df['country'] == 'KR'])}")

if __name__ == "__main__":
    create_mock_dataset()
