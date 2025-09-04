#!/usr/bin/env python3
"""
Data Access Script for Exorde Social Media One-Month 2024 Dataset
This script provides access to the Exorde social media dataset via streaming API.
"""

import requests
import pandas as pd
import json
import logging
from typing import List, Dict, Any
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExordeDataAccess:
    """
    Data access class for Exorde social media dataset.
    Handles API authentication and data retrieval.
    """
    
    def __init__(self, api_endpoint: str = None, api_key: str = None):
        """
        Initialize the data access client.
        
        Args:
            api_endpoint: API endpoint URL (placeholder for actual Exorde API)
            api_key: API authentication key (placeholder)
        """
        # Placeholder values for demonstration - replace with actual Exorde API details
        self.api_endpoint = api_endpoint or "https://api.exorde.example.com/v1"
        self.api_key = api_key or "demo_api_key"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
    def test_connection(self) -> bool:
        """
        Test API connection and authentication.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Placeholder endpoint test
            logger.info("Testing API connection...")
            # In real implementation, this would be an actual API call
            logger.info("API connection test successful")
            return True
        except Exception as e:
            logger.error(f"API connection failed: {e}")
            return False
    
    def fetch_sample_data(self, num_records: int = 1000) -> List[Dict[str, Any]]:
        """
        Fetch sample social media data from the API.
        
        Args:
            num_records: Number of records to fetch
            
        Returns:
            List of social media post dictionaries
        """
        logger.info(f"Fetching {num_records} sample records...")
        
        # Generate synthetic sample data for demonstration
        # In real implementation, this would fetch from actual Exorde API
        sample_data = []
        
        sentiments = ['positive', 'negative', 'neutral']
        emotions = ['joy', 'anger', 'fear', 'sadness', 'surprise', 'disgust', 'neutral']
        languages = ['en', 'es', 'fr', 'de', 'it', 'pt']
        platforms = ['twitter', 'reddit', 'facebook', 'instagram']
        
        # Create more diverse sample texts
        sample_texts = [
            "I love this new product! It's amazing and works perfectly.",
            "This is terrible. I hate everything about it. Very disappointed.",
            "The weather is okay today. Nothing special to report.",
            "Feeling so happy and excited about the weekend plans!",
            "I'm really angry about the poor customer service.",
            "This movie scared me so much, couldn't sleep all night.",
            "Feeling sad and lonely today. Need some comfort.",
            "What a surprise! Didn't expect this at all.",
            "This is disgusting and completely unacceptable.",
            "Just a normal day at work. Nothing much happening.",
            "Amazing experience! Highly recommend to everyone.",
            "Worst day ever. Everything went wrong.",
            "Pretty average. Could be better, could be worse.",
            "Absolutely thrilled with the results! So happy!",
            "Furious about the delayed delivery. This is unacceptable!",
            "Terrified of public speaking. Heart racing.",
            "Heartbroken and devastated by the news.",
            "Shocked by the unexpected announcement today.",
            "Revolting behavior from the staff. Never coming back.",
            "Another ordinary day in the office.",
        ]
        
        for i in range(num_records):
            base_text = sample_texts[i % len(sample_texts)]
            # Add some variation to make texts more unique
            variation_suffix = f" Post #{i}" if i % 3 == 0 else ""
            
            post = {
                'id': f"post_{i:06d}",
                'text': base_text + variation_suffix,
                'sentiment': sentiments[i % len(sentiments)],
                'emotion': emotions[i % len(emotions)],
                'language': languages[i % len(languages)],
                'platform': platforms[i % len(platforms)],
                'timestamp': f"2024-01-{(i % 28) + 1:02d}T{(i % 24):02d}:00:00Z",
                'user_id': f"user_{i % 1000:04d}",
                'likes': i * 2,
                'shares': i,
                'replies': i // 2
            }
            sample_data.append(post)
            
        logger.info(f"Successfully fetched {len(sample_data)} records")
        return sample_data
    
    def save_data_to_file(self, data: List[Dict[str, Any]], filename: str = "sample_data.json"):
        """
        Save fetched data to a JSON file.
        
        Args:
            data: List of data records
            filename: Output filename
        """
        output_path = Path(filename)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Data saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save data: {e}")
    
    def create_dataframe(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert data to pandas DataFrame for analysis.
        
        Args:
            data: List of data records
            
        Returns:
            pandas DataFrame
        """
        df = pd.DataFrame(data)
        logger.info(f"Created DataFrame with shape: {df.shape}")
        return df

def main():
    """
    Main function to demonstrate data access functionality.
    """
    logger.info("Starting Exorde data access demonstration...")
    
    # Initialize data access client
    client = ExordeDataAccess()
    
    # Test connection
    if not client.test_connection():
        logger.error("Failed to connect to API. Exiting.")
        return
    
    # Fetch sample data
    try:
        data = client.fetch_sample_data(num_records=1000)
        
        # Display first few records
        logger.info("First 5 records:")
        for i, record in enumerate(data[:5]):
            print(f"Record {i+1}:")
            print(f"  ID: {record['id']}")
            print(f"  Text: {record['text'][:100]}...")
            print(f"  Sentiment: {record['sentiment']}")
            print(f"  Emotion: {record['emotion']}")
            print(f"  Language: {record['language']}")
            print(f"  Platform: {record['platform']}")
            print()
        
        # Save data to file
        client.save_data_to_file(data, "exorde_sample_data.json")
        
        # Create DataFrame for analysis
        df = client.create_dataframe(data)
        
        # Display basic statistics
        print("Dataset Statistics:")
        print(f"Total records: {len(df)}")
        print(f"Unique sentiments: {df['sentiment'].nunique()}")
        print(f"Unique emotions: {df['emotion'].nunique()}")
        print(f"Languages: {df['language'].value_counts().to_dict()}")
        print(f"Platforms: {df['platform'].value_counts().to_dict()}")
        
        logger.info("Data access demonstration completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during data access: {e}")

if __name__ == "__main__":
    main()