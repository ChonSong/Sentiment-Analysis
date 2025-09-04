#!/usr/bin/env python3
"""
Preprocessing Pipeline for Sentiment and Emotion Analysis
This script implements a comprehensive data preprocessing pipeline for social media text data.
"""

import pandas as pd
import numpy as np
import json
import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import unicodedata
import string

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """
    Comprehensive text preprocessing class for multilingual social media data.
    """
    
    def __init__(self, max_length: int = 512, min_length: int = 5):
        """
        Initialize the preprocessor.
        
        Args:
            max_length: Maximum sequence length for padding/truncation
            min_length: Minimum text length to keep
        """
        self.max_length = max_length
        self.min_length = min_length
        self.stopwords = self._load_multilingual_stopwords()
        
    def _load_multilingual_stopwords(self) -> Dict[str, set]:
        """
        Load stopwords for multiple languages.
        For demo purposes, using basic English stopwords.
        In production, would use NLTK or spaCy multilingual stopwords.
        """
        # Basic English stopwords for demonstration
        english_stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
            'had', 'what', 'said', 'each', 'which', 'do', 'how', 'their', 'if'
        }
        
        return {
            'en': english_stopwords,
            'es': {'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se'},
            'fr': {'le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir'},
            'de': {'der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich'},
            'it': {'il', 'di', 'che', 'e', 'la', 'per', 'in', 'un', 'è', 'con'},
            'pt': {'o', 'de', 'e', 'do', 'da', 'em', 'para', 'é', 'com', 'um'}
        }
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text data.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove user mentions and hashtags (but keep the text content)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Normalize Unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Basic text cleaning
        text = text.strip()
        
        return text
    
    def tokenize_text(self, text: str, language: str = 'en') -> List[str]:
        """
        Tokenize text based on language.
        
        Args:
            text: Text to tokenize
            language: Language code
            
        Returns:
            List of tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Simple word tokenization (in production, use language-specific tokenizers)
        # Remove punctuation and split on whitespace
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = text.split()
        
        return tokens
    
    def remove_stopwords(self, tokens: List[str], language: str = 'en') -> List[str]:
        """
        Remove stopwords based on language.
        
        Args:
            tokens: List of tokens
            language: Language code
            
        Returns:
            Filtered tokens
        """
        stopwords = self.stopwords.get(language, self.stopwords['en'])
        return [token for token in tokens if token not in stopwords]
    
    def preprocess_text(self, text: str, language: str = 'en', remove_stops: bool = True) -> List[str]:
        """
        Complete text preprocessing pipeline.
        
        Args:
            text: Raw text
            language: Language code
            remove_stops: Whether to remove stopwords
            
        Returns:
            Preprocessed tokens
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Check minimum length
        if len(cleaned_text) < self.min_length:
            return []
        
        # Tokenize
        tokens = self.tokenize_text(cleaned_text, language)
        
        # Remove stopwords if requested
        if remove_stops:
            tokens = self.remove_stopwords(tokens, language)
        
        return tokens

class DataPreprocessingPipeline:
    """
    Complete data preprocessing pipeline for the sentiment analysis project.
    """
    
    def __init__(self, max_sequence_length: int = 512):
        """
        Initialize the preprocessing pipeline.
        
        Args:
            max_sequence_length: Maximum sequence length for padding/truncation
        """
        self.max_sequence_length = max_sequence_length
        self.text_preprocessor = TextPreprocessor(max_length=max_sequence_length)
        self.sentiment_encoder = LabelEncoder()
        self.emotion_encoder = LabelEncoder()
        self.vocabulary = {}
        self.word_to_idx = {}
        self.idx_to_word = {}
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load data from file.
        
        Args:
            data_path: Path to data file
            
        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading data from {data_path}")
        
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        logger.info(f"Loaded {len(df)} records")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values and removing invalid records.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning data...")
        initial_count = len(df)
        
        # Remove records with missing text
        df = df.dropna(subset=['text'])
        
        # Remove records with missing labels
        df = df.dropna(subset=['sentiment', 'emotion'])
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['text'])
        
        # Filter out very short or very long texts
        df['text_length'] = df['text'].str.len()
        df = df[(df['text_length'] >= self.text_preprocessor.min_length) & 
                (df['text_length'] <= 5000)]  # Remove extremely long texts
        
        final_count = len(df)
        logger.info(f"Cleaned data: {initial_count} -> {final_count} records "
                   f"({final_count/initial_count:.2%} retained)")
        
        return df.drop('text_length', axis=1).reset_index(drop=True)
    
    def preprocess_texts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess all texts in the DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with preprocessed texts
        """
        logger.info("Preprocessing texts...")
        
        processed_texts = []
        valid_indices = []
        
        for idx, row in df.iterrows():
            text = row['text']
            language = row.get('language', 'en')
            
            tokens = self.text_preprocessor.preprocess_text(text, language)
            
            # Keep only non-empty results
            if tokens:
                processed_texts.append(' '.join(tokens))
                valid_indices.append(idx)
            
            if (idx + 1) % 1000 == 0:
                logger.info(f"Processed {idx + 1}/{len(df)} texts")
        
        # Filter DataFrame to keep only valid records
        df_filtered = df.iloc[valid_indices].copy()
        df_filtered['processed_text'] = processed_texts
        
        logger.info(f"Text preprocessing complete: {len(df_filtered)} valid texts")
        return df_filtered.reset_index(drop=True)
    
    def build_vocabulary(self, texts: List[str], min_freq: int = 2) -> Dict[str, int]:
        """
        Build vocabulary from preprocessed texts.
        
        Args:
            texts: List of preprocessed text strings
            min_freq: Minimum frequency for word inclusion
            
        Returns:
            Word frequency dictionary
        """
        logger.info("Building vocabulary...")
        
        word_freq = {}
        for text in texts:
            for word in text.split():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Filter by minimum frequency
        self.vocabulary = {word: freq for word, freq in word_freq.items() 
                          if freq >= min_freq}
        
        # Create word-to-index mappings
        # Reserve indices for special tokens
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        
        for word in sorted(self.vocabulary.keys()):
            self.word_to_idx[word] = len(self.word_to_idx)
        
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        logger.info(f"Vocabulary built: {len(self.vocabulary)} unique words "
                   f"(total vocab size: {len(self.word_to_idx)})")
        
        return self.vocabulary
    
    def text_to_sequence(self, text: str) -> List[int]:
        """
        Convert text to sequence of indices.
        
        Args:
            text: Preprocessed text string
            
        Returns:
            List of word indices
        """
        words = text.split()
        sequence = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in words]
        return sequence
    
    def pad_sequences(self, sequences: List[List[int]]) -> np.ndarray:
        """
        Pad sequences to uniform length.
        
        Args:
            sequences: List of sequences
            
        Returns:
            Padded array
        """
        padded = np.zeros((len(sequences), self.max_sequence_length), dtype=np.int32)
        
        for i, seq in enumerate(sequences):
            length = min(len(seq), self.max_sequence_length)
            padded[i, :length] = seq[:length]
        
        return padded
    
    def encode_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode sentiment and emotion labels.
        
        Args:
            df: DataFrame with labels
            
        Returns:
            DataFrame with encoded labels
        """
        logger.info("Encoding labels...")
        
        df = df.copy()
        
        # Encode sentiment labels
        df['sentiment_encoded'] = self.sentiment_encoder.fit_transform(df['sentiment'])
        
        # Encode emotion labels
        df['emotion_encoded'] = self.emotion_encoder.fit_transform(df['emotion'])
        
        logger.info(f"Sentiment classes: {list(self.sentiment_encoder.classes_)}")
        logger.info(f"Emotion classes: {list(self.emotion_encoder.classes_)}")
        
        return df
    
    def stratified_split(self, df: pd.DataFrame, test_size: float = 0.2, 
                        val_size: float = 0.1, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Perform stratified split of the data.
        
        Args:
            df: Input DataFrame
            test_size: Proportion for test set
            val_size: Proportion for validation set (from remaining data)
            random_state: Random seed
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Performing stratified data split...")
        
        # Create combined label for stratification
        df['combined_label'] = df['sentiment'] + '_' + df['emotion']
        
        # Check if we have enough samples for stratified split
        label_counts = df['combined_label'].value_counts()
        min_count = label_counts.min()
        
        if min_count < 2:
            logger.warning(f"Some label combinations have fewer than 2 samples. "
                          f"Minimum count: {min_count}. Using simple random split instead.")
            
            # Simple random split without stratification
            train_val_df, test_df = train_test_split(
                df, test_size=test_size, random_state=random_state
            )
            
            val_size_adjusted = val_size / (1 - test_size)
            train_df, val_df = train_test_split(
                train_val_df, test_size=val_size_adjusted, random_state=random_state
            )
        else:
            # First split: train+val vs test
            train_val_df, test_df = train_test_split(
                df, test_size=test_size, stratify=df['combined_label'], 
                random_state=random_state
            )
            
            # Second split: train vs val
            val_size_adjusted = val_size / (1 - test_size)
            train_df, val_df = train_test_split(
                train_val_df, test_size=val_size_adjusted, 
                stratify=train_val_df['combined_label'], random_state=random_state
            )
        
        # Remove temporary column
        for df_split in [train_df, val_df, test_df]:
            df_split.drop('combined_label', axis=1, inplace=True)
        
        logger.info(f"Data split complete:")
        logger.info(f"  Train: {len(train_df)} samples ({len(train_df)/len(df):.1%})")
        logger.info(f"  Val: {len(val_df)} samples ({len(val_df)/len(df):.1%})")
        logger.info(f"  Test: {len(test_df)} samples ({len(test_df)/len(df):.1%})")
        
        return train_df, val_df, test_df
    
    def save_processed_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                           test_df: pd.DataFrame, output_dir: str = "processed_data"):
        """
        Save processed data and preprocessing artifacts.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        logger.info(f"Saving processed data to {output_path}")
        
        # Save DataFrames
        train_df.to_parquet(output_path / "train.parquet")
        val_df.to_parquet(output_path / "val.parquet")
        test_df.to_parquet(output_path / "test.parquet")
        
        # Save preprocessing artifacts
        artifacts = {
            'vocabulary': self.vocabulary,
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'sentiment_encoder': self.sentiment_encoder,
            'emotion_encoder': self.emotion_encoder,
            'max_sequence_length': self.max_sequence_length
        }
        
        with open(output_path / "preprocessing_artifacts.pkl", 'wb') as f:
            pickle.dump(artifacts, f)
        
        # Save vocabulary as text file for inspection
        with open(output_path / "vocabulary.txt", 'w', encoding='utf-8') as f:
            for word, freq in sorted(self.vocabulary.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{word}\t{freq}\n")
        
        logger.info("Processed data saved successfully")
    
    def get_dataset_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate dataset statistics.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Statistics dictionary
        """
        stats = {
            'total_samples': len(df),
            'sentiment_distribution': df['sentiment'].value_counts().to_dict(),
            'emotion_distribution': df['emotion'].value_counts().to_dict(),
            'language_distribution': df['language'].value_counts().to_dict(),
            'platform_distribution': df['platform'].value_counts().to_dict(),
            'text_length_stats': {
                'mean': df['processed_text'].str.len().mean(),
                'std': df['processed_text'].str.len().std(),
                'min': df['processed_text'].str.len().min(),
                'max': df['processed_text'].str.len().max()
            }
        }
        return stats

def main():
    """
    Main function to run the preprocessing pipeline.
    """
    logger.info("Starting data preprocessing pipeline...")
    
    # Initialize pipeline
    pipeline = DataPreprocessingPipeline(max_sequence_length=256)
    
    # Load data (using the sample data generated by data_access.py)
    df = pipeline.load_data("exorde_sample_data.json")
    
    # Clean data
    df_clean = pipeline.clean_data(df)
    
    # Preprocess texts
    df_processed = pipeline.preprocess_texts(df_clean)
    
    # Encode labels
    df_encoded = pipeline.encode_labels(df_processed)
    
    # Build vocabulary
    pipeline.build_vocabulary(df_encoded['processed_text'].tolist())
    
    # Convert texts to sequences
    sequences = [pipeline.text_to_sequence(text) for text in df_encoded['processed_text']]
    padded_sequences = pipeline.pad_sequences(sequences)
    df_encoded['sequence'] = list(padded_sequences)
    
    # Split data
    train_df, val_df, test_df = pipeline.stratified_split(df_encoded)
    
    # Save processed data
    pipeline.save_processed_data(train_df, val_df, test_df)
    
    # Generate and display statistics
    train_stats = pipeline.get_dataset_statistics(train_df)
    logger.info(f"Training set statistics: {train_stats}")
    
    logger.info("Data preprocessing pipeline completed successfully!")

if __name__ == "__main__":
    main()