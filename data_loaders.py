#!/usr/bin/env python3
"""
Data Loaders for TensorFlow and PyTorch
This script implements efficient data loading pipelines for both frameworks.
"""

import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json

# TensorFlow imports
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available. TensorFlow data loaders will not work.")

# PyTorch imports
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torch.nn.utils.rnn import pad_sequence
    TORCH_AVAILABLE = True
    
    class SentimentDatasetTorch(Dataset):
        """
        PyTorch Dataset for sentiment and emotion analysis.
        """
        
        def __init__(self, df: pd.DataFrame, max_length: int = 256):
            """
            Initialize the dataset.
            
            Args:
                df: DataFrame with processed data
                max_length: Maximum sequence length
            """
            self.texts = df['processed_text'].tolist()
            self.sequences = df['sequence'].tolist()
            self.sentiment_labels = df['sentiment_encoded'].tolist()
            self.emotion_labels = df['emotion_encoded'].tolist()
            self.max_length = max_length
            
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            """
            Get a single item from the dataset.
            
            Args:
                idx: Index
                
            Returns:
                Dictionary with text, sequence, sentiment_label, emotion_label
            """
            sequence = np.array(self.sequences[idx], dtype=np.int64)
            
            # Ensure sequence is within max_length
            if len(sequence) > self.max_length:
                sequence = sequence[:self.max_length]
            
            return {
                'text': self.texts[idx],
                'sequence': torch.tensor(sequence, dtype=torch.long),
                'sentiment_label': torch.tensor(self.sentiment_labels[idx], dtype=torch.long),
                'emotion_label': torch.tensor(self.emotion_labels[idx], dtype=torch.long)
            }

    def collate_fn(batch):
        """
        Custom collate function for PyTorch DataLoader.
        
        Args:
            batch: List of samples
            
        Returns:
            Batched data
        """
        texts = [item['text'] for item in batch]
        sequences = [item['sequence'] for item in batch]
        sentiment_labels = torch.stack([item['sentiment_label'] for item in batch])
        emotion_labels = torch.stack([item['emotion_label'] for item in batch])
        
        # Pad sequences to the same length
        sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
        
        return {
            'texts': texts,
            'sequences': sequences_padded,
            'sentiment_labels': sentiment_labels,
            'emotion_labels': emotion_labels
        }
        
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. PyTorch data loaders will not work.")
    
    # Dummy classes when PyTorch is not available
    class SentimentDatasetTorch:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is not available")
    
    def collate_fn(*args, **kwargs):
        raise ImportError("PyTorch is not available")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoaderManager:
    """
    Manager class for creating and configuring data loaders.
    """
    
    def __init__(self, data_dir: str = "processed_data"):
        """
        Initialize the data loader manager.
        
        Args:
            data_dir: Directory containing processed data
        """
        self.data_dir = Path(data_dir)
        self.artifacts = self._load_artifacts()
        
    def _load_artifacts(self) -> Dict[str, Any]:
        """
        Load preprocessing artifacts.
        
        Returns:
            Dictionary with preprocessing artifacts
        """
        artifacts_path = self.data_dir / "preprocessing_artifacts.pkl"
        
        if not artifacts_path.exists():
            raise FileNotFoundError(f"Preprocessing artifacts not found at {artifacts_path}")
        
        with open(artifacts_path, 'rb') as f:
            artifacts = pickle.load(f)
        
        logger.info("Loaded preprocessing artifacts")
        return artifacts
    
    def load_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load train, validation, and test DataFrames.
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        train_df = pd.read_parquet(self.data_dir / "train.parquet")
        val_df = pd.read_parquet(self.data_dir / "val.parquet")
        test_df = pd.read_parquet(self.data_dir / "test.parquet")
        
        logger.info(f"Loaded DataFrames - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        return train_df, val_df, test_df
    
    def get_class_weights(self, train_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Calculate class weights for handling imbalanced data.
        
        Args:
            train_df: Training DataFrame
            
        Returns:
            Dictionary with class weights for sentiment and emotion
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        # Calculate sentiment class weights
        sentiment_classes = np.unique(train_df['sentiment_encoded'])
        sentiment_weights = compute_class_weight(
            'balanced', classes=sentiment_classes, y=train_df['sentiment_encoded']
        )
        
        # Calculate emotion class weights
        emotion_classes = np.unique(train_df['emotion_encoded'])
        emotion_weights = compute_class_weight(
            'balanced', classes=emotion_classes, y=train_df['emotion_encoded']
        )
        
        class_weights = {
            'sentiment': sentiment_weights,
            'emotion': emotion_weights
        }
        
        logger.info(f"Calculated class weights - Sentiment: {sentiment_weights}, Emotion: {emotion_weights}")
        return class_weights
    
    def create_tensorflow_datasets(self, batch_size: int = 32, 
                                 buffer_size: int = 1000) -> Tuple[Any, Any, Any]:
        """
        Create TensorFlow datasets.
        
        Args:
            batch_size: Batch size for training
            buffer_size: Buffer size for shuffling
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is not available")
        
        train_df, val_df, test_df = self.load_dataframes()
        max_length = self.artifacts['max_sequence_length']
        
        def create_tf_dataset(df: pd.DataFrame, shuffle: bool = False):
            """Create a TensorFlow dataset from DataFrame."""
            sequences = np.array(df['sequence'].tolist())
            sentiment_labels = df['sentiment_encoded'].values
            emotion_labels = df['emotion_encoded'].values
            
            # Ensure sequences are padded to max_length
            padded_sequences = np.zeros((len(sequences), max_length), dtype=np.int32)
            for i, seq in enumerate(sequences):
                length = min(len(seq), max_length)
                padded_sequences[i, :length] = seq[:length]
            
            dataset = tf.data.Dataset.from_tensor_slices({
                'input_ids': padded_sequences,
                'sentiment_labels': sentiment_labels,
                'emotion_labels': emotion_labels
            })
            
            if shuffle:
                dataset = dataset.shuffle(buffer_size)
            
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            
            return dataset
        
        train_dataset = create_tf_dataset(train_df, shuffle=True)
        val_dataset = create_tf_dataset(val_df, shuffle=False)
        test_dataset = create_tf_dataset(test_df, shuffle=False)
        
        logger.info(f"Created TensorFlow datasets with batch size {batch_size}")
        return train_dataset, val_dataset, test_dataset
    
    def create_pytorch_dataloaders(self, batch_size: int = 32, 
                                 num_workers: int = 0) -> Tuple[Any, Any, Any]:
        """
        Create PyTorch DataLoaders.
        
        Args:
            batch_size: Batch size for training
            num_workers: Number of worker processes
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available")
        
        train_df, val_df, test_df = self.load_dataframes()
        max_length = self.artifacts['max_sequence_length']
        
        # Create datasets
        train_dataset = SentimentDatasetTorch(train_df, max_length)
        val_dataset = SentimentDatasetTorch(val_df, max_length)
        test_dataset = SentimentDatasetTorch(test_df, max_length)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=num_workers
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=num_workers
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=num_workers
        )
        
        logger.info(f"Created PyTorch DataLoaders with batch size {batch_size}")
        return train_loader, val_loader, test_loader
    
    def get_vocab_info(self) -> Dict[str, Any]:
        """
        Get vocabulary information.
        
        Returns:
            Dictionary with vocabulary information
        """
        vocab_info = {
            'vocab_size': len(self.artifacts['word_to_idx']),
            'word_to_idx': self.artifacts['word_to_idx'],
            'idx_to_word': self.artifacts['idx_to_word'],
            'max_sequence_length': self.artifacts['max_sequence_length']
        }
        
        return vocab_info
    
    def get_label_info(self) -> Dict[str, Any]:
        """
        Get label information.
        
        Returns:
            Dictionary with label information
        """
        label_info = {
            'sentiment_classes': self.artifacts['sentiment_encoder'].classes_,
            'emotion_classes': self.artifacts['emotion_encoder'].classes_,
            'num_sentiment_classes': len(self.artifacts['sentiment_encoder'].classes_),
            'num_emotion_classes': len(self.artifacts['emotion_encoder'].classes_)
        }
        
        return label_info

def demonstrate_data_loaders():
    """
    Demonstrate the data loader functionality.
    """
    logger.info("Demonstrating data loaders...")
    
    # Initialize data loader manager
    manager = DataLoaderManager()
    
    # Get vocabulary and label information
    vocab_info = manager.get_vocab_info()
    label_info = manager.get_label_info()
    
    print("Vocabulary Information:")
    print(f"  Vocabulary size: {vocab_info['vocab_size']}")
    print(f"  Max sequence length: {vocab_info['max_sequence_length']}")
    
    print("\nLabel Information:")
    print(f"  Sentiment classes: {label_info['sentiment_classes']}")
    print(f"  Emotion classes: {label_info['emotion_classes']}")
    print(f"  Number of sentiment classes: {label_info['num_sentiment_classes']}")
    print(f"  Number of emotion classes: {label_info['num_emotion_classes']}")
    
    # Load DataFrames and calculate class weights
    train_df, val_df, test_df = manager.load_dataframes()
    class_weights = manager.get_class_weights(train_df)
    
    print(f"\nClass Weights:")
    print(f"  Sentiment weights: {class_weights['sentiment']}")
    print(f"  Emotion weights: {class_weights['emotion']}")
    
    # Test TensorFlow data loaders
    if TF_AVAILABLE:
        try:
            train_tf, val_tf, test_tf = manager.create_tensorflow_datasets(batch_size=4)
            
            print("\nTensorFlow Dataset Test:")
            for batch in train_tf.take(1):
                print(f"  Input shape: {batch['input_ids'].shape}")
                print(f"  Sentiment labels shape: {batch['sentiment_labels'].shape}")
                print(f"  Emotion labels shape: {batch['emotion_labels'].shape}")
                print(f"  Sample input: {batch['input_ids'][0][:10]}")
                print(f"  Sample sentiment label: {batch['sentiment_labels'][0]}")
                print(f"  Sample emotion label: {batch['emotion_labels'][0]}")
                break
                
        except Exception as e:
            logger.error(f"TensorFlow data loader test failed: {e}")
    
    # Test PyTorch data loaders
    if TORCH_AVAILABLE:
        try:
            train_torch, val_torch, test_torch = manager.create_pytorch_dataloaders(batch_size=4)
            
            print("\nPyTorch DataLoader Test:")
            for batch in train_torch:
                print(f"  Sequences shape: {batch['sequences'].shape}")
                print(f"  Sentiment labels shape: {batch['sentiment_labels'].shape}")
                print(f"  Emotion labels shape: {batch['emotion_labels'].shape}")
                print(f"  Sample sequence: {batch['sequences'][0][:10]}")
                print(f"  Sample sentiment label: {batch['sentiment_labels'][0]}")
                print(f"  Sample emotion label: {batch['emotion_labels'][0]}")
                break
                
        except Exception as e:
            logger.error(f"PyTorch data loader test failed: {e}")
    
    logger.info("Data loader demonstration completed!")

def main():
    """
    Main function to test data loaders.
    """
    demonstrate_data_loaders()

if __name__ == "__main__":
    main()