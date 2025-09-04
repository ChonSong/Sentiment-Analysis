#!/usr/bin/env python3
"""
Baseline Model Architectures for Sentiment and Emotion Analysis
This script implements RNN, LSTM, GRU, and CNN-1D models using TensorFlow/Keras.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# TensorFlow Model Implementations
class TensorFlowBaselineModels:
    """
    TensorFlow/Keras implementations of baseline models.
    """
    
    @staticmethod
    def create_rnn_model(vocab_size: int, embedding_dim: int = 100, 
                        hidden_dim: int = 128, num_sentiment_classes: int = 3,
                        num_emotion_classes: int = 7, max_length: int = 256,
                        dropout_rate: float = 0.3) -> Model:
        """
        Create a simple RNN model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            hidden_dim: Hidden dimension for RNN
            num_sentiment_classes: Number of sentiment classes
            num_emotion_classes: Number of emotion classes
            max_length: Maximum sequence length
            dropout_rate: Dropout rate
            
        Returns:
            Keras Model
        """
        # Input layer
        input_ids = layers.Input(shape=(max_length,), name='input_ids')
        
        # Embedding layer
        embedding = layers.Embedding(
            vocab_size, embedding_dim, 
            input_length=max_length,
            mask_zero=True,
            name='embedding'
        )(input_ids)
        
        # RNN layer
        rnn_output = layers.SimpleRNN(
            hidden_dim, 
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate,
            name='rnn'
        )(embedding)
        
        # Dense layers for feature extraction
        dense1 = layers.Dense(64, activation='relu', name='dense1')(rnn_output)
        dropout1 = layers.Dropout(dropout_rate, name='dropout1')(dense1)
        
        # Output layers
        sentiment_output = layers.Dense(
            num_sentiment_classes, 
            activation='softmax',
            name='sentiment_output'
        )(dropout1)
        
        emotion_output = layers.Dense(
            num_emotion_classes,
            activation='softmax', 
            name='emotion_output'
        )(dropout1)
        
        # Create model
        model = Model(
            inputs=input_ids,
            outputs=[sentiment_output, emotion_output],
            name='RNN_Model'
        )
        
        return model
    
    @staticmethod
    def create_lstm_model(vocab_size: int, embedding_dim: int = 100,
                         hidden_dim: int = 128, num_sentiment_classes: int = 3,
                         num_emotion_classes: int = 7, max_length: int = 256,
                         dropout_rate: float = 0.3) -> Model:
        """
        Create an LSTM model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            hidden_dim: Hidden dimension for LSTM
            num_sentiment_classes: Number of sentiment classes
            num_emotion_classes: Number of emotion classes
            max_length: Maximum sequence length
            dropout_rate: Dropout rate
            
        Returns:
            Keras Model
        """
        # Input layer
        input_ids = layers.Input(shape=(max_length,), name='input_ids')
        
        # Embedding layer
        embedding = layers.Embedding(
            vocab_size, embedding_dim,
            input_length=max_length,
            mask_zero=True,
            name='embedding'
        )(input_ids)
        
        # LSTM layer
        lstm_output = layers.LSTM(
            hidden_dim,
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate,
            name='lstm'
        )(embedding)
        
        # Dense layers for feature extraction
        dense1 = layers.Dense(64, activation='relu', name='dense1')(lstm_output)
        dropout1 = layers.Dropout(dropout_rate, name='dropout1')(dense1)
        
        # Output layers
        sentiment_output = layers.Dense(
            num_sentiment_classes,
            activation='softmax',
            name='sentiment_output'
        )(dropout1)
        
        emotion_output = layers.Dense(
            num_emotion_classes,
            activation='softmax',
            name='emotion_output'
        )(dropout1)
        
        # Create model
        model = Model(
            inputs=input_ids,
            outputs=[sentiment_output, emotion_output],
            name='LSTM_Model'
        )
        
        return model
    
    @staticmethod
    def create_gru_model(vocab_size: int, embedding_dim: int = 100,
                        hidden_dim: int = 128, num_sentiment_classes: int = 3,
                        num_emotion_classes: int = 7, max_length: int = 256,
                        dropout_rate: float = 0.3) -> Model:
        """
        Create a GRU model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            hidden_dim: Hidden dimension for GRU
            num_sentiment_classes: Number of sentiment classes
            num_emotion_classes: Number of emotion classes
            max_length: Maximum sequence length
            dropout_rate: Dropout rate
            
        Returns:
            Keras Model
        """
        # Input layer
        input_ids = layers.Input(shape=(max_length,), name='input_ids')
        
        # Embedding layer
        embedding = layers.Embedding(
            vocab_size, embedding_dim,
            input_length=max_length,
            mask_zero=True,
            name='embedding'
        )(input_ids)
        
        # GRU layer
        gru_output = layers.GRU(
            hidden_dim,
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate,
            name='gru'
        )(embedding)
        
        # Dense layers for feature extraction
        dense1 = layers.Dense(64, activation='relu', name='dense1')(gru_output)
        dropout1 = layers.Dropout(dropout_rate, name='dropout1')(dense1)
        
        # Output layers
        sentiment_output = layers.Dense(
            num_sentiment_classes,
            activation='softmax',
            name='sentiment_output'
        )(dropout1)
        
        emotion_output = layers.Dense(
            num_emotion_classes,
            activation='softmax',
            name='emotion_output'
        )(dropout1)
        
        # Create model
        model = Model(
            inputs=input_ids,
            outputs=[sentiment_output, emotion_output],
            name='GRU_Model'
        )
        
        return model
    
    @staticmethod
    def create_cnn1d_model(vocab_size: int, embedding_dim: int = 100,
                          num_filters: int = 128, filter_sizes: list = [3, 4, 5],
                          num_sentiment_classes: int = 3, num_emotion_classes: int = 7,
                          max_length: int = 256, dropout_rate: float = 0.3) -> Model:
        """
        Create a 1D CNN model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            num_filters: Number of filters for each filter size
            filter_sizes: List of filter sizes
            num_sentiment_classes: Number of sentiment classes
            num_emotion_classes: Number of emotion classes
            max_length: Maximum sequence length
            dropout_rate: Dropout rate
            
        Returns:
            Keras Model
        """
        # Input layer
        input_ids = layers.Input(shape=(max_length,), name='input_ids')
        
        # Embedding layer
        embedding = layers.Embedding(
            vocab_size, embedding_dim,
            input_length=max_length,
            name='embedding'
        )(input_ids)
        
        # Multiple CNN branches with different filter sizes
        conv_outputs = []
        for filter_size in filter_sizes:
            conv = layers.Conv1D(
                num_filters, filter_size,
                activation='relu',
                name=f'conv1d_{filter_size}'
            )(embedding)
            
            # Global max pooling
            pool = layers.GlobalMaxPooling1D(
                name=f'global_max_pool_{filter_size}'
            )(conv)
            
            conv_outputs.append(pool)
        
        # Concatenate all CNN outputs
        if len(conv_outputs) > 1:
            concatenated = layers.Concatenate(name='concat')(conv_outputs)
        else:
            concatenated = conv_outputs[0]
        
        # Dense layers
        dense1 = layers.Dense(128, activation='relu', name='dense1')(concatenated)
        dropout1 = layers.Dropout(dropout_rate, name='dropout1')(dense1)
        
        dense2 = layers.Dense(64, activation='relu', name='dense2')(dropout1)
        dropout2 = layers.Dropout(dropout_rate, name='dropout2')(dense2)
        
        # Output layers
        sentiment_output = layers.Dense(
            num_sentiment_classes,
            activation='softmax',
            name='sentiment_output'
        )(dropout2)
        
        emotion_output = layers.Dense(
            num_emotion_classes,
            activation='softmax',
            name='emotion_output'
        )(dropout2)
        
        # Create model
        model = Model(
            inputs=input_ids,
            outputs=[sentiment_output, emotion_output],
            name='CNN1D_Model'
        )
        
        return model

# PyTorch Model Implementations
class PyTorchBaselineModels:
    """
    PyTorch implementations of baseline models.
    """
    
    class RNNModel(nn.Module):
        """PyTorch RNN Model."""
        
        def __init__(self, vocab_size: int, embedding_dim: int = 100,
                    hidden_dim: int = 128, num_sentiment_classes: int = 3,
                    num_emotion_classes: int = 7, dropout_rate: float = 0.3):
            super().__init__()
            
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True, dropout=dropout_rate)
            self.dropout = nn.Dropout(dropout_rate)
            self.dense1 = nn.Linear(hidden_dim, 64)
            self.sentiment_classifier = nn.Linear(64, num_sentiment_classes)
            self.emotion_classifier = nn.Linear(64, num_emotion_classes)
            
        def forward(self, x):
            # Embedding
            embedded = self.embedding(x)
            
            # RNN
            rnn_out, _ = self.rnn(embedded)
            # Take the last output
            last_output = rnn_out[:, -1, :]
            
            # Dense layers
            dense_out = F.relu(self.dense1(last_output))
            dense_out = self.dropout(dense_out)
            
            # Outputs
            sentiment_logits = self.sentiment_classifier(dense_out)
            emotion_logits = self.emotion_classifier(dense_out)
            
            return sentiment_logits, emotion_logits
    
    class LSTMModel(nn.Module):
        """PyTorch LSTM Model."""
        
        def __init__(self, vocab_size: int, embedding_dim: int = 100,
                    hidden_dim: int = 128, num_sentiment_classes: int = 3,
                    num_emotion_classes: int = 7, dropout_rate: float = 0.3):
            super().__init__()
            
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, dropout=dropout_rate)
            self.dropout = nn.Dropout(dropout_rate)
            self.dense1 = nn.Linear(hidden_dim, 64)
            self.sentiment_classifier = nn.Linear(64, num_sentiment_classes)
            self.emotion_classifier = nn.Linear(64, num_emotion_classes)
            
        def forward(self, x):
            # Embedding
            embedded = self.embedding(x)
            
            # LSTM
            lstm_out, _ = self.lstm(embedded)
            # Take the last output
            last_output = lstm_out[:, -1, :]
            
            # Dense layers
            dense_out = F.relu(self.dense1(last_output))
            dense_out = self.dropout(dense_out)
            
            # Outputs
            sentiment_logits = self.sentiment_classifier(dense_out)
            emotion_logits = self.emotion_classifier(dense_out)
            
            return sentiment_logits, emotion_logits
    
    class GRUModel(nn.Module):
        """PyTorch GRU Model."""
        
        def __init__(self, vocab_size: int, embedding_dim: int = 100,
                    hidden_dim: int = 128, num_sentiment_classes: int = 3,
                    num_emotion_classes: int = 7, dropout_rate: float = 0.3):
            super().__init__()
            
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, dropout=dropout_rate)
            self.dropout = nn.Dropout(dropout_rate)
            self.dense1 = nn.Linear(hidden_dim, 64)
            self.sentiment_classifier = nn.Linear(64, num_sentiment_classes)
            self.emotion_classifier = nn.Linear(64, num_emotion_classes)
            
        def forward(self, x):
            # Embedding
            embedded = self.embedding(x)
            
            # GRU
            gru_out, _ = self.gru(embedded)
            # Take the last output
            last_output = gru_out[:, -1, :]
            
            # Dense layers
            dense_out = F.relu(self.dense1(last_output))
            dense_out = self.dropout(dense_out)
            
            # Outputs
            sentiment_logits = self.sentiment_classifier(dense_out)
            emotion_logits = self.emotion_classifier(dense_out)
            
            return sentiment_logits, emotion_logits
    
    class CNN1DModel(nn.Module):
        """PyTorch 1D CNN Model."""
        
        def __init__(self, vocab_size: int, embedding_dim: int = 100,
                    num_filters: int = 128, filter_sizes: list = [3, 4, 5],
                    num_sentiment_classes: int = 3, num_emotion_classes: int = 7,
                    dropout_rate: float = 0.3):
            super().__init__()
            
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            
            # Multiple conv layers
            self.convs = nn.ModuleList([
                nn.Conv1d(embedding_dim, num_filters, kernel_size=fs)
                for fs in filter_sizes
            ])
            
            self.dropout = nn.Dropout(dropout_rate)
            total_filters = len(filter_sizes) * num_filters
            self.dense1 = nn.Linear(total_filters, 128)
            self.dense2 = nn.Linear(128, 64)
            self.sentiment_classifier = nn.Linear(64, num_sentiment_classes)
            self.emotion_classifier = nn.Linear(64, num_emotion_classes)
            
        def forward(self, x):
            # Embedding
            embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
            embedded = embedded.transpose(1, 2)  # (batch_size, embedding_dim, seq_len)
            
            # Apply convolutions
            conv_outputs = []
            for conv in self.convs:
                conv_out = F.relu(conv(embedded))  # (batch_size, num_filters, conv_len)
                pooled = F.max_pool1d(conv_out, conv_out.size(2))  # (batch_size, num_filters, 1)
                conv_outputs.append(pooled.squeeze(2))  # (batch_size, num_filters)
            
            # Concatenate
            concatenated = torch.cat(conv_outputs, dim=1)  # (batch_size, total_filters)
            
            # Dense layers
            dense_out = F.relu(self.dense1(concatenated))
            dense_out = self.dropout(dense_out)
            dense_out = F.relu(self.dense2(dense_out))
            dense_out = self.dropout(dense_out)
            
            # Outputs
            sentiment_logits = self.sentiment_classifier(dense_out)
            emotion_logits = self.emotion_classifier(dense_out)
            
            return sentiment_logits, emotion_logits

class ModelFactory:
    """
    Factory class for creating baseline models.
    """
    
    @staticmethod
    def create_tensorflow_model(model_type: str, vocab_size: int, 
                               num_sentiment_classes: int = 3,
                               num_emotion_classes: int = 7,
                               max_length: int = 256,
                               **kwargs) -> Model:
        """
        Create a TensorFlow model.
        
        Args:
            model_type: Type of model ('rnn', 'lstm', 'gru', 'cnn1d')
            vocab_size: Size of vocabulary
            num_sentiment_classes: Number of sentiment classes
            num_emotion_classes: Number of emotion classes
            max_length: Maximum sequence length
            **kwargs: Additional arguments
            
        Returns:
            Keras Model
        """
        tf_models = TensorFlowBaselineModels()
        
        if model_type.lower() == 'rnn':
            return tf_models.create_rnn_model(
                vocab_size, num_sentiment_classes=num_sentiment_classes,
                num_emotion_classes=num_emotion_classes, max_length=max_length, **kwargs
            )
        elif model_type.lower() == 'lstm':
            return tf_models.create_lstm_model(
                vocab_size, num_sentiment_classes=num_sentiment_classes,
                num_emotion_classes=num_emotion_classes, max_length=max_length, **kwargs
            )
        elif model_type.lower() == 'gru':
            return tf_models.create_gru_model(
                vocab_size, num_sentiment_classes=num_sentiment_classes,
                num_emotion_classes=num_emotion_classes, max_length=max_length, **kwargs
            )
        elif model_type.lower() == 'cnn1d':
            return tf_models.create_cnn1d_model(
                vocab_size, num_sentiment_classes=num_sentiment_classes,
                num_emotion_classes=num_emotion_classes, max_length=max_length, **kwargs
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def create_pytorch_model(model_type: str, vocab_size: int,
                           num_sentiment_classes: int = 3,
                           num_emotion_classes: int = 7,
                           **kwargs) -> nn.Module:
        """
        Create a PyTorch model.
        
        Args:
            model_type: Type of model ('rnn', 'lstm', 'gru', 'cnn1d')
            vocab_size: Size of vocabulary
            num_sentiment_classes: Number of sentiment classes
            num_emotion_classes: Number of emotion classes
            **kwargs: Additional arguments
            
        Returns:
            PyTorch Module
        """
        pytorch_models = PyTorchBaselineModels()
        
        if model_type.lower() == 'rnn':
            return pytorch_models.RNNModel(
                vocab_size, num_sentiment_classes=num_sentiment_classes,
                num_emotion_classes=num_emotion_classes, **kwargs
            )
        elif model_type.lower() == 'lstm':
            return pytorch_models.LSTMModel(
                vocab_size, num_sentiment_classes=num_sentiment_classes,
                num_emotion_classes=num_emotion_classes, **kwargs
            )
        elif model_type.lower() == 'gru':
            return pytorch_models.GRUModel(
                vocab_size, num_sentiment_classes=num_sentiment_classes,
                num_emotion_classes=num_emotion_classes, **kwargs
            )
        elif model_type.lower() == 'cnn1d':
            return pytorch_models.CNN1DModel(
                vocab_size, num_sentiment_classes=num_sentiment_classes,
                num_emotion_classes=num_emotion_classes, **kwargs
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

def demonstrate_models():
    """
    Demonstrate the baseline models.
    """
    logger.info("Demonstrating baseline models...")
    
    # Model parameters
    vocab_size = 112  # From our vocabulary
    num_sentiment_classes = 3
    num_emotion_classes = 7
    max_length = 256
    
    model_types = ['rnn', 'lstm', 'gru', 'cnn1d']
    
    print("TensorFlow Models:")
    print("=" * 50)
    
    for model_type in model_types:
        try:
            model = ModelFactory.create_tensorflow_model(
                model_type, vocab_size, num_sentiment_classes,
                num_emotion_classes, max_length
            )
            
            print(f"\n{model_type.upper()} Model:")
            print(f"  Total parameters: {model.count_params():,}")
            
            # Test with dummy input
            dummy_input = tf.random.uniform((2, max_length), maxval=vocab_size, dtype=tf.int32)
            sentiment_out, emotion_out = model(dummy_input)
            print(f"  Sentiment output shape: {sentiment_out.shape}")
            print(f"  Emotion output shape: {emotion_out.shape}")
            
        except Exception as e:
            logger.error(f"Failed to create TensorFlow {model_type} model: {e}")
    
    print("\n\nPyTorch Models:")
    print("=" * 50)
    
    for model_type in model_types:
        try:
            model = ModelFactory.create_pytorch_model(
                model_type, vocab_size, num_sentiment_classes, num_emotion_classes
            )
            
            total_params = sum(p.numel() for p in model.parameters())
            print(f"\n{model_type.upper()} Model:")
            print(f"  Total parameters: {total_params:,}")
            
            # Test with dummy input
            dummy_input = torch.randint(0, vocab_size, (2, max_length))
            sentiment_out, emotion_out = model(dummy_input)
            print(f"  Sentiment output shape: {sentiment_out.shape}")
            print(f"  Emotion output shape: {emotion_out.shape}")
            
        except Exception as e:
            logger.error(f"Failed to create PyTorch {model_type} model: {e}")
    
    logger.info("Model demonstration completed!")

def main():
    """
    Main function to demonstrate models.
    """
    demonstrate_models()

if __name__ == "__main__":
    main()