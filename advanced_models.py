#!/usr/bin/env python3
"""
Advanced Model Architectures for Sentiment and Emotion Analysis
This script implements BiLSTM and Transformer-based models.
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
from transformers import AutoTokenizer, AutoModel, AutoConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# TensorFlow Advanced Model Implementations
class TensorFlowAdvancedModels:
    """
    TensorFlow/Keras implementations of advanced models.
    """
    
    @staticmethod
    def create_bilstm_model(vocab_size: int, embedding_dim: int = 100,
                           hidden_dim: int = 128, num_sentiment_classes: int = 3,
                           num_emotion_classes: int = 7, max_length: int = 256,
                           dropout_rate: float = 0.3) -> Model:
        """
        Create a Bidirectional LSTM model.
        
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
        
        # Bidirectional LSTM layers
        bilstm1 = layers.Bidirectional(
            layers.LSTM(hidden_dim, return_sequences=True, dropout=dropout_rate),
            name='bilstm1'
        )(embedding)
        
        bilstm2 = layers.Bidirectional(
            layers.LSTM(hidden_dim // 2, dropout=dropout_rate),
            name='bilstm2'
        )(bilstm1)
        
        # Dense layers for feature extraction
        dense1 = layers.Dense(128, activation='relu', name='dense1')(bilstm2)
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
            name='BiLSTM_Model'
        )
        
        return model
    
    @staticmethod
    def create_transformer_model(num_sentiment_classes: int = 3,
                                num_emotion_classes: int = 7,
                                max_length: int = 256,
                                model_name: str = 'distilbert-base-uncased',
                                dropout_rate: float = 0.3,
                                trainable_layers: int = 2) -> Model:
        """
        Create a Transformer-based model using DistilBERT.
        
        Args:
            num_sentiment_classes: Number of sentiment classes
            num_emotion_classes: Number of emotion classes
            max_length: Maximum sequence length
            model_name: Pretrained model name
            dropout_rate: Dropout rate
            trainable_layers: Number of transformer layers to fine-tune
            
        Returns:
            Keras Model
        """
        # Load pretrained tokenizer and model
        try:
            from transformers import TFAutoModel
            
            # Input layers
            input_ids = layers.Input(shape=(max_length,), dtype=tf.int32, name='input_ids')
            attention_mask = layers.Input(shape=(max_length,), dtype=tf.int32, name='attention_mask')
            
            # Load pretrained transformer
            transformer = TFAutoModel.from_pretrained(
                model_name, 
                output_hidden_states=False,
                output_attentions=False
            )
            
            # Freeze most layers, keep only the last few trainable
            if trainable_layers > 0:
                for layer in transformer.layers[:-trainable_layers]:
                    layer.trainable = False
            else:
                transformer.trainable = False
            
            # Get transformer outputs
            transformer_outputs = transformer(input_ids, attention_mask=attention_mask)
            sequence_output = transformer_outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
            
            # Pooling: take [CLS] token representation
            pooled_output = sequence_output[:, 0, :]  # (batch_size, hidden_size)
            
            # Additional dense layers
            dense1 = layers.Dense(256, activation='relu', name='dense1')(pooled_output)
            dropout1 = layers.Dropout(dropout_rate, name='dropout1')(dense1)
            
            dense2 = layers.Dense(128, activation='relu', name='dense2')(dropout1)
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
                inputs=[input_ids, attention_mask],
                outputs=[sentiment_output, emotion_output],
                name='Transformer_Model'
            )
            
            return model
            
        except ImportError:
            logger.error("TensorFlow Transformers not available. Creating a simpler transformer-like model.")
            return TensorFlowAdvancedModels._create_simple_transformer(
                512, num_sentiment_classes, num_emotion_classes, max_length, dropout_rate
            )
    
    @staticmethod
    def _create_simple_transformer(vocab_size: int, num_sentiment_classes: int,
                                  num_emotion_classes: int, max_length: int,
                                  dropout_rate: float) -> Model:
        """
        Create a simple transformer-like model using MultiHeadAttention.
        
        Args:
            vocab_size: Size of vocabulary
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
        embedding_dim = 128
        embedding = layers.Embedding(
            vocab_size, embedding_dim,
            input_length=max_length,
            mask_zero=True,
            name='embedding'
        )(input_ids)
        
        # Positional encoding (simple)
        position_embedding = layers.Embedding(
            max_length, embedding_dim, name='position_embedding'
        )
        positions = tf.range(start=0, limit=max_length, delta=1)
        position_encoded = position_embedding(positions)
        
        # Add positional encoding
        encoded = embedding + position_encoded
        
        # Multi-head attention layers
        attention1 = layers.MultiHeadAttention(
            num_heads=8, key_dim=embedding_dim, name='attention1'
        )(encoded, encoded)
        
        # Add & Norm
        attention1 = layers.Add(name='add1')([encoded, attention1])
        attention1 = layers.LayerNormalization(name='norm1')(attention1)
        
        # Feed forward
        ff1 = layers.Dense(512, activation='relu', name='ff1')(attention1)
        ff1 = layers.Dropout(dropout_rate, name='ff_dropout1')(ff1)
        ff2 = layers.Dense(embedding_dim, name='ff2')(ff1)
        
        # Add & Norm
        ff_output = layers.Add(name='add2')([attention1, ff2])
        ff_output = layers.LayerNormalization(name='norm2')(ff_output)
        
        # Global average pooling
        pooled = layers.GlobalAveragePooling1D(name='global_pool')(ff_output)
        
        # Dense layers
        dense1 = layers.Dense(128, activation='relu', name='dense1')(pooled)
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
            name='SimpleTransformer_Model'
        )
        
        return model

# PyTorch Advanced Model Implementations
class PyTorchAdvancedModels:
    """
    PyTorch implementations of advanced models.
    """
    
    class BiLSTMModel(nn.Module):
        """PyTorch Bidirectional LSTM Model."""
        
        def __init__(self, vocab_size: int, embedding_dim: int = 100,
                    hidden_dim: int = 128, num_sentiment_classes: int = 3,
                    num_emotion_classes: int = 7, dropout_rate: float = 0.3,
                    num_layers: int = 2):
            super().__init__()
            
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.bilstm = nn.LSTM(
                embedding_dim, hidden_dim, 
                num_layers=num_layers, 
                bidirectional=True,
                batch_first=True, 
                dropout=dropout_rate if num_layers > 1 else 0
            )
            
            # Since bidirectional, hidden size is doubled
            bilstm_output_dim = hidden_dim * 2
            
            self.dropout = nn.Dropout(dropout_rate)
            self.dense1 = nn.Linear(bilstm_output_dim, 128)
            self.dense2 = nn.Linear(128, 64)
            self.sentiment_classifier = nn.Linear(64, num_sentiment_classes)
            self.emotion_classifier = nn.Linear(64, num_emotion_classes)
            
        def forward(self, x):
            # Embedding
            embedded = self.embedding(x)
            
            # BiLSTM
            lstm_out, _ = self.bilstm(embedded)
            # Take the last output
            last_output = lstm_out[:, -1, :]
            
            # Dense layers
            dense_out = F.relu(self.dense1(last_output))
            dense_out = self.dropout(dense_out)
            dense_out = F.relu(self.dense2(dense_out))
            dense_out = self.dropout(dense_out)
            
            # Outputs
            sentiment_logits = self.sentiment_classifier(dense_out)
            emotion_logits = self.emotion_classifier(dense_out)
            
            return sentiment_logits, emotion_logits
    
    class TransformerModel(nn.Module):
        """PyTorch Transformer Model using pretrained DistilBERT."""
        
        def __init__(self, num_sentiment_classes: int = 3,
                    num_emotion_classes: int = 7,
                    model_name: str = 'distilbert-base-uncased',
                    dropout_rate: float = 0.3,
                    trainable_layers: int = 2):
            super().__init__()
            
            try:
                # Load pretrained model
                self.transformer = AutoModel.from_pretrained(model_name)
                
                # Freeze most layers, keep only the last few trainable
                if trainable_layers > 0:
                    for param in list(self.transformer.parameters())[:-trainable_layers]:
                        param.requires_grad = False
                else:
                    for param in self.transformer.parameters():
                        param.requires_grad = False
                
                hidden_size = self.transformer.config.hidden_size
                
            except Exception as e:
                logger.warning(f"Failed to load pretrained transformer: {e}. Using simple transformer.")
                # Fallback to simple implementation
                hidden_size = 128
                self.transformer = None
            
            # Classification head
            self.dropout = nn.Dropout(dropout_rate)
            self.dense1 = nn.Linear(hidden_size, 256)
            self.dense2 = nn.Linear(256, 128)
            self.sentiment_classifier = nn.Linear(128, num_sentiment_classes)
            self.emotion_classifier = nn.Linear(128, num_emotion_classes)
            
        def forward(self, input_ids, attention_mask=None):
            if self.transformer is not None:
                # Use pretrained transformer
                outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
                pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            else:
                # Simple fallback: just use embeddings
                pooled_output = torch.mean(input_ids.float(), dim=1)
                pooled_output = F.linear(pooled_output, torch.randn(pooled_output.size(-1), 128))
            
            # Classification head
            dense_out = F.relu(self.dense1(pooled_output))
            dense_out = self.dropout(dense_out)
            dense_out = F.relu(self.dense2(dense_out))
            dense_out = self.dropout(dense_out)
            
            # Outputs
            sentiment_logits = self.sentiment_classifier(dense_out)
            emotion_logits = self.emotion_classifier(dense_out)
            
            return sentiment_logits, emotion_logits

class AdvancedModelFactory:
    """
    Factory class for creating advanced models.
    """
    
    @staticmethod
    def create_tensorflow_model(model_type: str, vocab_size: int = None,
                               num_sentiment_classes: int = 3,
                               num_emotion_classes: int = 7,
                               max_length: int = 256,
                               **kwargs) -> Model:
        """
        Create a TensorFlow advanced model.
        
        Args:
            model_type: Type of model ('bilstm', 'transformer')
            vocab_size: Size of vocabulary (required for BiLSTM)
            num_sentiment_classes: Number of sentiment classes
            num_emotion_classes: Number of emotion classes
            max_length: Maximum sequence length
            **kwargs: Additional arguments
            
        Returns:
            Keras Model
        """
        tf_models = TensorFlowAdvancedModels()
        
        if model_type.lower() == 'bilstm':
            if vocab_size is None:
                raise ValueError("vocab_size is required for BiLSTM model")
            return tf_models.create_bilstm_model(
                vocab_size, num_sentiment_classes=num_sentiment_classes,
                num_emotion_classes=num_emotion_classes, max_length=max_length, **kwargs
            )
        elif model_type.lower() == 'transformer':
            return tf_models.create_transformer_model(
                num_sentiment_classes=num_sentiment_classes,
                num_emotion_classes=num_emotion_classes, max_length=max_length, **kwargs
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def create_pytorch_model(model_type: str, vocab_size: int = None,
                           num_sentiment_classes: int = 3,
                           num_emotion_classes: int = 7,
                           **kwargs) -> nn.Module:
        """
        Create a PyTorch advanced model.
        
        Args:
            model_type: Type of model ('bilstm', 'transformer')
            vocab_size: Size of vocabulary (required for BiLSTM)
            num_sentiment_classes: Number of sentiment classes
            num_emotion_classes: Number of emotion classes
            **kwargs: Additional arguments
            
        Returns:
            PyTorch Module
        """
        pytorch_models = PyTorchAdvancedModels()
        
        if model_type.lower() == 'bilstm':
            if vocab_size is None:
                raise ValueError("vocab_size is required for BiLSTM model")
            return pytorch_models.BiLSTMModel(
                vocab_size, num_sentiment_classes=num_sentiment_classes,
                num_emotion_classes=num_emotion_classes, **kwargs
            )
        elif model_type.lower() == 'transformer':
            return pytorch_models.TransformerModel(
                num_sentiment_classes=num_sentiment_classes,
                num_emotion_classes=num_emotion_classes, **kwargs
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

def demonstrate_advanced_models():
    """
    Demonstrate the advanced models.
    """
    logger.info("Demonstrating advanced models...")
    
    # Model parameters
    vocab_size = 112  # From our vocabulary
    num_sentiment_classes = 3
    num_emotion_classes = 7
    max_length = 256
    
    model_types = ['bilstm', 'transformer']
    
    print("TensorFlow Advanced Models:")
    print("=" * 50)
    
    for model_type in model_types:
        try:
            if model_type == 'bilstm':
                model = AdvancedModelFactory.create_tensorflow_model(
                    model_type, vocab_size, num_sentiment_classes,
                    num_emotion_classes, max_length
                )
                
                # Test with dummy input
                dummy_input = tf.random.uniform((2, max_length), maxval=vocab_size, dtype=tf.int32)
                sentiment_out, emotion_out = model(dummy_input)
                
            else:  # transformer
                model = AdvancedModelFactory.create_tensorflow_model(
                    model_type, num_sentiment_classes=num_sentiment_classes,
                    num_emotion_classes=num_emotion_classes, max_length=max_length
                )
                
                # Test with dummy input for simple transformer
                dummy_input = tf.random.uniform((2, max_length), maxval=vocab_size, dtype=tf.int32)
                sentiment_out, emotion_out = model(dummy_input)
            
            print(f"\n{model_type.upper()} Model:")
            print(f"  Total parameters: {model.count_params():,}")
            print(f"  Sentiment output shape: {sentiment_out.shape}")
            print(f"  Emotion output shape: {emotion_out.shape}")
            
        except Exception as e:
            logger.error(f"Failed to create TensorFlow {model_type} model: {e}")
    
    print("\n\nPyTorch Advanced Models:")
    print("=" * 50)
    
    for model_type in model_types:
        try:
            if model_type == 'bilstm':
                model = AdvancedModelFactory.create_pytorch_model(
                    model_type, vocab_size, num_sentiment_classes, num_emotion_classes
                )
                
                # Test with dummy input
                dummy_input = torch.randint(0, vocab_size, (2, max_length))
                sentiment_out, emotion_out = model(dummy_input)
                
            else:  # transformer
                model = AdvancedModelFactory.create_pytorch_model(
                    model_type, num_sentiment_classes=num_sentiment_classes,
                    num_emotion_classes=num_emotion_classes
                )
                
                # Test with dummy input
                dummy_input = torch.randint(0, vocab_size, (2, max_length))
                attention_mask = torch.ones_like(dummy_input)
                sentiment_out, emotion_out = model(dummy_input, attention_mask=attention_mask)
            
            total_params = sum(p.numel() for p in model.parameters())
            print(f"\n{model_type.upper()} Model:")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Sentiment output shape: {sentiment_out.shape}")
            print(f"  Emotion output shape: {emotion_out.shape}")
            
        except Exception as e:
            logger.error(f"Failed to create PyTorch {model_type} model: {e}")
    
    logger.info("Advanced model demonstration completed!")

def main():
    """
    Main function to demonstrate advanced models.
    """
    demonstrate_advanced_models()

if __name__ == "__main__":
    main()