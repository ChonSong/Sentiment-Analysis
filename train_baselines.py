#!/usr/bin/env python3
"""
Training Script for Baseline Models
This script implements training loops for RNN, LSTM, GRU, and CNN-1D models.
"""

import argparse
import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List
import time

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from baseline_models import ModelFactory
from data_loaders import DataLoaderManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TensorFlowTrainer:
    """
    Trainer for TensorFlow/Keras models.
    """
    
    def __init__(self, model_type: str, vocab_size: int, num_sentiment_classes: int,
                 num_emotion_classes: int, max_length: int, class_weights: Dict[str, np.ndarray]):
        """
        Initialize the trainer.
        
        Args:
            model_type: Type of model to train
            vocab_size: Size of vocabulary
            num_sentiment_classes: Number of sentiment classes
            num_emotion_classes: Number of emotion classes
            max_length: Maximum sequence length
            class_weights: Class weights for handling imbalance
        """
        self.model_type = model_type
        self.vocab_size = vocab_size
        self.num_sentiment_classes = num_sentiment_classes
        self.num_emotion_classes = num_emotion_classes
        self.max_length = max_length
        self.class_weights = class_weights
        self.model = None
        self.history = None
        
    def create_model(self, **kwargs) -> tf.keras.Model:
        """
        Create the model.
        
        Returns:
            TensorFlow model
        """
        self.model = ModelFactory.create_tensorflow_model(
            self.model_type, self.vocab_size, 
            self.num_sentiment_classes, self.num_emotion_classes,
            self.max_length, **kwargs
        )
        
        logger.info(f"Created {self.model_type} model with {self.model.count_params():,} parameters")
        return self.model
    
    def compile_model(self, learning_rate: float = 0.001):
        """
        Compile the model.
        
        Args:
            learning_rate: Learning rate for optimizer
        """
        if self.model is None:
            raise ValueError("Model must be created before compilation")
        
        # Define optimizer
        optimizer = Adam(learning_rate=learning_rate)
        
        # Define losses with class weights
        sentiment_loss = tf.keras.losses.SparseCategoricalCrossentropy()
        emotion_loss = tf.keras.losses.SparseCategoricalCrossentropy()
        
        # Compile model
        self.model.compile(
            optimizer=optimizer,
            loss={
                'sentiment_output': sentiment_loss,
                'emotion_output': emotion_loss
            },
            metrics={
                'sentiment_output': ['accuracy'],
                'emotion_output': ['accuracy']
            },
            loss_weights={'sentiment_output': 1.0, 'emotion_output': 1.0}
        )
        
        logger.info(f"Compiled model with learning rate {learning_rate}")
    
    def train(self, train_dataset, val_dataset, epochs: int = 50, 
              batch_size: int = 32, patience: int = 5) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of epochs
            batch_size: Batch size
            patience: Early stopping patience
            
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model must be created and compiled before training")
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=f'models/{self.model_type}_best.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Train the model
        logger.info(f"Starting training for {epochs} epochs...")
        start_time = time.time()
        
        self.history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        return self.history.history
    
    def evaluate(self, test_dataset) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        logger.info("Evaluating model on test set...")
        
        # Get predictions
        predictions = self.model.predict(test_dataset)
        sentiment_probs, emotion_probs = predictions
        
        # Get true labels
        y_sentiment_true = []
        y_emotion_true = []
        
        for x_batch, y_batch in test_dataset:
            y_sentiment_true.extend(y_batch['sentiment_output'].numpy())
            y_emotion_true.extend(y_batch['emotion_output'].numpy())
        
        y_sentiment_true = np.array(y_sentiment_true)
        y_emotion_true = np.array(y_emotion_true)
        
        # Get predicted labels
        y_sentiment_pred = np.argmax(sentiment_probs, axis=1)
        y_emotion_pred = np.argmax(emotion_probs, axis=1)
        
        # Calculate metrics
        metrics = {
            'sentiment_accuracy': accuracy_score(y_sentiment_true, y_sentiment_pred),
            'emotion_accuracy': accuracy_score(y_emotion_true, y_emotion_pred),
            'sentiment_f1_macro': f1_score(y_sentiment_true, y_sentiment_pred, average='macro'),
            'emotion_f1_macro': f1_score(y_emotion_true, y_emotion_pred, average='macro'),
            'sentiment_f1_weighted': f1_score(y_sentiment_true, y_sentiment_pred, average='weighted'),
            'emotion_f1_weighted': f1_score(y_emotion_true, y_emotion_pred, average='weighted')
        }
        
        logger.info(f"Evaluation results: {metrics}")
        return metrics

class PyTorchTrainer:
    """
    Trainer for PyTorch models.
    """
    
    def __init__(self, model_type: str, vocab_size: int, num_sentiment_classes: int,
                 num_emotion_classes: int, class_weights: Dict[str, np.ndarray],
                 device: str = None):
        """
        Initialize the trainer.
        
        Args:
            model_type: Type of model to train
            vocab_size: Size of vocabulary
            num_sentiment_classes: Number of sentiment classes
            num_emotion_classes: Number of emotion classes
            class_weights: Class weights for handling imbalance
            device: Device to use for training
        """
        self.model_type = model_type
        self.vocab_size = vocab_size
        self.num_sentiment_classes = num_sentiment_classes
        self.num_emotion_classes = num_emotion_classes
        self.class_weights = class_weights
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.criterion_sentiment = None
        self.criterion_emotion = None
        self.training_history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
    def create_model(self, **kwargs) -> nn.Module:
        """
        Create the model.
        
        Returns:
            PyTorch model
        """
        self.model = ModelFactory.create_pytorch_model(
            self.model_type, self.vocab_size,
            self.num_sentiment_classes, self.num_emotion_classes, **kwargs
        )
        
        self.model.to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Created {self.model_type} model with {total_params:,} parameters on {self.device}")
        
        return self.model
    
    def setup_training(self, learning_rate: float = 0.001):
        """
        Setup optimizer and loss functions.
        
        Args:
            learning_rate: Learning rate for optimizer
        """
        if self.model is None:
            raise ValueError("Model must be created before setup")
        
        # Setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Setup loss functions with class weights
        sentiment_weights = torch.FloatTensor(self.class_weights['sentiment']).to(self.device)
        emotion_weights = torch.FloatTensor(self.class_weights['emotion']).to(self.device)
        
        self.criterion_sentiment = nn.CrossEntropyLoss(weight=sentiment_weights)
        self.criterion_emotion = nn.CrossEntropyLoss(weight=emotion_weights)
        
        logger.info(f"Setup training with learning rate {learning_rate}")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Tuple of (average_loss, average_accuracy)
        """
        self.model.train()
        total_loss = 0.0
        total_sentiment_correct = 0
        total_emotion_correct = 0
        total_samples = 0
        
        for batch in train_loader:
            sequences = batch['sequences'].to(self.device)
            sentiment_labels = batch['sentiment_labels'].to(self.device)
            emotion_labels = batch['emotion_labels'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            sentiment_logits, emotion_logits = self.model(sequences)
            
            # Calculate losses
            sentiment_loss = self.criterion_sentiment(sentiment_logits, sentiment_labels)
            emotion_loss = self.criterion_emotion(emotion_logits, emotion_labels)
            total_loss_batch = sentiment_loss + emotion_loss
            
            # Backward pass
            total_loss_batch.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += total_loss_batch.item()
            
            sentiment_pred = torch.argmax(sentiment_logits, dim=1)
            emotion_pred = torch.argmax(emotion_logits, dim=1)
            
            total_sentiment_correct += (sentiment_pred == sentiment_labels).sum().item()
            total_emotion_correct += (emotion_pred == emotion_labels).sum().item()
            total_samples += sequences.size(0)
        
        avg_loss = total_loss / len(train_loader)
        avg_accuracy = (total_sentiment_correct + total_emotion_correct) / (2 * total_samples)
        
        return avg_loss, avg_accuracy
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average_loss, average_accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        total_sentiment_correct = 0
        total_emotion_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                sequences = batch['sequences'].to(self.device)
                sentiment_labels = batch['sentiment_labels'].to(self.device)
                emotion_labels = batch['emotion_labels'].to(self.device)
                
                # Forward pass
                sentiment_logits, emotion_logits = self.model(sequences)
                
                # Calculate losses
                sentiment_loss = self.criterion_sentiment(sentiment_logits, sentiment_labels)
                emotion_loss = self.criterion_emotion(emotion_logits, emotion_labels)
                total_loss_batch = sentiment_loss + emotion_loss
                
                # Statistics
                total_loss += total_loss_batch.item()
                
                sentiment_pred = torch.argmax(sentiment_logits, dim=1)
                emotion_pred = torch.argmax(emotion_logits, dim=1)
                
                total_sentiment_correct += (sentiment_pred == sentiment_labels).sum().item()
                total_emotion_correct += (emotion_pred == emotion_labels).sum().item()
                total_samples += sequences.size(0)
        
        avg_loss = total_loss / len(val_loader)
        avg_accuracy = (total_sentiment_correct + total_emotion_correct) / (2 * total_samples)
        
        return avg_loss, avg_accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 50, patience: int = 5) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            patience: Early stopping patience
            
        Returns:
            Training history
        """
        if self.model is None or self.optimizer is None:
            raise ValueError("Model must be created and setup before training")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        logger.info(f"Starting training for {epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Store history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_acc'].append(val_acc)
            
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), f'models/{self.model_type}_best.pth')
                logger.info("New best model saved!")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        return self.training_history
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        # Load best model
        self.model.load_state_dict(torch.load(f'models/{self.model_type}_best.pth'))
        self.model.eval()
        
        logger.info("Evaluating model on test set...")
        
        y_sentiment_true = []
        y_emotion_true = []
        y_sentiment_pred = []
        y_emotion_pred = []
        
        with torch.no_grad():
            for batch in test_loader:
                sequences = batch['sequences'].to(self.device)
                sentiment_labels = batch['sentiment_labels'].to(self.device)
                emotion_labels = batch['emotion_labels'].to(self.device)
                
                # Forward pass
                sentiment_logits, emotion_logits = self.model(sequences)
                
                # Get predictions
                sentiment_pred = torch.argmax(sentiment_logits, dim=1)
                emotion_pred = torch.argmax(emotion_logits, dim=1)
                
                # Store results
                y_sentiment_true.extend(sentiment_labels.cpu().numpy())
                y_emotion_true.extend(emotion_labels.cpu().numpy())
                y_sentiment_pred.extend(sentiment_pred.cpu().numpy())
                y_emotion_pred.extend(emotion_pred.cpu().numpy())
        
        # Calculate metrics
        metrics = {
            'sentiment_accuracy': accuracy_score(y_sentiment_true, y_sentiment_pred),
            'emotion_accuracy': accuracy_score(y_emotion_true, y_emotion_pred),
            'sentiment_f1_macro': f1_score(y_sentiment_true, y_sentiment_pred, average='macro'),
            'emotion_f1_macro': f1_score(y_emotion_true, y_emotion_pred, average='macro'),
            'sentiment_f1_weighted': f1_score(y_sentiment_true, y_sentiment_pred, average='weighted'),
            'emotion_f1_weighted': f1_score(y_emotion_true, y_emotion_pred, average='weighted')
        }
        
        logger.info(f"Evaluation results: {metrics}")
        return metrics

def train_model(args):
    """
    Train a single model.
    
    Args:
        args: Parsed command line arguments
    """
    logger.info(f"Training {args.model_type} model with {args.framework}")
    
    # Load data
    data_manager = DataLoaderManager()
    vocab_info = data_manager.get_vocab_info()
    label_info = data_manager.get_label_info()
    
    # Get class weights
    train_df, val_df, test_df = data_manager.load_dataframes()
    class_weights = data_manager.get_class_weights(train_df)
    
    if args.framework.lower() == 'tensorflow':
        # Create TensorFlow datasets
        train_dataset, val_dataset, test_dataset = data_manager.create_tensorflow_datasets(
            batch_size=args.batch_size
        )
        
        # Create and train model
        trainer = TensorFlowTrainer(
            args.model_type, vocab_info['vocab_size'],
            label_info['num_sentiment_classes'], label_info['num_emotion_classes'],
            vocab_info['max_sequence_length'], class_weights
        )
        
        trainer.create_model(dropout_rate=args.dropout_rate)
        trainer.compile_model(learning_rate=args.learning_rate)
        
        history = trainer.train(
            train_dataset, val_dataset, epochs=args.epochs,
            batch_size=args.batch_size, patience=args.patience
        )
        
        metrics = trainer.evaluate(test_dataset)
        
    elif args.framework.lower() == 'pytorch':
        # Create PyTorch data loaders
        train_loader, val_loader, test_loader = data_manager.create_pytorch_dataloaders(
            batch_size=args.batch_size
        )
        
        # Create and train model
        trainer = PyTorchTrainer(
            args.model_type, vocab_info['vocab_size'],
            label_info['num_sentiment_classes'], label_info['num_emotion_classes'],
            class_weights
        )
        
        trainer.create_model(dropout_rate=args.dropout_rate)
        trainer.setup_training(learning_rate=args.learning_rate)
        
        history = trainer.train(
            train_loader, val_loader, epochs=args.epochs, patience=args.patience
        )
        
        metrics = trainer.evaluate(test_loader)
    
    else:
        raise ValueError(f"Unknown framework: {args.framework}")
    
    # Save results
    results = {
        'model_type': args.model_type,
        'framework': args.framework,
        'hyperparameters': {
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'dropout_rate': args.dropout_rate,
            'epochs': args.epochs
        },
        'metrics': metrics,
        'history': history
    }
    
    results_file = f'results/{args.model_type}_{args.framework}_results.json'
    os.makedirs('results', exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {results_file}")

def main():
    """
    Main function for training baseline models.
    """
    parser = argparse.ArgumentParser(description='Train baseline models for sentiment analysis')
    
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['rnn', 'lstm', 'gru', 'cnn1d'],
                       help='Type of model to train')
    parser.add_argument('--framework', type=str, default='tensorflow',
                       choices=['tensorflow', 'pytorch'],
                       help='Framework to use')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience')
    
    args = parser.parse_args()
    
    train_model(args)

if __name__ == "__main__":
    main()