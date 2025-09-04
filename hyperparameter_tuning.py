#!/usr/bin/env python3
"""
Hyperparameter Tuning Script for Sentiment and Emotion Analysis Models
This script implements systematic hyperparameter optimization for models.
"""

import argparse
import os
import json
import logging
import numpy as np
import itertools
from pathlib import Path
from typing import Dict, Any, List, Tuple
import time
import random

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

from baseline_models import ModelFactory as BaselineModelFactory
from advanced_models import AdvancedModelFactory
from data_loaders import DataLoaderManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HyperparameterTuner:
    """
    Hyperparameter tuning class for sentiment analysis models.
    """
    
    def __init__(self, model_type: str, framework: str, data_manager: DataLoaderManager,
                 search_type: str = 'random', max_trials: int = 20):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            model_type: Type of model to tune
            framework: Framework to use ('tensorflow' or 'pytorch')
            data_manager: Data loader manager
            search_type: Type of search ('random' or 'grid')
            max_trials: Maximum number of trials
        """
        self.model_type = model_type
        self.framework = framework
        self.data_manager = data_manager
        self.search_type = search_type
        self.max_trials = max_trials
        
        # Get data info
        self.vocab_info = data_manager.get_vocab_info()
        self.label_info = data_manager.get_label_info()
        
        # Get class weights
        train_df, val_df, test_df = data_manager.load_dataframes()
        self.class_weights = data_manager.get_class_weights(train_df)
        
        # Define hyperparameter search space
        self.search_space = self._define_search_space()
        
        # Store results
        self.trial_results = []
        
    def _define_search_space(self) -> Dict[str, List[Any]]:
        """
        Define the hyperparameter search space.
        
        Returns:
            Dictionary of hyperparameter options
        """
        base_space = {
            'learning_rate': [0.001, 0.0005, 0.002, 0.0001],
            'batch_size': [16, 32, 64],
            'dropout_rate': [0.2, 0.3, 0.4, 0.5],
            'embedding_dim': [50, 100, 150] if self.model_type != 'transformer' else [100],
        }
        
        if self.model_type in ['rnn', 'lstm', 'gru', 'bilstm']:
            base_space.update({
                'hidden_dim': [64, 128, 256],
            })
        elif self.model_type == 'cnn1d':
            base_space.update({
                'num_filters': [64, 128, 256],
                'filter_sizes': [[3, 4, 5], [2, 3, 4], [3, 5, 7]],
            })
        elif self.model_type == 'transformer':
            base_space.update({
                'trainable_layers': [0, 1, 2, 3],
            })
        
        return base_space
    
    def _generate_hyperparameter_combinations(self) -> List[Dict[str, Any]]:
        """
        Generate hyperparameter combinations based on search type.
        
        Returns:
            List of hyperparameter dictionaries
        """
        if self.search_type == 'grid':
            # Grid search: all combinations
            keys = list(self.search_space.keys())
            values = list(self.search_space.values())
            combinations = []
            
            for combination in itertools.product(*values):
                param_dict = dict(zip(keys, combination))
                combinations.append(param_dict)
                
                if len(combinations) >= self.max_trials:
                    break
            
            return combinations
        
        else:  # random search
            combinations = []
            for _ in range(self.max_trials):
                param_dict = {}
                for param, options in self.search_space.items():
                    param_dict[param] = random.choice(options)
                combinations.append(param_dict)
            
            return combinations
    
    def _create_model(self, hyperparams: Dict[str, Any]):
        """
        Create a model with given hyperparameters.
        
        Args:
            hyperparams: Hyperparameter dictionary
            
        Returns:
            Model instance
        """
        # Extract common parameters
        common_params = {
            'dropout_rate': hyperparams['dropout_rate'],
        }
        
        # Add model-specific parameters
        if self.model_type in ['rnn', 'lstm', 'gru', 'bilstm']:
            common_params.update({
                'embedding_dim': hyperparams['embedding_dim'],
                'hidden_dim': hyperparams['hidden_dim'],
            })
        elif self.model_type == 'cnn1d':
            common_params.update({
                'embedding_dim': hyperparams['embedding_dim'],
                'num_filters': hyperparams['num_filters'],
                'filter_sizes': hyperparams['filter_sizes'],
            })
        elif self.model_type == 'transformer':
            common_params.update({
                'trainable_layers': hyperparams['trainable_layers'],
            })
        
        if self.framework == 'tensorflow':
            if self.model_type in ['bilstm', 'transformer']:
                if self.model_type == 'transformer':
                    return AdvancedModelFactory.create_tensorflow_model(
                        self.model_type,
                        num_sentiment_classes=self.label_info['num_sentiment_classes'],
                        num_emotion_classes=self.label_info['num_emotion_classes'],
                        max_length=self.vocab_info['max_sequence_length'],
                        **common_params
                    )
                else:
                    return AdvancedModelFactory.create_tensorflow_model(
                        self.model_type, self.vocab_info['vocab_size'],
                        self.label_info['num_sentiment_classes'],
                        self.label_info['num_emotion_classes'],
                        self.vocab_info['max_sequence_length'],
                        **common_params
                    )
            else:
                return BaselineModelFactory.create_tensorflow_model(
                    self.model_type, self.vocab_info['vocab_size'],
                    self.label_info['num_sentiment_classes'],
                    self.label_info['num_emotion_classes'],
                    self.vocab_info['max_sequence_length'],
                    **common_params
                )
        else:  # pytorch
            if self.model_type in ['bilstm', 'transformer']:
                if self.model_type == 'transformer':
                    return AdvancedModelFactory.create_pytorch_model(
                        self.model_type,
                        num_sentiment_classes=self.label_info['num_sentiment_classes'],
                        num_emotion_classes=self.label_info['num_emotion_classes'],
                        **common_params
                    )
                else:
                    return AdvancedModelFactory.create_pytorch_model(
                        self.model_type, self.vocab_info['vocab_size'],
                        self.label_info['num_sentiment_classes'],
                        self.label_info['num_emotion_classes'],
                        **common_params
                    )
            else:
                return BaselineModelFactory.create_pytorch_model(
                    self.model_type, self.vocab_info['vocab_size'],
                    self.label_info['num_sentiment_classes'],
                    self.label_info['num_emotion_classes'],
                    **common_params
                )
    
    def _train_and_evaluate_tensorflow(self, model: tf.keras.Model, 
                                     hyperparams: Dict[str, Any]) -> float:
        """
        Train and evaluate a TensorFlow model.
        
        Args:
            model: TensorFlow model
            hyperparams: Hyperparameters
            
        Returns:
            Validation macro F1 score
        """
        # Create datasets
        train_dataset, val_dataset, _ = self.data_manager.create_tensorflow_datasets(
            batch_size=hyperparams['batch_size']
        )
        
        # Compile model
        optimizer = Adam(learning_rate=hyperparams['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss={
                'sentiment_output': 'sparse_categorical_crossentropy',
                'emotion_output': 'sparse_categorical_crossentropy'
            },
            metrics={
                'sentiment_output': ['accuracy'],
                'emotion_output': ['accuracy']
            }
        )
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True,
                verbose=0
            )
        ]
        
        # Train model
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=10,  # Limited epochs for hyperparameter tuning
            callbacks=callbacks,
            verbose=0
        )
        
        # Get validation predictions
        val_predictions = model.predict(val_dataset, verbose=0)
        sentiment_probs, emotion_probs = val_predictions
        
        # Get true labels
        y_sentiment_true = []
        y_emotion_true = []
        for x_batch, y_batch in val_dataset:
            y_sentiment_true.extend(y_batch['sentiment_output'].numpy())
            y_emotion_true.extend(y_batch['emotion_output'].numpy())
        
        # Calculate F1 scores
        y_sentiment_pred = np.argmax(sentiment_probs, axis=1)
        y_emotion_pred = np.argmax(emotion_probs, axis=1)
        
        sentiment_f1 = f1_score(y_sentiment_true, y_sentiment_pred, average='macro')
        emotion_f1 = f1_score(y_emotion_true, y_emotion_pred, average='macro')
        
        # Return average F1 score
        return (sentiment_f1 + emotion_f1) / 2
    
    def _train_and_evaluate_pytorch(self, model: nn.Module, 
                                  hyperparams: Dict[str, Any]) -> float:
        """
        Train and evaluate a PyTorch model.
        
        Args:
            model: PyTorch model
            hyperparams: Hyperparameters
            
        Returns:
            Validation macro F1 score
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        
        # Create data loaders
        train_loader, val_loader, _ = self.data_manager.create_pytorch_dataloaders(
            batch_size=hyperparams['batch_size']
        )
        
        # Setup training
        optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
        
        sentiment_weights = torch.FloatTensor(self.class_weights['sentiment']).to(device)
        emotion_weights = torch.FloatTensor(self.class_weights['emotion']).to(device)
        
        criterion_sentiment = nn.CrossEntropyLoss(weight=sentiment_weights)
        criterion_emotion = nn.CrossEntropyLoss(weight=emotion_weights)
        
        # Training loop
        model.train()
        best_val_f1 = 0.0
        patience_counter = 0
        
        for epoch in range(10):  # Limited epochs for hyperparameter tuning
            # Training
            for batch in train_loader:
                sequences = batch['sequences'].to(device)
                sentiment_labels = batch['sentiment_labels'].to(device)
                emotion_labels = batch['emotion_labels'].to(device)
                
                optimizer.zero_grad()
                
                if self.model_type == 'transformer':
                    attention_mask = torch.ones_like(sequences)
                    sentiment_logits, emotion_logits = model(sequences, attention_mask=attention_mask)
                else:
                    sentiment_logits, emotion_logits = model(sequences)
                
                sentiment_loss = criterion_sentiment(sentiment_logits, sentiment_labels)
                emotion_loss = criterion_emotion(emotion_logits, emotion_labels)
                total_loss = sentiment_loss + emotion_loss
                
                total_loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            y_sentiment_true = []
            y_emotion_true = []
            y_sentiment_pred = []
            y_emotion_pred = []
            
            with torch.no_grad():
                for batch in val_loader:
                    sequences = batch['sequences'].to(device)
                    sentiment_labels = batch['sentiment_labels'].to(device)
                    emotion_labels = batch['emotion_labels'].to(device)
                    
                    if self.model_type == 'transformer':
                        attention_mask = torch.ones_like(sequences)
                        sentiment_logits, emotion_logits = model(sequences, attention_mask=attention_mask)
                    else:
                        sentiment_logits, emotion_logits = model(sequences)
                    
                    sentiment_pred = torch.argmax(sentiment_logits, dim=1)
                    emotion_pred = torch.argmax(emotion_logits, dim=1)
                    
                    y_sentiment_true.extend(sentiment_labels.cpu().numpy())
                    y_emotion_true.extend(emotion_labels.cpu().numpy())
                    y_sentiment_pred.extend(sentiment_pred.cpu().numpy())
                    y_emotion_pred.extend(emotion_pred.cpu().numpy())
            
            # Calculate F1 scores
            sentiment_f1 = f1_score(y_sentiment_true, y_sentiment_pred, average='macro')
            emotion_f1 = f1_score(y_emotion_true, y_emotion_pred, average='macro')
            val_f1 = (sentiment_f1 + emotion_f1) / 2
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 3:  # Early stopping
                    break
            
            model.train()
        
        return best_val_f1
    
    def run_hyperparameter_search(self) -> Dict[str, Any]:
        """
        Run the hyperparameter search.
        
        Returns:
            Best hyperparameters and results
        """
        logger.info(f"Starting {self.search_type} hyperparameter search for {self.model_type}")
        logger.info(f"Maximum trials: {self.max_trials}")
        
        # Generate hyperparameter combinations
        param_combinations = self._generate_hyperparameter_combinations()
        logger.info(f"Generated {len(param_combinations)} parameter combinations")
        
        best_score = 0.0
        best_params = None
        
        for trial_idx, hyperparams in enumerate(param_combinations):
            logger.info(f"Trial {trial_idx + 1}/{len(param_combinations)}: {hyperparams}")
            
            try:
                start_time = time.time()
                
                # Create model
                model = self._create_model(hyperparams)
                
                # Train and evaluate
                if self.framework == 'tensorflow':
                    score = self._train_and_evaluate_tensorflow(model, hyperparams)
                else:
                    score = self._train_and_evaluate_pytorch(model, hyperparams)
                
                trial_time = time.time() - start_time
                
                # Store results
                trial_result = {
                    'trial': trial_idx + 1,
                    'hyperparams': hyperparams,
                    'score': score,
                    'time': trial_time
                }
                self.trial_results.append(trial_result)
                
                logger.info(f"Trial {trial_idx + 1} completed - Score: {score:.4f}, Time: {trial_time:.2f}s")
                
                # Update best
                if score > best_score:
                    best_score = score
                    best_params = hyperparams.copy()
                    logger.info(f"New best score: {best_score:.4f}")
                
            except Exception as e:
                logger.error(f"Trial {trial_idx + 1} failed: {e}")
                trial_result = {
                    'trial': trial_idx + 1,
                    'hyperparams': hyperparams,
                    'score': 0.0,
                    'error': str(e)
                }
                self.trial_results.append(trial_result)
        
        # Prepare results
        results = {
            'model_type': self.model_type,
            'framework': self.framework,
            'search_type': self.search_type,
            'max_trials': self.max_trials,
            'best_score': best_score,
            'best_hyperparams': best_params,
            'all_trials': self.trial_results,
            'search_space': self.search_space
        }
        
        logger.info(f"Hyperparameter search completed!")
        logger.info(f"Best score: {best_score:.4f}")
        logger.info(f"Best hyperparams: {best_params}")
        
        return results

def main():
    """
    Main function for hyperparameter tuning.
    """
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for sentiment analysis models')
    
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['rnn', 'lstm', 'gru', 'cnn1d', 'bilstm', 'transformer'],
                       help='Type of model to tune')
    parser.add_argument('--framework', type=str, default='tensorflow',
                       choices=['tensorflow', 'pytorch'],
                       help='Framework to use')
    parser.add_argument('--search_type', type=str, default='random',
                       choices=['random', 'grid'],
                       help='Type of hyperparameter search')
    parser.add_argument('--max_trials', type=int, default=20,
                       help='Maximum number of trials')
    parser.add_argument('--output_dir', type=str, default='hyperparameter_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize data manager
    data_manager = DataLoaderManager()
    
    # Create tuner
    tuner = HyperparameterTuner(
        args.model_type, args.framework, data_manager,
        args.search_type, args.max_trials
    )
    
    # Run hyperparameter search
    results = tuner.run_hyperparameter_search()
    
    # Save results
    results_file = os.path.join(
        args.output_dir, 
        f'{args.model_type}_{args.framework}_hyperparameter_results.json'
    )
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {results_file}")
    
    # Print summary
    print("\n" + "="*50)
    print("HYPERPARAMETER TUNING SUMMARY")
    print("="*50)
    print(f"Model: {args.model_type}")
    print(f"Framework: {args.framework}")
    print(f"Search type: {args.search_type}")
    print(f"Trials completed: {len(results['all_trials'])}")
    print(f"Best validation F1 score: {results['best_score']:.4f}")
    print(f"Best hyperparameters:")
    for param, value in results['best_hyperparams'].items():
        print(f"  {param}: {value}")

if __name__ == "__main__":
    main()