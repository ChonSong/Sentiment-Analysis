#!/usr/bin/env python3
"""
Model Evaluation Script for Sentiment and Emotion Analysis
This script implements comprehensive evaluation of trained models.
"""

import argparse
import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd

import tensorflow as tf
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder

from baseline_models import ModelFactory as BaselineModelFactory
from advanced_models import AdvancedModelFactory
from data_loaders import DataLoaderManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Comprehensive model evaluation class.
    """
    
    def __init__(self, data_manager: DataLoaderManager):
        """
        Initialize the evaluator.
        
        Args:
            data_manager: Data loader manager
        """
        self.data_manager = data_manager
        self.vocab_info = data_manager.get_vocab_info()
        self.label_info = data_manager.get_label_info()
        
        # Load label encoders for interpretation
        with open('processed_data/preprocessing_artifacts.pkl', 'rb') as f:
            import pickle
            artifacts = pickle.load(f)
            self.sentiment_encoder = artifacts['sentiment_encoder']
            self.emotion_encoder = artifacts['emotion_encoder']
    
    def load_tensorflow_model(self, model_type: str, model_path: str) -> tf.keras.Model:
        """
        Load a trained TensorFlow model.
        
        Args:
            model_type: Type of model
            model_path: Path to model file
            
        Returns:
            Loaded TensorFlow model
        """
        # Create model architecture
        if model_type in ['bilstm', 'transformer']:
            if model_type == 'transformer':
                model = AdvancedModelFactory.create_tensorflow_model(
                    model_type,
                    num_sentiment_classes=self.label_info['num_sentiment_classes'],
                    num_emotion_classes=self.label_info['num_emotion_classes'],
                    max_length=self.vocab_info['max_sequence_length']
                )
            else:
                model = AdvancedModelFactory.create_tensorflow_model(
                    model_type, self.vocab_info['vocab_size'],
                    self.label_info['num_sentiment_classes'],
                    self.label_info['num_emotion_classes'],
                    self.vocab_info['max_sequence_length']
                )
        else:
            model = BaselineModelFactory.create_tensorflow_model(
                model_type, self.vocab_info['vocab_size'],
                self.label_info['num_sentiment_classes'],
                self.label_info['num_emotion_classes'],
                self.vocab_info['max_sequence_length']
            )
        
        # Load weights
        model.load_weights(model_path)
        
        logger.info(f"Loaded TensorFlow {model_type} model from {model_path}")
        return model
    
    def load_pytorch_model(self, model_type: str, model_path: str) -> nn.Module:
        """
        Load a trained PyTorch model.
        
        Args:
            model_type: Type of model
            model_path: Path to model file
            
        Returns:
            Loaded PyTorch model
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create model architecture
        if model_type in ['bilstm', 'transformer']:
            if model_type == 'transformer':
                model = AdvancedModelFactory.create_pytorch_model(
                    model_type,
                    num_sentiment_classes=self.label_info['num_sentiment_classes'],
                    num_emotion_classes=self.label_info['num_emotion_classes']
                )
            else:
                model = AdvancedModelFactory.create_pytorch_model(
                    model_type, self.vocab_info['vocab_size'],
                    self.label_info['num_sentiment_classes'],
                    self.label_info['num_emotion_classes']
                )
        else:
            model = BaselineModelFactory.create_pytorch_model(
                model_type, self.vocab_info['vocab_size'],
                self.label_info['num_sentiment_classes'],
                self.label_info['num_emotion_classes']
            )
        
        # Load weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        logger.info(f"Loaded PyTorch {model_type} model from {model_path}")
        return model
    
    def evaluate_tensorflow_model(self, model: tf.keras.Model, 
                                 test_dataset) -> Dict[str, Any]:
        """
        Evaluate a TensorFlow model.
        
        Args:
            model: TensorFlow model
            test_dataset: Test dataset
            
        Returns:
            Evaluation results
        """
        logger.info("Evaluating TensorFlow model...")
        
        # Get predictions
        predictions = model.predict(test_dataset, verbose=0)
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
        results = self._calculate_metrics(
            y_sentiment_true, y_sentiment_pred,
            y_emotion_true, y_emotion_pred,
            sentiment_probs, emotion_probs
        )
        
        return results
    
    def evaluate_pytorch_model(self, model: nn.Module, 
                             test_loader) -> Dict[str, Any]:
        """
        Evaluate a PyTorch model.
        
        Args:
            model: PyTorch model
            test_loader: Test data loader
            
        Returns:
            Evaluation results
        """
        logger.info("Evaluating PyTorch model...")
        
        device = next(model.parameters()).device
        
        y_sentiment_true = []
        y_emotion_true = []
        sentiment_probs_list = []
        emotion_probs_list = []
        
        with torch.no_grad():
            for batch in test_loader:
                sequences = batch['sequences'].to(device)
                sentiment_labels = batch['sentiment_labels'].to(device)
                emotion_labels = batch['emotion_labels'].to(device)
                
                # Get model predictions
                if hasattr(model, 'transformer') and model.transformer is not None:
                    attention_mask = torch.ones_like(sequences)
                    sentiment_logits, emotion_logits = model(sequences, attention_mask=attention_mask)
                else:
                    sentiment_logits, emotion_logits = model(sequences)
                
                # Convert to probabilities
                sentiment_probs = torch.softmax(sentiment_logits, dim=1)
                emotion_probs = torch.softmax(emotion_logits, dim=1)
                
                # Store results
                y_sentiment_true.extend(sentiment_labels.cpu().numpy())
                y_emotion_true.extend(emotion_labels.cpu().numpy())
                sentiment_probs_list.append(sentiment_probs.cpu().numpy())
                emotion_probs_list.append(emotion_probs.cpu().numpy())
        
        # Concatenate probabilities
        sentiment_probs = np.vstack(sentiment_probs_list)
        emotion_probs = np.vstack(emotion_probs_list)
        
        # Get predicted labels
        y_sentiment_pred = np.argmax(sentiment_probs, axis=1)
        y_emotion_pred = np.argmax(emotion_probs, axis=1)
        
        # Calculate metrics
        results = self._calculate_metrics(
            np.array(y_sentiment_true), y_sentiment_pred,
            np.array(y_emotion_true), y_emotion_pred,
            sentiment_probs, emotion_probs
        )
        
        return results
    
    def _calculate_metrics(self, y_sentiment_true: np.ndarray, y_sentiment_pred: np.ndarray,
                          y_emotion_true: np.ndarray, y_emotion_pred: np.ndarray,
                          sentiment_probs: np.ndarray, emotion_probs: np.ndarray) -> Dict[str, Any]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_sentiment_true: True sentiment labels
            y_sentiment_pred: Predicted sentiment labels
            y_emotion_true: True emotion labels
            y_emotion_pred: Predicted emotion labels
            sentiment_probs: Sentiment probabilities
            emotion_probs: Emotion probabilities
            
        Returns:
            Dictionary of metrics
        """
        # Basic metrics
        sentiment_accuracy = accuracy_score(y_sentiment_true, y_sentiment_pred)
        emotion_accuracy = accuracy_score(y_emotion_true, y_emotion_pred)
        
        sentiment_f1_macro = f1_score(y_sentiment_true, y_sentiment_pred, average='macro')
        emotion_f1_macro = f1_score(y_emotion_true, y_emotion_pred, average='macro')
        
        sentiment_f1_weighted = f1_score(y_sentiment_true, y_sentiment_pred, average='weighted')
        emotion_f1_weighted = f1_score(y_emotion_true, y_emotion_pred, average='weighted')
        
        sentiment_precision = precision_score(y_sentiment_true, y_sentiment_pred, average='macro')
        emotion_precision = precision_score(y_emotion_true, y_emotion_pred, average='macro')
        
        sentiment_recall = recall_score(y_sentiment_true, y_sentiment_pred, average='macro')
        emotion_recall = recall_score(y_emotion_true, y_emotion_pred, average='macro')
        
        # Confusion matrices
        sentiment_cm = confusion_matrix(y_sentiment_true, y_sentiment_pred)
        emotion_cm = confusion_matrix(y_emotion_true, y_emotion_pred)
        
        # Classification reports
        sentiment_report = classification_report(
            y_sentiment_true, y_sentiment_pred,
            target_names=self.sentiment_encoder.classes_,
            output_dict=True
        )
        
        emotion_report = classification_report(
            y_emotion_true, y_emotion_pred,
            target_names=self.emotion_encoder.classes_,
            output_dict=True
        )
        
        # Confidence scores
        sentiment_confidence = np.max(sentiment_probs, axis=1)
        emotion_confidence = np.max(emotion_probs, axis=1)
        
        results = {
            'sentiment_metrics': {
                'accuracy': sentiment_accuracy,
                'f1_macro': sentiment_f1_macro,
                'f1_weighted': sentiment_f1_weighted,
                'precision_macro': sentiment_precision,
                'recall_macro': sentiment_recall,
                'confusion_matrix': sentiment_cm.tolist(),
                'classification_report': sentiment_report,
                'mean_confidence': np.mean(sentiment_confidence),
                'std_confidence': np.std(sentiment_confidence)
            },
            'emotion_metrics': {
                'accuracy': emotion_accuracy,
                'f1_macro': emotion_f1_macro,
                'f1_weighted': emotion_f1_weighted,
                'precision_macro': emotion_precision,
                'recall_macro': emotion_recall,
                'confusion_matrix': emotion_cm.tolist(),
                'classification_report': emotion_report,
                'mean_confidence': np.mean(emotion_confidence),
                'std_confidence': np.std(emotion_confidence)
            },
            'overall_metrics': {
                'average_accuracy': (sentiment_accuracy + emotion_accuracy) / 2,
                'average_f1_macro': (sentiment_f1_macro + emotion_f1_macro) / 2,
                'average_f1_weighted': (sentiment_f1_weighted + emotion_f1_weighted) / 2
            },
            'predictions': {
                'y_sentiment_true': y_sentiment_true.tolist(),
                'y_sentiment_pred': y_sentiment_pred.tolist(),
                'y_emotion_true': y_emotion_true.tolist(),
                'y_emotion_pred': y_emotion_pred.tolist(),
                'sentiment_probs': sentiment_probs.tolist(),
                'emotion_probs': emotion_probs.tolist()
            }
        }
        
        return results
    
    def create_confusion_matrix_plots(self, results: Dict[str, Any], 
                                    model_name: str, output_dir: str):
        """
        Create and save confusion matrix plots.
        
        Args:
            results: Evaluation results
            model_name: Name of the model
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Sentiment confusion matrix
        plt.figure(figsize=(8, 6))
        sentiment_cm = np.array(results['sentiment_metrics']['confusion_matrix'])
        sns.heatmap(
            sentiment_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.sentiment_encoder.classes_,
            yticklabels=self.sentiment_encoder.classes_
        )
        plt.title(f'{model_name} - Sentiment Classification Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_sentiment_confusion_matrix.png'))
        plt.close()
        
        # Emotion confusion matrix
        plt.figure(figsize=(10, 8))
        emotion_cm = np.array(results['emotion_metrics']['confusion_matrix'])
        sns.heatmap(
            emotion_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.emotion_encoder.classes_,
            yticklabels=self.emotion_encoder.classes_
        )
        plt.title(f'{model_name} - Emotion Classification Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_emotion_confusion_matrix.png'))
        plt.close()
        
        logger.info(f"Confusion matrix plots saved to {output_dir}")
    
    def compare_models(self, model_results: Dict[str, Dict[str, Any]], 
                      output_dir: str):
        """
        Compare multiple models and create comparison plots.
        
        Args:
            model_results: Dictionary of model results
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data for comparison
        model_names = list(model_results.keys())
        
        sentiment_accuracy = [model_results[name]['sentiment_metrics']['accuracy'] for name in model_names]
        emotion_accuracy = [model_results[name]['emotion_metrics']['accuracy'] for name in model_names]
        sentiment_f1 = [model_results[name]['sentiment_metrics']['f1_macro'] for name in model_names]
        emotion_f1 = [model_results[name]['emotion_metrics']['f1_macro'] for name in model_names]
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy comparison
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, sentiment_accuracy, width, label='Sentiment', alpha=0.8)
        axes[0, 0].bar(x + width/2, emotion_accuracy, width, label='Emotion', alpha=0.8)
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(model_names, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # F1 Score comparison
        axes[0, 1].bar(x - width/2, sentiment_f1, width, label='Sentiment', alpha=0.8)
        axes[0, 1].bar(x + width/2, emotion_f1, width, label='Emotion', alpha=0.8)
        axes[0, 1].set_title('Model F1 Score Comparison')
        axes[0, 1].set_ylabel('F1 Score (Macro)')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(model_names, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Overall performance scatter plot
        overall_accuracy = [(s + e) / 2 for s, e in zip(sentiment_accuracy, emotion_accuracy)]
        overall_f1 = [(s + e) / 2 for s, e in zip(sentiment_f1, emotion_f1)]
        
        axes[1, 0].scatter(overall_accuracy, overall_f1, s=100, alpha=0.7)
        for i, name in enumerate(model_names):
            axes[1, 0].annotate(name, (overall_accuracy[i], overall_f1[i]), 
                              xytext=(5, 5), textcoords='offset points')
        axes[1, 0].set_title('Overall Performance (Accuracy vs F1)')
        axes[1, 0].set_xlabel('Average Accuracy')
        axes[1, 0].set_ylabel('Average F1 Score')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Model complexity (parameter count) if available
        # This would require storing parameter counts during evaluation
        axes[1, 1].text(0.5, 0.5, 'Model Complexity\nComparison\n(requires parameter counts)', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Model Complexity')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create summary table
        summary_data = []
        for name in model_names:
            summary_data.append({
                'Model': name,
                'Sentiment Accuracy': f"{model_results[name]['sentiment_metrics']['accuracy']:.4f}",
                'Emotion Accuracy': f"{model_results[name]['emotion_metrics']['accuracy']:.4f}",
                'Sentiment F1': f"{model_results[name]['sentiment_metrics']['f1_macro']:.4f}",
                'Emotion F1': f"{model_results[name]['emotion_metrics']['f1_macro']:.4f}",
                'Overall F1': f"{model_results[name]['overall_metrics']['average_f1_macro']:.4f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(output_dir, 'model_comparison_summary.csv'), index=False)
        
        logger.info(f"Model comparison plots and summary saved to {output_dir}")

def evaluate_models_from_directory(models_dir: str, results_dir: str):
    """
    Evaluate all trained models in a directory.
    
    Args:
        models_dir: Directory containing trained models
        results_dir: Directory to save evaluation results
    """
    logger.info(f"Evaluating models from {models_dir}")
    
    # Initialize evaluator
    data_manager = DataLoaderManager()
    evaluator = ModelEvaluator(data_manager)
    
    # Create datasets
    try:
        train_tf, val_tf, test_tf = data_manager.create_tensorflow_datasets(batch_size=32)
        train_torch, val_torch, test_torch = data_manager.create_pytorch_dataloaders(batch_size=32)
    except Exception as e:
        logger.error(f"Failed to create datasets: {e}")
        return
    
    # Find model files
    model_files = list(Path(models_dir).glob('*'))
    
    all_results = {}
    
    for model_file in model_files:
        if model_file.suffix in ['.h5', '.pth']:
            model_name = model_file.stem
            
            # Parse model type and framework from filename
            if '_best' in model_name:
                base_name = model_name.replace('_best', '')
                
                # Determine framework from file extension
                framework = 'tensorflow' if model_file.suffix == '.h5' else 'pytorch'
                model_type = base_name
                
                try:
                    logger.info(f"Evaluating {model_name} ({framework} {model_type})")
                    
                    if framework == 'tensorflow':
                        model = evaluator.load_tensorflow_model(model_type, str(model_file))
                        results = evaluator.evaluate_tensorflow_model(model, test_tf)
                    else:
                        model = evaluator.load_pytorch_model(model_type, str(model_file))
                        results = evaluator.evaluate_pytorch_model(model, test_torch)
                    
                    # Save individual results
                    results_file = os.path.join(results_dir, f'{model_name}_evaluation.json')
                    os.makedirs(results_dir, exist_ok=True)
                    
                    with open(results_file, 'w') as f:
                        json.dump(results, f, indent=2, default=str)
                    
                    # Create confusion matrix plots
                    evaluator.create_confusion_matrix_plots(results, model_name, 
                                                           os.path.join(results_dir, 'plots'))
                    
                    all_results[model_name] = results
                    
                    logger.info(f"Evaluation complete for {model_name}")
                    
                except Exception as e:
                    logger.error(f"Failed to evaluate {model_name}: {e}")
    
    # Compare all models
    if len(all_results) > 1:
        evaluator.compare_models(all_results, os.path.join(results_dir, 'comparison'))
        
        # Save combined results
        with open(os.path.join(results_dir, 'all_model_results.json'), 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
    
    logger.info(f"Evaluation complete. Results saved to {results_dir}")

def main():
    """
    Main function for model evaluation.
    """
    parser = argparse.ArgumentParser(description='Evaluate trained sentiment analysis models')
    
    parser.add_argument('--models_dir', type=str, default='models',
                       help='Directory containing trained models')
    parser.add_argument('--results_dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    evaluate_models_from_directory(args.models_dir, args.results_dir)

if __name__ == "__main__":
    main()