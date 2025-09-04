# Sentiment-Analysis

A comprehensive sentiment and emotion analysis project comparing traditional and modern deep learning architectures on social media data.

## Overview

This project implements and evaluates multiple deep learning architectures for sentiment and emotion analysis, including:

**Baseline Models:**
- Recurrent Neural Network (RNN)
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)
- 1D Convolutional Neural Network (CNN-1D)

**Advanced Models:**
- Bidirectional LSTM (BiLSTM)
- Transformer-based models (DistilBERT)

## Project Structure

```
├── data_access.py              # Data collection and API access
├── preprocessing_pipeline.py   # Text preprocessing and data splitting
├── data_loaders.py            # TensorFlow and PyTorch data loaders
├── baseline_models.py         # Implementation of baseline architectures
├── advanced_models.py         # Implementation of advanced architectures
├── train_baselines.py         # Training script with hyperparameter support
├── hyperparameter_tuning.py   # Systematic hyperparameter optimization
├── evaluate_models.py         # Comprehensive model evaluation
├── error_analysis.md          # Detailed error analysis and insights
├── final_report.tex           # Complete LaTeX research report
├── literature_review.md       # Comprehensive literature review
├── project_plan.md           # Detailed project methodology
├── requirements.txt          # Python dependencies
└── processed_data/           # Preprocessed datasets and artifacts
```

## Key Features

- **Multi-framework Support**: Implementations in both TensorFlow/Keras and PyTorch
- **Comprehensive Preprocessing**: Multilingual text processing with stratified data splitting
- **Systematic Evaluation**: Multiple metrics including accuracy, F1-score, confusion matrices
- **Hyperparameter Tuning**: Automated optimization with random and grid search
- **Visualization**: Confusion matrices and model comparison plots
- **Error Analysis**: Detailed analysis of model failures and challenging content types

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ChonSong/Sentiment-Analysis.git
cd Sentiment-Analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation
```bash
# Generate sample data and run preprocessing
python data_access.py
python preprocessing_pipeline.py
```

### 2. Model Training
```bash
# Train baseline models
python train_baselines.py --model_type lstm --framework pytorch --epochs 10

# Available models: rnn, lstm, gru, cnn1d, bilstm, transformer
# Available frameworks: tensorflow, pytorch
```

### 3. Hyperparameter Tuning
```bash
# Optimize hyperparameters
python hyperparameter_tuning.py --model_type lstm --max_trials 20
```

### 4. Model Evaluation
```bash
# Evaluate all trained models
python evaluate_models.py --models_dir models --results_dir evaluation_results
```

## Results Summary

The project reveals several key insights:

- **Class Imbalance Impact**: High accuracy (95.8%) but poor macro F1 scores (48.9% sentiment, 2.7% emotion)
- **Architecture Comparison**: Similar performance across baseline models due to data limitations
- **Challenging Content**: Sarcasm, multilingual text, and context-dependent sentiment pose significant challenges
- **Data Quality**: Synthetic data limitations highlight the importance of authentic social media datasets

## Key Challenges Identified

1. **Severe class imbalance** affecting model performance evaluation
2. **Limited training data** (247 samples) insufficient for deep learning
3. **Synthetic data artifacts** reducing real-world applicability
4. **Complex linguistic patterns** in social media text (sarcasm, slang, code-switching)

## Technical Achievements

- ✅ Complete implementation of 6 different architectures
- ✅ Multi-framework codebase (TensorFlow + PyTorch)
- ✅ Comprehensive preprocessing pipeline
- ✅ Automated hyperparameter tuning
- ✅ Systematic evaluation framework
- ✅ Detailed error analysis and reporting

## Future Improvements

1. **Real Data Integration**: Use authentic social media datasets
2. **Advanced Preprocessing**: Emoji-aware and context-preserving preprocessing
3. **Ensemble Methods**: Combine multiple models for improved robustness
4. **Multimodal Analysis**: Integrate text with images and metadata
5. **Real-time Adaptation**: Models that evolve with language changes

## Research Contributions

This project provides:
- A systematic comparison framework for sentiment analysis architectures
- Insights into practical challenges of social media text analysis
- Modular, extensible codebase for future research
- Comprehensive documentation and reproducible experiments

## Dependencies

See `requirements.txt` for complete list. Key dependencies:
- TensorFlow >= 2.12.0
- PyTorch >= 2.0.0
- Transformers >= 4.21.0
- Scikit-learn >= 1.1.0
- Pandas >= 1.5.0
- NumPy >= 1.21.0

## License

This project is available for educational and research purposes.

## Citation

If you use this code in your research, please cite:
```
Sentiment Analysis Project: Comparative Analysis of Deep Learning Architectures
for Sentiment and Emotion Analysis on Social Media Data (2024)
```