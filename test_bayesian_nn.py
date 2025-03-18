"""
Test script for bayesian_nn.py

This script runs a simplified version of the experiment to verify that all components
work correctly. We use only a few epochs to make the test run faster.
"""

import os
import torch
from bayesian_nn import ExperimentManager

# Create output directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create a test experiment manager
test_manager = ExperimentManager()

# Load MNIST dataset
test_manager.load_mnist_dataset(batch_size=64)

# Train baseline CNN for a few epochs
print("\n=== Training Baseline CNN ===")
test_manager.train_baseline_cnn(num_epochs=2, lr=0.001)

# Train MC Dropout CNN for a few epochs
print("\n=== Training MC Dropout CNN ===")
test_manager.train_mc_dropout(num_epochs=2, lr=0.001)

# Train BNN with VI for a few epochs
print("\n=== Training Bayesian Neural Network with VI ===")
test_manager.train_bnn_with_vi(num_epochs=2, lr=0.001)

# Evaluate all models
print("\n=== Evaluating Models ===")
test_manager.evaluate_models()

# Visualize uncertainty
print("\n=== Visualizing Uncertainty ===")
test_manager.visualize_uncertainty()

print("\nTest completed! Check the 'results' directory for outputs.") 