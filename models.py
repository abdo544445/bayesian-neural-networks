"""
Model architectures for Bayesian Neural Networks

This file contains the model architectures used in the Bayesian Neural Networks
for uncertainty quantification project, including:
1. Baseline CNN (deterministic)
2. Monte Carlo Dropout CNN
3. Bayesian Neural Network with Variational Inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample

class BaselineCNN(nn.Module):
    """
    Standard CNN model with fixed weights (deterministic).
    This will serve as our baseline for comparison.
    
    Architecture:
    - 2 convolutional layers with max pooling
    - 2 fully connected layers
    """
    def __init__(self, input_channels=1, hidden_size=64, num_classes=10):
        super(BaselineCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Compute the size of the feature maps after the convolutional layers
        # For MNIST (28x28 images), after 2 pooling layers, we have 7x7 feature maps
        feature_size = 7 * 7 * 64
        
        # Fully connected layers
        self.fc1 = nn.Linear(feature_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # First convolutional block
        x = self.pool(F.relu(self.conv1(x)))
        
        # Second convolutional block
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten the feature maps
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        # For MNIST (classification), use softmax
        return F.log_softmax(x, dim=1)
    
    def predict(self, x):
        """
        Make a single deterministic prediction
        """
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            return self.forward(x)


class MCDropoutCNN(nn.Module):
    """
    CNN model with Monte Carlo Dropout for uncertainty estimation.
    Dropout is used both during training and inference to approximate 
    Bayesian inference.
    
    Architecture:
    - 2 convolutional layers with max pooling and dropout
    - 2 fully connected layers with dropout
    """
    def __init__(self, input_channels=1, hidden_size=64, num_classes=10, dropout_rate=0.5):
        super(MCDropoutCNN, self).__init__()
        self.dropout_rate = dropout_rate
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Compute the size of the feature maps after the convolutional layers
        # For MNIST (28x28 images), after 2 pooling layers, we have 7x7 feature maps
        feature_size = 7 * 7 * 64
        
        # Fully connected layers
        self.fc1 = nn.Linear(feature_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # First convolutional block with dropout
        x = self.pool(F.relu(self.conv1(x)))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # Second convolutional block with dropout
        x = self.pool(F.relu(self.conv2(x)))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # Flatten the feature maps
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.fc2(x)
        
        # For MNIST (classification), use softmax
        return F.log_softmax(x, dim=1)
    
    def predict_with_uncertainty(self, x, num_samples=50):
        """
        Make multiple predictions with dropout enabled to estimate uncertainty
        
        Args:
            x: Input tensor
            num_samples: Number of Monte Carlo samples
            
        Returns:
            mean: Mean of the predictions (class probabilities)
            variance: Variance of the predictions (uncertainty)
        """
        self.eval()  # Set to evaluation mode
        
        # Enable dropout during inference
        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                pred = self.forward(x)
                # Convert from log probabilities to probabilities
                pred = torch.exp(pred)
                predictions.append(pred)
        
        # Stack and calculate statistics
        predictions = torch.stack(predictions)
        mean = torch.mean(predictions, dim=0)
        variance = torch.var(predictions, dim=0)
        
        return mean, variance


class BayesianCNN(PyroModule):
    """
    Bayesian Neural Network with Variational Inference using Pyro.
    This model uses PyroSample to place priors on its weights and
    defines a forward pass that samples from a Categorical likelihood.
    """
    def __init__(self, input_channels=1, num_classes=10, hidden_size=64):
        super().__init__()
        # Convolutional layer 1
        self.conv1 = PyroModule[nn.Conv2d](in_channels=input_channels,
                                           out_channels=32,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1)
        self.conv1.weight = PyroSample(dist.Normal(0., 1.).expand([32, input_channels, 3, 3]).to_event(4))
        self.conv1.bias = PyroSample(dist.Normal(0., 1.).expand([32]).to_event(1))
        
        # Convolutional layer 2
        self.conv2 = PyroModule[nn.Conv2d](in_channels=32,
                                           out_channels=64,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1)
        self.conv2.weight = PyroSample(dist.Normal(0., 1.).expand([64, 32, 3, 3]).to_event(4))
        self.conv2.bias = PyroSample(dist.Normal(0., 1.).expand([64]).to_event(1))
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        # After 2 pooling layers, for MNIST (28x28), feature maps become 7x7
        self.fc1 = PyroModule[nn.Linear](7 * 7 * 64, hidden_size)
        self.fc1.weight = PyroSample(dist.Normal(0., 1.).expand([hidden_size, 7 * 7 * 64]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0., 1.).expand([hidden_size]).to_event(1))
        
        self.fc2 = PyroModule[nn.Linear](hidden_size, num_classes)
        self.fc2.weight = PyroSample(dist.Normal(0., 1.).expand([num_classes, hidden_size]).to_event(2))
        self.fc2.bias = PyroSample(dist.Normal(0., 1.).expand([num_classes]).to_event(1))

    def forward(self, x, y=None):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        if y is not None:
            with pyro.plate("data", x.shape[0]):
                pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
        return logits

# The BayesianCNN will be implemented in a later update
"""
class BayesianCNN(PyroModule):
    Bayesian Neural Network with Variational Inference
    using Pyro for probabilistic programming.
    
    Will be implemented in a future update.
""" 