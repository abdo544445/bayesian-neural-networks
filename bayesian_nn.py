"""
Bayesian Neural Networks for Uncertainty Quantification

This script implements Bayesian Neural Networks for uncertainty quantification
in high-stakes applications, following the approach described in plan2.md.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

# For Bayesian Neural Networks
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

# For evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.calibration import calibration_curve

# For progress tracking
from tqdm import tqdm

# Import our model definitions
from models import BaselineCNN, MCDropoutCNN, BayesianCNN

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
pyro.set_rng_seed(RANDOM_SEED)

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ExperimentManager:
    """
    Main class to manage all experiments including:
    - Dataset loading
    - Model training
    - Uncertainty quantification
    - Evaluation
    """
    def __init__(self):
        self.data_loaders = {}
        self.models = {}
        self.results = {}
        
        # Create directories if they don't exist
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)
    
    def load_mnist_dataset(self, batch_size=128):
        """Load MNIST dataset for initial testing"""
        print("Loading MNIST dataset...")
        
        # Define transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
        ])
        
        # Download and load the training set
        train_dataset = torchvision.datasets.MNIST(
            root='./data', 
            train=True, 
            download=True, 
            transform=transform
        )
        
        # Split training set into training and validation
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(
            train_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(RANDOM_SEED)
        )
        
        # Download and load the test set
        test_dataset = torchvision.datasets.MNIST(
            root='./data', 
            train=False, 
            download=True, 
            transform=transform
        )
        
        # Create data loaders
        self.data_loaders['mnist'] = {
            'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2),
            'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2),
            'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        }
        
        print(f"MNIST dataset loaded. Training samples: {len(train_dataset)}, "
              f"Validation samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
        
        # Load FashionMNIST for out-of-distribution testing
        fashion_test_dataset = torchvision.datasets.FashionMNIST(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )
        
        self.data_loaders['fashion_mnist'] = {
            'test': DataLoader(fashion_test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        }
        
        print(f"FashionMNIST dataset loaded for OOD testing. Samples: {len(fashion_test_dataset)}")
        
        # Visualize a few examples
        self._visualize_mnist_examples(train_dataset)
        
        return self.data_loaders['mnist']
    
    def _visualize_mnist_examples(self, dataset, num_examples=5):
        """Visualize a few examples from the MNIST dataset"""
        # Create a figure with subplots
        fig, axes = plt.subplots(1, num_examples, figsize=(12, 3))
        
        # Get random indices
        indices = np.random.choice(len(dataset), num_examples, replace=False)
        
        # Plot each example
        for i, idx in enumerate(indices):
            img, label = dataset[idx]
            img = img.squeeze().numpy()  # Remove channel dimension and convert to numpy
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f"Label: {label}")
            axes[i].axis('off')
        
        plt.tight_layout()
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/mnist_examples.png')
        plt.close()
        
    def load_medical_dataset(self):
        """Load medical dataset (e.g., Pneumonia) for main experiment"""
        print("Loading medical dataset...")
        # This will be implemented in a later stage
        pass
        
    def train_baseline_cnn(self, dataset_name='mnist', num_epochs=10, lr=0.001, save_model=True):
        """Train a baseline CNN model without uncertainty quantification"""
        print(f"Training baseline CNN on {dataset_name} dataset...")
        
        # Check if dataset is loaded
        if dataset_name not in self.data_loaders:
            raise ValueError(f"Dataset {dataset_name} not loaded. Call load_{dataset_name}_dataset() first.")
        
        # Get data loaders
        data_loaders = self.data_loaders[dataset_name]
        
        # Create model
        input_channels = 1  # MNIST has 1 channel
        num_classes = 10    # MNIST has 10 classes
        model = BaselineCNN(input_channels=input_channels, num_classes=num_classes).to(device)
        
        # Define loss function and optimizer
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        # Training loop
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, labels in tqdm(data_loaders['train'], desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Calculate training statistics
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Calculate average training loss and accuracy
            train_loss = train_loss / len(data_loaders['train'].dataset)
            train_acc = train_correct / train_total
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in tqdm(data_loaders['val'], desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # Calculate validation statistics
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # Calculate average validation loss and accuracy
            val_loss = val_loss / len(data_loaders['val'].dataset)
            val_acc = val_correct / val_total
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Print statistics
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_model:
                    os.makedirs('models', exist_ok=True)
                    torch.save(model.state_dict(), f'models/baseline_cnn_{dataset_name}.pth')
                    print(f"Saved best model at epoch {epoch+1}")
        
        # Save training history
        self.models['baseline_cnn'] = model
        self.results['baseline_cnn'] = {
            'history': history,
            'dataset': dataset_name
        }
        
        # Visualize training history
        self._visualize_training_history('baseline_cnn', history)
        
        # Evaluate on test set
        test_metrics = self.evaluate_model('baseline_cnn', dataset_name, 'test')
        
        return model, history, test_metrics
    
    def _visualize_training_history(self, model_name, history):
        """Visualize training history"""
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot loss
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'{model_name} - Loss')
        ax1.legend()
        
        # Plot accuracy
        ax2.plot(history['train_acc'], label='Train Accuracy')
        ax2.plot(history['val_acc'], label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title(f'{model_name} - Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        os.makedirs('results', exist_ok=True)
        plt.savefig(f'results/{model_name}_training_history.png')
        plt.close()
    
    def evaluate_model(self, model_name, dataset_name, split='test'):
        """Evaluate model on a specific dataset split"""
        print(f"Evaluating {model_name} on {dataset_name} {split} set...")
        
        # Check if model exists
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained. Call train_{model_name}() first.")
        
        # Check if dataset is loaded
        if dataset_name not in self.data_loaders:
            raise ValueError(f"Dataset {dataset_name} not loaded. Call load_{dataset_name}_dataset() first.")
        
        # Get model and data loader
        model = self.models[model_name]
        data_loader = self.data_loaders[dataset_name][split]
        
        # Set model to evaluation mode
        model.eval()
        
        # Evaluation variables
        all_labels = []
        all_predictions = []
        all_probabilities = []
        
        # Evaluate model
        with torch.no_grad():
            for inputs, labels in tqdm(data_loader, desc=f"Evaluating {model_name} on {dataset_name} {split}"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Get predictions
                if model_name == 'baseline_cnn':
                    # Single deterministic prediction
                    log_probs = model(inputs)
                    probs = torch.exp(log_probs)
                    _, predicted = torch.max(log_probs, 1)
                elif model_name == 'mc_dropout':
                    # Multiple predictions with dropout for uncertainty
                    mean_probs, _ = model.predict_with_uncertainty(inputs)
                    _, predicted = torch.max(mean_probs, 1)
                    probs = mean_probs
                else:
                    raise ValueError(f"Evaluation not implemented for {model_name}")
                
                # Store results
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probs.cpu().numpy())
        
        # Convert to numpy arrays
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(all_labels, all_predictions),
            'precision': precision_score(all_labels, all_predictions, average='macro'),
            'recall': recall_score(all_labels, all_predictions, average='macro'),
            'f1': f1_score(all_labels, all_predictions, average='macro'),
            'confusion_matrix': confusion_matrix(all_labels, all_predictions)
        }
        
        # Print metrics
        print(f"Results for {model_name} on {dataset_name} {split} set:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        
        # Visualize confusion matrix
        self._visualize_confusion_matrix(
            metrics['confusion_matrix'], 
            title=f"{model_name} - {dataset_name} {split} set",
            save_path=f"results/{model_name}_{dataset_name}_{split}_confusion_matrix.png"
        )
        
        # Store metrics in results
        if 'evaluation' not in self.results[model_name]:
            self.results[model_name]['evaluation'] = {}
        
        self.results[model_name]['evaluation'][f"{dataset_name}_{split}"] = {
            'metrics': metrics,
            'labels': all_labels,
            'predictions': all_predictions,
            'probabilities': all_probabilities
        }
        
        return metrics
    
    def _visualize_confusion_matrix(self, cm, title='Confusion Matrix', save_path=None):
        """Visualize confusion matrix"""
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        
        # Add labels and ticks
        classes = [str(i) for i in range(cm.shape[0])]
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        # Save figure
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
    
    def train_mc_dropout(self, dataset_name='mnist', num_epochs=10, lr=0.001, save_model=True):
        """Train a model using Monte Carlo Dropout for uncertainty estimation"""
        print(f"Training MC Dropout model on {dataset_name} dataset...")
        
        # Check if dataset is loaded
        if dataset_name not in self.data_loaders:
            raise ValueError(f"Dataset {dataset_name} not loaded. Call load_{dataset_name}_dataset() first.")
        
        # Get data loaders
        data_loaders = self.data_loaders[dataset_name]
        
        # Create model using MCDropoutCNN
        input_channels = 1  # MNIST has 1 channel
        num_classes = 10    # MNIST has 10 classes
        model = MCDropoutCNN(input_channels=input_channels, num_classes=num_classes, dropout_rate=0.5).to(device)
        
        # Define loss function and optimizer
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, labels in tqdm(data_loaders['train'], desc=f"Epoch {epoch+1}/{num_epochs} - Training (MC Dropout)"):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_loss /= len(data_loaders['train'].dataset)
            train_acc = train_correct / train_total
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, labels in tqdm(data_loaders['val'], desc=f"Epoch {epoch+1}/{num_epochs} - Validation (MC Dropout)"):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_loss /= len(data_loaders['val'].dataset)
            val_acc = val_correct / val_total
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_model:
                    os.makedirs('models', exist_ok=True)
                    torch.save(model.state_dict(), f'models/mc_dropout_{dataset_name}.pth')
                    print(f"Saved best MC Dropout model at epoch {epoch+1}")
        
        # Save training history
        self.models['mc_dropout'] = model
        self.results['mc_dropout'] = {
            'history': history,
            'dataset': dataset_name
        }
        
        # Visualize training history
        self._visualize_training_history('mc_dropout', history)
        
        # Evaluate on test set
        test_metrics = self.evaluate_model('mc_dropout', dataset_name, 'test')
        
        return model, history, test_metrics
        
    def train_bnn_with_vi(self, dataset_name='mnist', num_epochs=10, lr=0.001, save_model=True):
        """Train a full Bayesian Neural Network using Variational Inference"""
        print(f"Training BNN with Variational Inference on {dataset_name} dataset...")
        
        # Check if dataset is loaded
        if dataset_name not in self.data_loaders:
            raise ValueError(f"Dataset {dataset_name} not loaded. Call load_{dataset_name}_dataset() first.")
        
        # Get data loaders
        data_loaders = self.data_loaders[dataset_name]
        
        # Create Bayesian model
        bayes_model = BayesianCNN(input_channels=1, num_classes=10, hidden_size=64).to(device)
        
        # Create an AutoNormal guide
        from pyro.infer.autoguide import AutoNormal
        guide = AutoNormal(bayes_model)
        
        # Clear Pyro's parameter store
        pyro.clear_param_store()
        
        # Setup optimizer and SVI
        optimizer = pyro.optim.Adam({'lr': lr})
        svi = SVI(bayes_model, guide, optimizer, loss=Trace_ELBO())
        
        # Training history
        history = { 'train_loss': [], 'val_loss': [] }
        best_val_loss = float('inf')
        
        num_train = len(data_loaders['train'].dataset)
        num_val = len(data_loaders['val'].dataset)
        
        for epoch in range(num_epochs):
            bayes_model.train()
            epoch_loss = 0.0
            for inputs, labels in tqdm(data_loaders['train'], desc=f"Epoch {epoch+1}/{num_epochs} - Training (BNN VI)"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                loss = svi.step(inputs, labels)
                epoch_loss += loss
            avg_train_loss = epoch_loss / num_train
            history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            bayes_model.eval()
            val_loss = 0.0
            for inputs, labels in tqdm(data_loaders['val'], desc=f"Epoch {epoch+1}/{num_epochs} - Validation (BNN VI)"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                loss = svi.evaluate_loss(inputs, labels)
                val_loss += loss
            avg_val_loss = val_loss / num_val
            history['val_loss'].append(avg_val_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                if save_model:
                    os.makedirs('models', exist_ok=True)
                    torch.save(bayes_model.state_dict(), f'models/bnn_vi_{dataset_name}.pth')
                    print(f"Saved best BNN VI model at epoch {epoch+1}")
        
        # Save model and guide
        self.models['bnn_vi'] = (bayes_model, guide)
        self.results['bnn_vi'] = { 'history': history, 'dataset': dataset_name }
        
        # Evaluate on test set using Predictive
        from pyro.infer import Predictive
        bayes_model.eval()
        predictive = Predictive(bayes_model, guide=guide, num_samples=100, return_sites=['_RETURN'])
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in tqdm(data_loaders['test'], desc="Evaluating BNN VI on test set"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                samples = predictive(inputs, y=None)
                # Extract logits from the '_RETURN' key
                logits_samples = samples['_RETURN']
                # Average logits over the num_samples dimension
                logits_mean = torch.mean(logits_samples, dim=0)
                _, predicted = torch.max(logits_mean, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        import numpy as np
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='macro'),
            'recall': recall_score(all_labels, all_preds, average='macro'),
            'f1': f1_score(all_labels, all_preds, average='macro'),
            'confusion_matrix': confusion_matrix(all_labels, all_preds)
        }
        
        # Store evaluation metrics
        if 'evaluation' not in self.results['bnn_vi']:
            self.results['bnn_vi']['evaluation'] = {}
        self.results['bnn_vi']['evaluation'] = metrics
        
        print("BNN VI Test Metrics:", metrics)
        
        return bayes_model, guide, history, metrics
        
    def evaluate_models(self):
        """Evaluate all trained models and create comparison bar charts for key metrics."""
        print("Aggregating evaluation metrics for all models...")
        # Collect evaluation metrics from results
        metrics_dict = {}
        for model_name, result in self.results.items():
            if 'evaluation' in result and result['evaluation'] is not None:
                # For baseline_cnn and mc_dropout, evaluation is stored as a dict keyed by dataset_split
                if isinstance(result['evaluation'], dict) and 'metrics' in result['evaluation']:
                    metrics_dict[model_name] = result['evaluation']['metrics']
                else:
                    # For models like bnn_vi where evaluation was stored directly
                    metrics_dict[model_name] = result['evaluation']

        if not metrics_dict:
            print("No evaluation metrics found.")
            return
        
        # Extract metrics for comparison
        models = []
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []

        for model_name, metrics in metrics_dict.items():
            models.append(model_name)
            accuracies.append(metrics.get('accuracy', 0))
            precisions.append(metrics.get('precision', 0))
            recalls.append(metrics.get('recall', 0))
            f1_scores.append(metrics.get('f1', 0))

        # Create bar charts for each metric
        import matplotlib.pyplot as plt
        import numpy as np
        x = np.arange(len(models))
        width = 0.6

        # Accuracy
        plt.figure(figsize=(8, 6))
        plt.bar(x, accuracies, width, color='skyblue')
        plt.xticks(x, models)
        plt.ylabel('Accuracy')
        plt.title('Model Comparison: Accuracy')
        plt.ylim(0, 1)
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/model_comparison_accuracy.png')
        plt.close()

        # Precision
        plt.figure(figsize=(8, 6))
        plt.bar(x, precisions, width, color='lightgreen')
        plt.xticks(x, models)
        plt.ylabel('Precision')
        plt.title('Model Comparison: Precision')
        plt.ylim(0, 1)
        plt.savefig('results/model_comparison_precision.png')
        plt.close()

        # Recall
        plt.figure(figsize=(8, 6))
        plt.bar(x, recalls, width, color='salmon')
        plt.xticks(x, models)
        plt.ylabel('Recall')
        plt.title('Model Comparison: Recall')
        plt.ylim(0, 1)
        plt.savefig('results/model_comparison_recall.png')
        plt.close()

        # F1 Score
        plt.figure(figsize=(8, 6))
        plt.bar(x, f1_scores, width, color='plum')
        plt.xticks(x, models)
        plt.ylabel('F1 Score')
        plt.title('Model Comparison: F1 Score')
        plt.ylim(0, 1)
        plt.savefig('results/model_comparison_f1.png')
        plt.close()

        print("Evaluation comparison plots saved in results/ directory.")
        
    def visualize_uncertainty(self):
        """Visualize uncertainty estimates for MC Dropout and BNN VI models using a test batch."""
        print("Visualizing uncertainty estimates...")
        # Use MNIST test set for visualization
        if 'mnist' not in self.data_loaders or 'test' not in self.data_loaders['mnist']:
            print("MNIST test set not loaded.")
            return
        test_loader = self.data_loaders['mnist']['test']
        
        # Get one batch
        batch = next(iter(test_loader))
        inputs, labels = batch
        inputs = inputs.to(device)
        
        # For MC Dropout model
        if 'mc_dropout' in self.models:
            model = self.models['mc_dropout']
            model.eval()
            # Obtain multiple predictions with dropout enabled
            with torch.no_grad():
                mean_probs, variance = model.predict_with_uncertainty(inputs, num_samples=50)
            # For each sample, compute confidence as the max probability and uncertainty as max variance
            conf_mc = torch.max(mean_probs, dim=1)[0].cpu().numpy()
            uncert_mc = torch.max(variance, dim=1)[0].cpu().numpy()
            
            # Plot scatter of confidence vs uncertainty
            plt.figure(figsize=(8,6))
            plt.scatter(conf_mc, uncert_mc, alpha=0.6, color='blue')
            plt.xlabel('Prediction Confidence (max prob)')
            plt.ylabel('Uncertainty (max variance)')
            plt.title('MC Dropout: Confidence vs Uncertainty')
            os.makedirs('results', exist_ok=True)
            plt.savefig('results/mc_dropout_uncertainty.png')
            plt.close()
            print("MC Dropout uncertainty visualization saved in results/mc_dropout_uncertainty.png")
        else:
            print("MC Dropout model not found; skipping its uncertainty visualization.")
        
        # For BNN VI model
        if 'bnn_vi' in self.models:
            bayes_model, guide = self.models['bnn_vi']
            bayes_model.eval()
            from pyro.infer import Predictive
            predictive = Predictive(bayes_model, guide=guide, num_samples=50)
            with torch.no_grad():
                # Get multiple samples of logits
                samples = predictive(inputs)  # samples is a tensor of shape [num_samples, batch_size, num_classes]
                if isinstance(samples, dict):
                    # In case Predictive returns a dict
                    logits_samples = samples[list(samples.keys())[0]]
                else:
                    logits_samples = samples
                # Convert log probabilities to probabilities
                probs = torch.exp(logits_samples)
                mean_probs = torch.mean(probs, dim=0)
                variance = torch.var(probs, dim=0)
            conf_vi = torch.max(mean_probs, dim=1)[0].cpu().numpy()
            uncert_vi = torch.max(variance, dim=1)[0].cpu().numpy()
            
            plt.figure(figsize=(8,6))
            plt.scatter(conf_vi, uncert_vi, alpha=0.6, color='red')
            plt.xlabel('Prediction Confidence (max prob)')
            plt.ylabel('Uncertainty (max variance)')
            plt.title('BNN VI: Confidence vs Uncertainty')
            plt.savefig('results/bnn_vi_uncertainty.png')
            plt.close()
            print("BNN VI uncertainty visualization saved in results/bnn_vi_uncertainty.png")
        else:
            print("BNN VI model not found; skipping its uncertainty visualization.")

    def run_experiment(self, dataset_name='mnist'):
        """Run the complete experiment pipeline"""
        # Load dataset
        if dataset_name == 'mnist':
            self.load_mnist_dataset()
        elif dataset_name == 'medical':
            self.load_medical_dataset()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Train models
        self.train_baseline_cnn(dataset_name=dataset_name)
        self.train_mc_dropout(dataset_name=dataset_name)
        self.train_bnn_with_vi(dataset_name=dataset_name)
        
        # Evaluate and visualize
        self.evaluate_models()
        self.visualize_uncertainty()
        
if __name__ == "__main__":
    # Create experiment manager
    experiment = ExperimentManager()
    
    # Run experiment with MNIST (initial testing)
    print("Starting experiment with MNIST dataset...")
    experiment.run_experiment(dataset_name='mnist')
    
    # Run experiment with medical dataset (main experiment)
    # Uncomment when ready to run the main experiment
    # print("Starting experiment with medical dataset...")
    # experiment.run_experiment(dataset_name='medical') 