# Bayesian Neural Networks for Uncertainty Quantification

This project implements Bayesian Neural Networks for uncertainty quantification in high-stakes applications like medical diagnosis or autonomous systems. The goal is to develop deep learning models that can provide reliable uncertainty estimates to guide decision-making in critical scenarios.

## Project Structure

- `bayesian_nn.py`: Main implementation file for all models and experiments
- `report.md`: Detailed explanations of statistical and mathematical concepts
- `requirements.txt`: List of required dependencies
- `models/`: Directory to store trained models
- `data/`: Directory to store datasets
- `results/`: Directory to store experiment results and visualizations

## Approach

This project implements three different approaches:

1. **Baseline CNN**: Standard neural network with fixed weights
2. **Monte Carlo Dropout**: Approximate Bayesian inference using dropout at inference time
3. **Bayesian Neural Network with Variational Inference**: Full Bayesian approach using Pyro for variational inference

We evaluate these approaches on two datasets:
- MNIST (for initial testing and validation)
- Medical dataset (e.g., Pneumonia Detection) for real-world high-stakes application

## Setup and Installation

```bash
# Clone the repository
https://github.com/abdo544445/bayesian-neural-networks.git
cd bayesian-neural-networks

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Experiments

```bash
# Run the main experiment script
python bayesian_nn.py
```

## Evaluation Metrics

We evaluate our models using:
- Standard classification metrics (accuracy, precision, recall, F1-score)
- Calibration metrics (Expected Calibration Error)
- Out-of-distribution detection performance
- Decision-making simulation with uncertainty thresholds

## Results

Results including model performance metrics, uncertainty visualizations, and calibration plots will be stored in the `results/` directory.

## References

- Blundell, C., Cornebise, J., Kavukcuoglu, K., & Wierstra, D. (2015). Weight uncertainty in neural networks. arXiv preprint arXiv:1505.05424.
- Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. In International conference on machine learning (pp. 1050-1059).
- Kendall, A., & Gal, Y. (2017). What uncertainties do we need in bayesian deep learning for computer vision? Advances in neural information processing systems, 30. 
