import torch
import torch.nn as nn


class PINN(nn.Module):
    """
    Physics-Informed Neural Network for predicting LPBF process outcomes
    (residual stress, porosity, geometric accuracy) from process parameters.
    
    Args:
        in_dim (int): Total input dimension (process parameters + spatial coordinates + time)
        out_dim (int): Output dimension (number of predicted outcomes)
        width (int): Width of hidden layers
        depth (int): Number of hidden layers
    """
    def __init__(self, in_dim=10, out_dim=3, width=512, depth=5, dropout_rate=0.1):
        """
        Initialize the PINN model.
        
        Args:
            in_dim (int): Total input dimension (process parameters + spatial coordinates + time)
            out_dim (int): Output dimension (number of predicted outcomes)
            width (int): Width of hidden layers
            depth (int): Number of hidden layers
            dropout_rate (float): Dropout rate for MC Dropout (Gal & Ghahramani, 2016).
                                Default 0.1 is common for regression tasks.
        """
        super().__init__()
        self.in_dim = in_dim # Store input dimension
        
        # Build the network with SiLU activation and Dropout
        layers = []
        layers.append(nn.Linear(in_dim, width))
        
        for _ in range(depth-1):
            layers += [
                nn.SiLU(), 
                nn.Dropout(p=dropout_rate), # Monte Carlo Dropout layer
                nn.Linear(width, width)
            ]
        
        self.hidden = nn.Sequential(*layers)
        self.out = nn.Linear(width, out_dim)
        
    def forward(self, S):
        """
        Forward pass through the network
        
        Args:
            S (torch.Tensor): Process parameter tensor [batch_size, in_dim]
        
        Returns:
            torch.Tensor: Predicted outcomes [batch_size, out_dim]
        """
        return self.out(self.hidden(S))

    def predict_with_uncertainty(self, x, num_samples=100):
        """
        Perform Monte Carlo Dropout inference to estimate predictive uncertainty.
        
        Reference:
            Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation. ICML.
            
            For relevance to Additive Manufacturing (AM):
            [1] Zhao, Mirihanage, et al. (2025). "Revealing melt flow instabilities in LPBF".
            [2] "Physics-Informed Neural Networks for Additive Manufacturing: A Review" (2025). 
        
        Args:
            x (torch.Tensor): Input tensor
            num_samples (int): Number of stochastic forward passes
            
        Returns:
            tuple: (mean_prediction, std_deviation)
        """
        # Enable dropout during inference
        self.train() 
        
        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                predictions.append(self.forward(x).unsqueeze(0))
        
        # Stack predictions: [num_samples, batch_size, out_dim]
        predictions = torch.cat(predictions, dim=0)
        
        # Calculate mean and standard deviation (aleatoric + epistemic uncertainty approx)
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        return mean_pred, std_pred