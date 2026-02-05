"""
Neural network architectures for MoB experts.

This module provides various network architectures that can be used as expert models
in the MoB framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SimpleCNN(nn.Module):
    """
    Simple CNN architecture for MNIST and similar small-scale image datasets.

    Architecture:
    - Conv1: 1/3 -> 32 channels, 3x3 kernel
    - Conv2: 32 -> 64 channels, 3x3 kernel
    - MaxPool: 2x2
    - Dropout: 0.25
    - FC1: flatten -> 128 hidden units
    - Dropout: 0.5
    - FC2: 128 -> num_classes
    """

    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 1,
        dropout: float = 0.5,
        width_multiplier: int = 1
    ):
        """
        Initialize the Simple CNN.

        Parameters:
        -----------
        num_classes : int
            Number of output classes.
        input_channels : int
            Number of input channels (1 for grayscale, 3 for RGB).
        dropout : float
            Dropout probability.
        width_multiplier : int
            Multiplier for channel width (default 1). Use 4 for fair comparison
            when competing against 4-expert systems.
        """
        super(SimpleCNN, self).__init__()

        self.width_multiplier = width_multiplier
        self.conv1 = nn.Conv2d(input_channels, 32 * width_multiplier, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32 * width_multiplier, 64 * width_multiplier, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)

        # Calculate the size after convolutions and pooling
        # For MNIST (28x28): after 2 pools -> 7x7
        # For CIFAR (32x32): after 2 pools -> 8x8
        self.fc1_input_size = 64 * width_multiplier * 7 * 7  # Will be dynamically computed

        self.fc1 = None  # Will be initialized on first forward pass
        self.fc2 = None
        self.dropout2 = nn.Dropout(dropout)
        self.num_classes = num_classes
        self._initialized = False

    def _initialize_fc_layers(self, x_shape):
        """Dynamically initialize fully connected layers based on input size."""
        # Compute the flattened size after conv layers
        device = next(self.conv1.parameters()).device
        with torch.no_grad():
            x_dummy = torch.zeros(1, *x_shape[1:], device=device)
            x_dummy = self.pool(F.relu(self.conv1(x_dummy)))
            x_dummy = self.pool(F.relu(self.conv2(x_dummy)))
            x_dummy = self.dropout1(x_dummy)
            flattened_size = x_dummy.view(1, -1).shape[1]

        # Scale hidden layer size with width multiplier
        hidden_size = 128 * self.width_multiplier
        self.fc1 = nn.Linear(flattened_size, hidden_size).to(device)
        self.fc2 = nn.Linear(hidden_size, self.num_classes).to(device)
        self._initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width).

        Returns:
        --------
        logits : torch.Tensor
            Output logits of shape (batch_size, num_classes).
        """
        # Initialize FC layers on first forward pass
        if not self._initialized:
            self._initialize_fc_layers(x.shape)

        # Convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        return x


class LeNet5(nn.Module):
    """
    LeNet-5 architecture, a classic CNN for MNIST.

    Based on LeCun et al. (1998) "Gradient-based learning applied to document recognition"
    """

    def __init__(self, num_classes: int = 10, input_channels: int = 1, width_multiplier: int = 1):
        """
        Initialize LeNet-5.

        Parameters:
        -----------
        num_classes : int
            Number of output classes.
        input_channels : int
            Number of input channels.
        width_multiplier : int
            Multiplier for channel width (default 1). Use 4 for fair comparison
            when competing against 4-expert systems.
        """
        super(LeNet5, self).__init__()

        self.width_multiplier = width_multiplier
        self.conv1 = nn.Conv2d(input_channels, 6 * width_multiplier, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6 * width_multiplier, 16 * width_multiplier, kernel_size=5)
        self.fc1 = nn.Linear(16 * width_multiplier * 5 * 5, 120 * width_multiplier)
        self.fc2 = nn.Linear(120 * width_multiplier, 84 * width_multiplier)
        self.fc3 = nn.Linear(84 * width_multiplier, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LeNet-5.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width).

        Returns:
        --------
        logits : torch.Tensor
            Output logits of shape (batch_size, num_classes).
        """
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MLP(nn.Module):
    """
    Simple Multi-Layer Perceptron for comparison purposes.
    """

    def __init__(
        self,
        input_size: int = 784,
        hidden_sizes: list = [256, 128],
        num_classes: int = 10,
        dropout: float = 0.5,
        width_multiplier: int = 1
    ):
        """
        Initialize the MLP.

        Parameters:
        -----------
        input_size : int
            Size of flattened input (e.g., 784 for 28x28 images).
        hidden_sizes : list
            List of hidden layer sizes.
        num_classes : int
            Number of output classes.
        dropout : float
            Dropout probability.
        width_multiplier : int
            Multiplier for hidden layer width (default 1). Use 4 for fair comparison
            when competing against 4-expert systems.
        """
        super(MLP, self).__init__()

        self.width_multiplier = width_multiplier
        layers = []
        prev_size = input_size

        # Scale all hidden sizes by width_multiplier
        scaled_hidden_sizes = [h * width_multiplier for h in hidden_sizes]

        for hidden_size in scaled_hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width)
            or (batch_size, input_size).

        Returns:
        --------
        logits : torch.Tensor
            Output logits of shape (batch_size, num_classes).
        """
        x = x.view(x.size(0), -1)  # Flatten
        return self.network(x)


def create_model(
    architecture: str,
    num_classes: int = 10,
    input_channels: int = 1,
    **kwargs
) -> nn.Module:
    """
    Factory function to create expert models.

    Parameters:
    -----------
    architecture : str
        Model architecture name ('simple_cnn', 'lenet5', 'mlp').
    num_classes : int
        Number of output classes.
    input_channels : int
        Number of input channels.
    **kwargs : dict
        Additional architecture-specific parameters.

    Returns:
    --------
    model : nn.Module
        Initialized neural network model.
    """
    architecture = architecture.lower()

    if architecture == 'simple_cnn':
        return SimpleCNN(
            num_classes=num_classes,
            input_channels=input_channels,
            dropout=kwargs.get('dropout', 0.5),
            width_multiplier=kwargs.get('width_multiplier', 1)
        )
    elif architecture == 'lenet5':
        return LeNet5(
            num_classes=num_classes,
            input_channels=input_channels,
            width_multiplier=kwargs.get('width_multiplier', 1)
        )
    elif architecture == 'mlp':
        return MLP(
            input_size=kwargs.get('input_size', 784),
            hidden_sizes=kwargs.get('hidden_sizes', [256, 128]),
            num_classes=num_classes,
            dropout=kwargs.get('dropout', 0.5),
            width_multiplier=kwargs.get('width_multiplier', 1)
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
