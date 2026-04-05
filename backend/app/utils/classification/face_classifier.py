import torch
import torch.nn as nn
import logging
from typing import Optional, List
from pathlib import Path


class FaceClassifier(nn.Module):
    """Takes concatenated features from spatial (EfficientNet) and frequency
    (FFT/DCT) analysis and outputs a probability score indicating whether
    the face is real or fake.
    
    Architecture:
        - Input: [batch, input_dim] concatenated features
        - Hidden layers: Fully connected layers with ReLU, BatchNorm, and Dropout
        - Output: [batch, num_classes] logits (default: 2 for binary classification)
    """

    def __init__(self,
        input_dim: int,
        num_classes: int =2,
        hidden_dims: Optional[List[int]] =None,
        dropout_rate: float =0.2, 
        use_batch_norm: bool =True,
        path_to_weights: Optional[str] =None
    ):
        """Initialize FaceClassifier.
        Parameters:
            input_dim (int): Input feature dimension
            num_classes (int): Number of output classes (e.g 2 -> REAL/FAKE)
            hidden_dims (List[int], optional): Hidden layer dimensions. If None, uses [1024, 512, 256]
            dropout_rate (float): Dropout rate for regularization
            use_batch_norm (bool): Whether to use batch normalization
            path_to_weights (str, optional): Path to pretrained weights
        Raises:
            ValueError: if parameters are invalid
        """
        super(FaceClassifier, self).__init__()

        self.logger: logging.Logger = logging.getLogger("/".join(__file__.split("/")[-2:]))
        """Logger instance for the FaceClassifier class"""

        try:
            assert input_dim > 0, f"Input dimension must be positive, got {input_dim}"
            assert num_classes >= 2, f"Number of classes must be at least 2, got {num_classes}"
            assert 0 <= dropout_rate < 1, f"Dropout rate must be in [0, 1), got {dropout_rate}"

        except AssertionError as e:
            self.logger.fatal(f"Invalid parameters: {e}")
            raise ValueError(f"Invalid parameters: {e}")

        self.input_dim: int = input_dim
        """Input feature dimension"""
        self.num_classes: int = num_classes
        """Number of output classes"""
        self.hidden_dims: List[int] = hidden_dims if hidden_dims is not None else [1024, 512, 256]
        """Hidden layer dimensions"""
        self.dropout_rate: float = dropout_rate
        """Dropout rate for regularization"""
        self.use_batch_norm: bool = use_batch_norm
        """Whether to use batch normalization"""
        self.classifier = self._build_classifier()
        """Classifier network"""

        # Initialize weights with Kaiming initialization for ReLU
        self._initialize_weights()

        #pretrained weights
        if path_to_weights is not None:
            self.load_weights(path_to_weights)

        self.logger.info(
            f"FaceClassifier initialized: input_dim={input_dim}, num_classes={num_classes}, "
            f"hidden_dims={self.hidden_dims}, dropout_rate={dropout_rate}, "
            f"use_batch_norm={use_batch_norm}"
        )

    def _build_classifier(self) -> nn.Sequential:
        """Build the classifier network architecture.
        - Linear layers for each hidden dimension
        - Optional BatchNorm1d after each linear layer
        - ReLU activation after each hidden layer
        - Dropout for regularization
        - Final output layer without activation
        Returns:
            nn.Sequential: The classifier network
        """
        layers = []
        current_dim = self.input_dim

        #build hidden layers
        for hidden_dim in self.hidden_dims:

            layers.append(nn.Linear(current_dim, hidden_dim))

            if self.use_batch_norm:
                # Use LayerNorm instead of BatchNorm to support batch_size=1
                layers.append(nn.LayerNorm(hidden_dim))

            layers.append(nn.ReLU(inplace=True))

            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))

            current_dim = hidden_dim

        #output layer (no activation - raw logits)
        layers.append(nn.Linear(current_dim, self.num_classes))

        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        """Initialize network weights with Kaiming initialization for ReLU.

        Uses He initialization (Kaiming normal) for Linear layers with ReLU activation,
        which helps prevent vanishing/exploding gradients and improves convergence.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.logger.info("Weights initialized with Kaiming initialization")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the classifier.
        Parameters:
            x (torch.Tensor): Input feature
                - Shape: [batch, input_dim
        Returns:
            torch.Tensor: Classification logits
                - Shape: [batch, num_classes]
                - Raw logits (no softmax applied)
        Raises:
            ValueError: if input shape is invalid
        """
        try:
            assert len(x.shape) == 2, f"Input must be 2D tensor [batch, dim], got shape {x.shape}"
            assert x.shape[1] == self.input_dim, f"Expected input dimension {self.input_dim}, got {x.shape[1]}"

        except AssertionError as e:
            self.logger.error(e)
            raise ValueError(f"Invalid input shape: {e}")

        logits = self.classifier(x)
        self.logger.debug(f"Input shape {x.shape} -> Output logits shape {logits.shape}")
        
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predicts class probabilities using softmax.
        Parameters:
            x (torch.Tensor): Input features or logits
                - If shape is [batch, input_dim]: will call forward() first
                - If shape is [batch, num_classes]: treats as logits directly
        Returns:
            torch.Tensor: Class probabilities
                - Shape: [batch, num_classes]
                - Values in [0, 1], sum to 1 across classes
        Raises:
            ValueError: if input shape is invalid
        """
        # If input has input_dim features, run forward pass first
        if x.shape[1] == self.input_dim:
            logits = self.forward(x)
        else:
            logits = x

        probabilities = torch.softmax(logits, dim=1)
        self.logger.debug(f"Computed probabilities with shape {probabilities.shape}")
        return probabilities

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class labels (argmax).
        Parameters:
            x (torch.Tensor): Input features or logits
                - If shape is [batch, input_dim]: will call forward() first
                - If shape is [batch, num_classes]: treats as logits directly
        Returns:
            torch.Tensor: Predicted class indices
                - Shape: [batch]
                - Values in [0, num_classes)
        Raises:
            ValueError: if input shape is invalid
        """
        # If input has input_dim features, run forward pass first
        if x.shape[1] == self.input_dim:
            logits = self.forward(x)
        else:
            logits = x

        predictions = torch.argmax(logits, dim=1)
        self.logger.debug(f"Predicted classes with shape {predictions.shape}")
        return predictions

    def save_weights(self, output_path: str) -> None:
        """Save classifier weights to disk.
        Parameters:
            output_path (str): Output file path (.pth)
        Raises:
            RuntimeError: if saving fails
        """
        try:
            path_obj = Path(output_path)
            #create parent directory if it doesn't exist
            path_obj.parent.mkdir(parents=True, exist_ok=True)

            torch.save(self.state_dict(), path_obj)
            self.logger.info(f"Saved weights to {path_obj}")

        except Exception as e:
            self.logger.error(f"Error saving weights: {e}")
            raise RuntimeError(f"Error saving weights: {e}")

    def load_weights(self, input_path: str) -> None:
        """Load pretrained classifier weights from disk.
        Parameters:
            input_path (str): Path to weights file (.pth)
        Raises:
            ValueError: if weights file doesn't exist
            RuntimeError: if loading fails
        """
        try:
            path = Path(input_path)
            assert path.exists(), f"Weights file does not exist: {input_path}"

        except AssertionError as e:
            self.logger.error(f"Invalid parameter: {e}")
            raise ValueError(f"Invalid parameter: {e}")

        try:
            state_dict = torch.load(path, map_location='cpu')
            self.load_state_dict(state_dict)
            self.logger.info(f"Loaded weights from {path}")

        except Exception as e:
            self.logger.error(f"Error loading weights: {e}")
            raise RuntimeError(f"Error loading weights: {e}")
