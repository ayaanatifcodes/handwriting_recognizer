import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ConvBlock(nn.Module):
    """
    Convolutional block with BatchNorm, activation, and optional pooling.
    Forms the building blocks of the feature extraction pipeline.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_batchnorm: bool = True,
        activation: str = 'relu',
        pool_size: Optional[Tuple[int, int]] = None,
        dropout_rate: float = 0.0
    ):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batchnorm
        )
        
        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.batchnorm = nn.BatchNorm2d(out_channels)
        
        # Activation function selection
        if activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU(inplace=True)
        
        # Optional pooling
        self.pool = None
        if pool_size is not None:
            self.pool = nn.MaxPool2d(kernel_size=pool_size)
        
        # Optional dropout
        self.dropout = None
        if dropout_rate > 0:
            self.dropout = nn.Dropout2d(dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.use_batchnorm:
            x = self.batchnorm(x)
        x = self.activation(x)
        if self.pool is not None:
            x = self.pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class FeatureExtractor(nn.Module):
    """
    Deep convolutional network for extracting visual features from handwriting.
    Progressively reduces spatial dimensions while increasing feature depth.
    """
    def __init__(
        self,
        in_channels: int = 1,
        base_filters: int = 48,
        activation: str = 'relu',
        dropout_rate: float = 0.15
    ):
        super(FeatureExtractor, self).__init__()
        
        # Progressive channel expansion: 1 -> 48 -> 96 -> 192 -> 384
        # Spatial reduction through pooling
        
        # Stage 1: Initial feature extraction
        self.stage1 = nn.Sequential(
            ConvBlock(in_channels, base_filters, kernel_size=3, 
                     activation=activation, pool_size=(2, 2)),
            ConvBlock(base_filters, base_filters, kernel_size=3, 
                     activation=activation, dropout_rate=dropout_rate)
        )
        
        # Stage 2: Deeper features
        self.stage2 = nn.Sequential(
            ConvBlock(base_filters, base_filters * 2, kernel_size=3,
                     activation=activation, pool_size=(2, 2)),
            ConvBlock(base_filters * 2, base_filters * 2, kernel_size=3,
                     activation=activation, dropout_rate=dropout_rate)
        )
        
        # Stage 3: High-level patterns (preserve width more)
        self.stage3 = nn.Sequential(
            ConvBlock(base_filters * 2, base_filters * 4, kernel_size=3,
                     activation=activation, pool_size=(2, 1)),
            ConvBlock(base_filters * 4, base_filters * 4, kernel_size=3,
                     activation=activation, dropout_rate=dropout_rate)
        )
        
        # Stage 4: Abstract features (minimal height reduction)
        self.stage4 = nn.Sequential(
            ConvBlock(base_filters * 4, base_filters * 8, kernel_size=3,
                     activation=activation, pool_size=(2, 1)),
            ConvBlock(base_filters * 8, base_filters * 8, kernel_size=3,
                     activation=activation, dropout_rate=dropout_rate)
        )
        
        # Final refinement layer
        self.final_conv = nn.Conv2d(
            base_filters * 8, 
            base_filters * 8, 
            kernel_size=2, 
            padding=0
        )
        
        self.output_channels = base_filters * 8
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input images.
        
        Args:
            x: Input tensor (batch, channels, height, width)
        
        Returns:
            Feature maps (batch, output_channels, reduced_height, width')
        """
        x = self.stage1(x)  # Reduce by 2x2
        x = self.stage2(x)  # Reduce by 2x2
        x = self.stage3(x)  # Reduce by 2x1 (preserve width)
        x = self.stage4(x)  # Reduce by 2x1 (preserve width)
        x = self.final_conv(x)  # Final refinement
        return x


class SequenceEncoder(nn.Module):
    """
    Bidirectional LSTM encoder for modeling sequential dependencies
    in the extracted features.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout_rate: float = 0.2
    ):
        super(SequenceEncoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Stack of bidirectional LSTMs
        self.lstm_layers = nn.ModuleList()
        
        for layer_idx in range(num_layers):
            layer_input_size = input_size if layer_idx == 0 else hidden_size * 2
            
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=layer_input_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                    dropout=0  # We'll add dropout manually
                )
            )
        
        # Dropout between LSTM layers
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode sequences with bidirectional context.
        
        Args:
            x: Input sequences (batch, seq_len, features)
        
        Returns:
            Encoded sequences (batch, seq_len, hidden_size * 2)
        """
        for idx, lstm_layer in enumerate(self.lstm_layers):
            x, _ = lstm_layer(x)
            
            # Apply dropout between layers (but not after the last one)
            if self.dropout is not None and idx < self.num_layers - 1:
                x = self.dropout(x)
        
        return x


class HandwritingRecognizer(nn.Module):
    """
    Complete handwriting recognition model.
    Architecture: CNN Feature Extraction -> Sequence Encoding -> Character Prediction
    """
    def __init__(
        self,
        num_classes: int,
        input_channels: int = 1,
        base_filters: int = 48,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        activation: str = 'relu',
        dropout_rate: float = 0.15
    ):
        """
        Initialize the handwriting recognizer.
        
        Args:
            num_classes: Number of output classes (vocab size + 1 for CTC blank)
            input_channels: Number of input image channels (1 for grayscale, 3 for RGB)
            base_filters: Base number of convolutional filters
            lstm_hidden: Hidden size for LSTM layers
            lstm_layers: Number of stacked LSTM layers
            activation: Activation function type ('relu', 'leaky_relu', 'gelu')
            dropout_rate: Dropout probability
        """
        super(HandwritingRecognizer, self).__init__()
        
        self.num_classes = num_classes
        
        # Feature extraction network
        self.feature_extractor = FeatureExtractor(
            in_channels=input_channels,
            base_filters=base_filters,
            activation=activation,
            dropout_rate=dropout_rate
        )
        
        # Calculate sequence input dimension after CNN
        # The CNN output will be flattened along height dimension
        # For 224x224 input: (224/2/2/2/2 - 1) = 13 height after all pooling/conv
        self.feature_dim = self.feature_extractor.output_channels
        
        # Bridge layer to map CNN features to sequence
        self.feature_to_seq = nn.Linear(
            self.feature_dim,  # We'll flatten height into this
            lstm_hidden
        )
        
        # Sequence modeling network
        self.sequence_encoder = SequenceEncoder(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout_rate=dropout_rate
        )
        
        # Character prediction head
        self.classifier = nn.Linear(lstm_hidden * 2, num_classes)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the recognition pipeline.
        
        Args:
            images: Input images (batch, channels, height, width)
        
        Returns:
            Character predictions (seq_len, batch, num_classes) for CTC loss
        """
        # Extract visual features
        features = self.feature_extractor(images)
        # Shape: (batch, channels, height, width)
        
        batch_size, channels, height, width = features.size()
        
        # Reshape for sequence processing
        # Collapse height into channels, use width as sequence length
        features = features.permute(0, 3, 1, 2)  # (batch, width, channels, height)
        features = features.contiguous().view(batch_size, width, channels * height)
        # Shape: (batch, width, channels*height)
        
        # Map to sequence dimension
        sequence = self.feature_to_seq(features)
        # Shape: (batch, seq_len, lstm_hidden)
        
        # Encode sequences with bidirectional context
        encoded = self.sequence_encoder(sequence)
        # Shape: (batch, seq_len, lstm_hidden * 2)
        
        # Predict characters
        logits = self.classifier(encoded)
        # Shape: (batch, seq_len, num_classes)
        
        # Apply log softmax for CTC loss
        log_probs = F.log_softmax(logits, dim=2)
        
        # Transpose to CTC format: (seq_len, batch, num_classes)
        log_probs = log_probs.permute(1, 0, 2)
        
        return log_probs
    
    def predict(self, image: torch.Tensor, vocab: str, device: str = 'cpu') -> str:
        """
        Generate text prediction from an image.
        
        Args:
            image: Input image tensor (C, H, W) or (1, C, H, W)
            vocab: Vocabulary string for decoding
            device: Device to run inference on
        
        Returns:
            Decoded text string
        """
        self.eval()
        
        with torch.no_grad():
            # Add batch dimension if needed
            if image.dim() == 3:
                image = image.unsqueeze(0)
            
            image = image.to(device)
            
            # Forward pass
            log_probs = self.forward(image)
            
            # Greedy decode (take argmax at each timestep)
            _, predictions = torch.max(log_probs, dim=2)
            predictions = predictions.squeeze().cpu().numpy()
            
            # CTC decode: remove blanks and consecutive duplicates
            decoded = self._ctc_greedy_decode(predictions, vocab)
            
            return decoded
    
    @staticmethod
    def _ctc_greedy_decode(predictions, vocab: str) -> str:
        """
        Simple CTC greedy decoding.
        
        Args:
            predictions: Array of predicted class indices
            vocab: Vocabulary string
        
        Returns:
            Decoded text
        """
        blank_idx = len(vocab)
        result = []
        prev_idx = None
        
        for idx in predictions:
            # Skip blank tokens
            if idx == blank_idx:
                prev_idx = None
                continue
            
            # Skip consecutive duplicates
            if idx != prev_idx:
                if idx < len(vocab):
                    result.append(vocab[idx])
                prev_idx = idx
        
        return ''.join(result)


def build_recognizer(
    vocab_size: int,
    input_channels: int = 1,
    base_filters: int = 48,
    lstm_hidden: int = 256,
    lstm_layers: int = 2,
    activation: str = 'relu',
    dropout_rate: float = 0.15
) -> HandwritingRecognizer:
    """
    Factory function to create a handwriting recognition model.
    
    Args:
        vocab_size: Size of character vocabulary
        input_channels: Number of input channels (1 for grayscale, 3 for RGB)
        base_filters: Base number of filters in CNN
        lstm_hidden: LSTM hidden dimension
        lstm_layers: Number of LSTM layers
        activation: Activation function
        dropout_rate: Dropout probability
    
    Returns:
        Initialized model
    """
    # Add 1 for CTC blank token
    num_classes = vocab_size + 1
    
    model = HandwritingRecognizer(
        num_classes=num_classes,
        input_channels=input_channels,
        base_filters=base_filters,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        activation=activation,
        dropout_rate=dropout_rate
    )
    
    return model


if __name__ == "__main__":
    # Test the model
    vocab = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?'-"
    
    model = build_recognizer(
        vocab_size=len(vocab),
        input_channels=1,
        base_filters=48,
        lstm_hidden=256,
        lstm_layers=2
    )
    
    # Test forward pass
    dummy_batch = torch.randn(4, 1, 224, 224)
    output = model(dummy_batch)
    
    print("=" * 60)
    print("Model Architecture Test")
    print("=" * 60)
    print(f"Input shape: {dummy_batch.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected format: (seq_len, batch, num_classes)")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of classes (vocab + blank): {len(vocab) + 1}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 60)
    
    # Test prediction
    single_image = torch.randn(1, 224, 224)
    prediction = model.predict(single_image, vocab)
    print(f"\nSample prediction (random input): '{prediction}'")
    print("\nModel is ready for training! 🚀")