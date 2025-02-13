"""
TransformerModel - PyTorch Implementation for Time Series Classification

This script defines the `TransformerModel` class, a PyTorch-based Transformer Encoder model
designed for time series classification. The model is optimized for processing sequential data,
such as agricultural time series, leveraging attention mechanisms for feature extraction.


Features:
- Implements a Transformer Encoder architecture for time series classification.
- Customizable hyperparameters:
  - Input dimension (`input_dim`): Defines the number of input features.
  - Model depth (`d_model`): Specifies the dimensionality of embeddings.
  - Number of heads (`n_head`): Multi-head attention mechanism.
  - Number of layers (`n_layers`): Depth of the Transformer Encoder.
  - Inner-layer dimension (`d_inner`): Size of the feedforward network.
  - Activation function (`activation`): Can be `"relu"` or `"gelu"`.
  - Dropout rate (`dropout`): Controls regularization.
- Includes a Flatten layer to reshape the outputs for classification.
- Utilizes log-softmax activation for output probabilities.


Original source:
BreizhCrops GitHub repository:
https://github.com/dl4sits/BreizhCrops/blob/master/breizhcrops/models/TransformerModel.py


Acknowledgment:
This script is a direct copy of the Transformer model implementation from the BreizhCrops repository,
with added comments to improve clarity and documentation.


Author:
- BreizhCrops Team
- Original repository: https://github.com/dl4sits/BreizhCrops
"""

import torch
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules import LayerNorm, Linear, ReLU

import torch.nn as nn
import torch.nn.functional as F

__all__ = ['TransformerModel']


class TransformerModel(nn.Module):
    def __init__(self, input_dim=13, num_classes=9, d_model=64, n_head=2, n_layers=5,
                 d_inner=128, activation="relu", dropout=0.017998950510888446):
        super(TransformerModel, self).__init__()

        # Generate the model name based on the parameters
        self.modelname = f"TransformerEncoder_input-dim={input_dim}_num-classes={num_classes}_" \
                         f"d-model={d_model}_d-inner={d_inner}_n-layers={n_layers}_n-head={n_head}_" \
                         f"dropout={dropout}"

        # Input linear layer
        self.inlinear = Linear(input_dim, d_model)

        # ReLU activation function
        self.relu = ReLU()

        # Transformer encoder layers
        encoder_layer = TransformerEncoderLayer(d_model, n_head, d_inner, dropout, activation)
        encoder_norm = LayerNorm(d_model)
        self.transformerencoder = TransformerEncoder(encoder_layer, n_layers, encoder_norm)

        # Flatten layer to convert tensors into vectors
        self.flatten = Flatten()

        # Output linear layer
        self.outlinear = Linear(d_model, num_classes)

    def forward(self, x):
        """
        Defines the data flow through the model.

        Returns the class probabilities.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            torch.Tensor: Log probabilities of shape (batch_size, num_classes).
        """
        # Input linear layer
        x = self.inlinear(x)

        # ReLU activation function
        x = self.relu(x)

        # Transpose dimensions for the Transformer
        x = x.transpose(0, 1)  # Shape transformation: (N x T x D -> T x N x D)

        # Transformer encoder
        x = self.transformerencoder(x)

        # Transpose dimensions back after Transformer
        x = x.transpose(0, 1)  # Shape transformation: (T x N x D -> N x T x D)

        # Select the maximum value for each time sequence
        x = x.max(1)[0]

        # ReLU activation function
        x = self.relu(x)

        # Output linear layer
        logits = self.outlinear(x)

        # Compute log probabilities with softmax
        logprobabilities = F.log_softmax(logits, dim=-1)
        return logprobabilities


class Flatten(nn.Module):
    def forward(self, input):
        """
        Flattens the input tensor.

        Args:
            input (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Flattened tensor of shape (batch_size, -1).
        """
        return input.reshape(input.size(0), -1)
