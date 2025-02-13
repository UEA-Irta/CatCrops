"""
TempCNN - PyTorch Implementation for Time Series Classification

This script defines the `TempCNN` class, a PyTorch-based Temporal Convolutional Neural Network (TempCNN)
for time series classification.

This script is a direct copy of the TempCNN implementation from the BreizhCrops repository,
which is based on the original work by **Pelletier et al. (2019)**:
- Original repository: https://github.com/charlotte-pel/temporalCNN
- Research paper: https://www.mdpi.com/2072-4292/11/5/523

Original source:
BreizhCrops GitHub repository:
https://github.com/dl4sits/BreizhCrops/blob/master/breizhcrops/models/TempCNN.py

Acknowledgment:
This script is a direct copy of the TempCNN implementation from the BreizhCrops repository,
which in turn is based on the original TempCNN model by Pelletier et al. No modifications have been made.

Author:
- Pelletier et al. (Original TempCNN)
- BreizhCrops Team
- Original repository: https://github.com/dl4sits/BreizhCrops

"""

import os
import torch
import torch.nn as nn
import torch.utils.data



__all__ = ['TempCNN']

class TempCNN(torch.nn.Module):
    def __init__(self, input_dim=13, num_classes=9, sequencelength=45, kernel_size=7, hidden_dims=128, dropout=0.18203942949809093):
        super(TempCNN, self).__init__()
        self.modelname = f"TempCNN_input-dim={input_dim}_num-classes={num_classes}_sequencelenght={sequencelength}_" \
                         f"kernelsize={kernel_size}_hidden-dims={hidden_dims}_dropout={dropout}"

        self.hidden_dims = hidden_dims

        self.conv_bn_relu1 = Conv1D_BatchNorm_Relu_Dropout(input_dim, hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        self.conv_bn_relu2 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        self.conv_bn_relu3 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        self.flatten = Flatten()
        self.dense = FC_BatchNorm_Relu_Dropout(hidden_dims * sequencelength, 4 * hidden_dims, drop_probability=dropout)
        self.logsoftmax = nn.Sequential(nn.Linear(4 * hidden_dims, num_classes), nn.LogSoftmax(dim=-1))

    def forward(self, x):
        # require NxTxD
        x = x.transpose(1,2)
        x = self.conv_bn_relu1(x)
        x = self.conv_bn_relu2(x)
        x = self.conv_bn_relu3(x)
        x = self.flatten(x)
        x = self.dense(x)
        return self.logsoftmax(x)

    def save(self, path="model.pth", **kwargs):
        print("\nsaving model to " + path)
        model_state = self.state_dict()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(dict(model_state=model_state, **kwargs), path)

    def load(self, path):
        print("loading model from " + path)
        snapshot = torch.load(path, map_location="cpu")
        model_state = snapshot.pop('model_state', snapshot)
        self.load_state_dict(model_state)
        return snapshot


class Conv1D_BatchNorm_Relu_Dropout(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_size=5, drop_probability=0.5):
        super(Conv1D_BatchNorm_Relu_Dropout, self).__init__()

        self.block = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dims, kernel_size, padding=(kernel_size // 2)),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_probability)
        )

    def forward(self, X):
        return self.block(X)


class FC_BatchNorm_Relu_Dropout(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, drop_probability=0.5):
        super(FC_BatchNorm_Relu_Dropout, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_probability)
        )

    def forward(self, X):
        return self.block(X)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
