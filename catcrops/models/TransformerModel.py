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
        # Afegim el nom del model a partir dels paràmetres
        self.modelname = f"TransformerEncoder_input-dim={input_dim}_num-classes={num_classes}_" \
                         f"d-model={d_model}_d-inner={d_inner}_n-layers={n_layers}_n-head={n_head}_" \
                         f"dropout={dropout}"

        # Capa d'entrada lineal
        self.inlinear = Linear(input_dim, d_model)

        # Funció d'activació ReLU
        self.relu = ReLU()

        # Capes de l'encoder Transformer
        encoder_layer = TransformerEncoderLayer(d_model, n_head, d_inner, dropout, activation)
        encoder_norm = LayerNorm(d_model)
        self.transformerencoder = TransformerEncoder(encoder_layer, n_layers, encoder_norm)

        # Capa de sortida lineal
        self.flatten = Flatten()

        # Capa de flatten per convertir els tensors en vectors.
        self.outlinear = Linear(d_model, num_classes)

    def forward(self, x):
        """
        Defineix el flux de dades a través del model.

        Retorna les probabilitats de cada classe
        :rtype: object
        """
        # Capa d'entrada lineal
        x = self.inlinear(x)

        # Funció d'activació ReLU
        x = self.relu(x)

        # Transposició de dimensions per al Transformer
        x = x.transpose(0, 1)  # N x T x D -> T x N x D

        # Encoder Transformer
        x = self.transformerencoder(x)

        # Transposició de dimensions després del Transformer
        x = x.transpose(0, 1)  # T x N x D -> N x T x D

        # Selecció del màxim per a cada seqüència temporal
        x = x.max(1)[0]

        # Funció d'activació ReLU
        x = self.relu(x)

        # Capa de sortida lineal
        logits = self.outlinear(x)

        # Càlcul de les log-probabilitats amb softmax
        logprobabilities = F.log_softmax(logits, dim=-1)
        return logprobabilities


class Flatten(nn.Module):
    def forward(self, input):
        return input.reshape(input.size(0), -1)
