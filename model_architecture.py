import torch
from torch import nn

class IDACModelV0(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape, activation='relu'):
        super().__init__()

        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'relu':
            self.activation= nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise(ValueError(f'{activation} is not valid for this model'))

        self.layer_stack = nn.Sequential(
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            self.activation,
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            self.activation,
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            self.activation,
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )

    def forward(self, x):
        return self.layer_stack(x)