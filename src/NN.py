import torch
import torch.nn as nn

# Perceptron (MLP) implementation
class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, hidden_units, activation_fn=nn.Tanh()):
        super(MLP, self).__init__()
        self.in_dim = input_size
        self.out_dim = output_size
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.activation_fn = activation_fn

        self.in_layer = nn.Linear(self.in_dim, self.hidden_units)
        self.hidden = nn.ModuleList(
            [nn.Linear(self.hidden_units, self.hidden_units) for _ in range(self.hidden_layers)]
        )
        self.out_layer = nn.Linear(self.hidden_units, self.out_dim)

    def forward(self, x, y, z, t):
        x = torch.cat([x, y, z, t], dim=1)
        x = self.in_layer(x)
        x = self.activation_fn(x)
        for layer in self.hidden:
            x = layer(x)
            x = self.activation_fn(x)
        x = self.out_layer(x)
        return x




