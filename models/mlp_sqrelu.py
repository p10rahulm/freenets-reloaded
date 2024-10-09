import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP_SqReLU(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP_SqReLU, self).__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.model:
            if isinstance(layer, nn.ReLU):
                x = F.relu(x) ** 2  # Squared ReLU activation
            else:
                x = layer(x)
        return x


class MLP_SqReLUTrig(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP_SqReLUTrig, self).__init__()
        layers = []
        dims = [input_dim * 2] + hidden_dims  # *2 for sin and cos
        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Compute sine and cosine of input
        x_sin = torch.sin(x)
        x_cos = torch.cos(x)
        x = torch.cat([x_sin, x_cos], dim=1)

        for layer in self.model:
            if isinstance(layer, nn.ReLU):
                x = F.relu(x) ** 2  # Squared ReLU activation
            else:
                x = layer(x)
        return x