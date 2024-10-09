import torch
import torch.nn as nn
import torch.nn.functional as F

class SquareReLU(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2

class FreeNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FreeNet, self).__init__()
        self.hidden_dim = hidden_dim

        # Input to all hidden neurons
        self.input_to_hidden = nn.Linear(input_dim, hidden_dim)

        # Hidden to subsequent hidden neurons
        self.hidden_to_hidden = nn.ModuleList([
            nn.Linear(i + 1, hidden_dim - i - 1) for i in range(hidden_dim - 1)
        ])

        # All hidden to output
        self.hidden_to_output = nn.Linear(hidden_dim, output_dim)

        # Activation function
        self.activation = SquareReLU()

    def forward(self, x):
        # Input to all hidden neurons
        h = self.activation(self.input_to_hidden(x))

        # Hidden to subsequent hidden neurons
        for i in range(self.hidden_dim - 1):
            next_input = h[:, :i+1]
            outputs = self.activation(self.hidden_to_hidden[i](next_input))
            h = torch.cat([h[:, :i+1], h[:, i+1:] + outputs], dim=1)

        # All hidden to output
        out = self.hidden_to_output(h)
        return out


class FreeNetTrig(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FreeNetTrig, self).__init__()
        self.hidden_dim = hidden_dim

        # Input to all hidden neurons
        self.input_to_hidden = nn.Linear(input_dim * 2, hidden_dim)  # *2 for sin and cos

        # Hidden to subsequent hidden neurons
        self.hidden_to_hidden = nn.ModuleList([
            nn.Linear(i + 1, hidden_dim - i - 1) for i in range(hidden_dim - 1)
        ])

        # All hidden to output
        self.hidden_to_output = nn.Linear(hidden_dim, output_dim)

        # Activation function
        self.activation = SquareReLU()

    def forward(self, x):
        # Compute sine and cosine of input
        x_sin = torch.sin(x)
        x_cos = torch.cos(x)
        x = torch.cat([x_sin, x_cos], dim=1)

        # Input to all hidden neurons
        h = self.activation(self.input_to_hidden(x))

        # Hidden to subsequent hidden neurons
        for i in range(self.hidden_dim - 1):
            next_input = h[:, :i+1]
            outputs = self.activation(self.hidden_to_hidden[i](next_input))
            h = torch.cat([h[:, :i+1], h[:, i+1:] + outputs], dim=1)

        # All hidden to output
        out = self.hidden_to_output(h)
        return out