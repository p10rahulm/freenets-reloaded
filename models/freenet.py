import torch
import torch.nn as nn

class FreeNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FreeNet, self).__init__()
        self.hidden_dim = hidden_dim

        # Input to hidden neurons
        self.input_to_hidden = nn.Linear(input_dim, hidden_dim)

        # Hidden to subsequent hidden neurons
        self.hidden_to_hidden = nn.ModuleList([
            nn.Linear(1, hidden_dim - i - 1) for i in range(hidden_dim - 1)
        ])

        # Hidden to output
        self.hidden_to_output = nn.Linear(hidden_dim, output_dim)

        # Activation function (Squared ReLU)
        self.activation = lambda x: torch.relu(x) ** 2

    def forward(self, x):
        # Input to hidden
        h = self.activation(self.input_to_hidden(x))

        # Hidden to hidden connections
        for i in range(self.hidden_dim - 1):
            next_input = h[:, i].unsqueeze(1)
            outputs = self.activation(self.hidden_to_hidden[i](next_input))
            h = torch.cat([h[:, :i+1], h[:, i+1:] + outputs], dim=1)

        # Hidden to output
        out = self.hidden_to_output(h)
        return out