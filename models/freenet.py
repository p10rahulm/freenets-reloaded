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


class SquareReLUCapped(nn.Module):
    def __init__(self, cap_value=10.0):
        """
        :param clip_value: The maximum value to clip the output after squaring the ReLU result.
        """
        super(SquareReLUCapped, self).__init__()
        self.cap_value = cap_value  # Set the value to which we want to clip the output.

    def forward(self, x):
        # Apply ReLU and then square the result
        squared_relu = F.relu(x) ** 2
        
        # Cap the output values to prevent them from getting too large
        capped_output = torch.clamp(squared_relu, max=self.cap_value)
        
        return capped_output


class FreeNetCapped(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FreeNetCapped, self).__init__()
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
        self.activation = SquareReLUCapped()

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