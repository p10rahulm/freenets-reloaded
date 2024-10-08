import os
import sys
from pathlib import Path


# Add project root to system path
project_root = Path(__file__).resolve().parent
while not (project_root / '.git').exists() and project_root != project_root.parent:
    project_root = project_root.parent
sys.path.insert(0, str(project_root))


import torch
from torch.utils.data import TensorDataset, DataLoader
from models.freenet import FreeNet
from models.mlp import MLP
from data_generators.monomials import generate_monomial_data
from utilities.data_utilities import split_data
from trainers.train_nn import train_model
from testers.test_nn import test_model

torch.autograd.set_detect_anomaly(True)



# Parameters
degree = 64
num_data_points = 100000
batch_size = 32
num_epochs = 1

# Generate data
x, y = generate_monomial_data(degree, num_data_points)

# Split data
x_train, x_test, y_train, y_test = split_data(x, y)

# Create data loaders
train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
test_dataset = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Initialize models
input_dim = 1
hidden_dim = 8  # O(log N)
output_dim = 1

freenet_model = FreeNet(input_dim, hidden_dim, output_dim)
mlp_model = MLP(input_dim, [4, 4], output_dim)

# Train models
print("Training FreeNet:")
trained_freenet = train_model(freenet_model, train_loader, num_epochs)
print("Training MLP:")
trained_mlp = train_model(mlp_model, train_loader, num_epochs)

# Test models
print("Testing FreeNet:")
test_model(trained_freenet, test_loader)
print("Testing MLP:")
test_model(trained_mlp, test_loader)
