import os
import sys
from pathlib import Path

# Add project root to system path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import TensorDataset, DataLoader
from models.freenet import FreeNet
from models.mlp import MLP
from models.mlp_sqrelu import MLP_SqReLU
from data_generators.monomials import generate_monomial_data
from utilities.data_utilities import split_data
from utilities.general_utilities import set_random_seed
from trainers.train_nn import train_model
from testers.test_nn import test_model
from optimizers.optimizer_params import get_optimizer_and_scheduler
from plotters.get_monomial_values import get_plotter_values

from plotters.draw_monomial_plot import draw_monomial_plot

torch.autograd.set_detect_anomaly(True)

# Set random seed
set_random_seed(42)

# Parameters
degree = 32
num_data_points = 1000
batch_size = 32
num_epochs = 10


learning_rate = 0.01
optimizer_name = "adamw"
percentage_test_split = 0.01

# Generate data
x, y, coefficient = generate_monomial_data(degree, num_data_points, coefficient=None)

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
mlp_sqrelu_model = MLP_SqReLU(input_dim, [4, 4], output_dim)

# Get optimizers and schedulers
freenet_optimizer, freenet_scheduler = get_optimizer_and_scheduler(freenet_model, optimizer_name, learning_rate)
mlp_optimizer, mlp_scheduler = get_optimizer_and_scheduler(mlp_model, optimizer_name, learning_rate)
mlp_sqrelu_optimizer, mlp_sqrelu_scheduler = get_optimizer_and_scheduler(mlp_sqrelu_model, optimizer_name, learning_rate)

# Train models
print("Training FreeNet:")
trained_freenet = train_model(freenet_model, train_loader, num_epochs, freenet_optimizer, freenet_scheduler)
print("Training MLP:")
trained_mlp = train_model(mlp_model, train_loader, num_epochs, mlp_optimizer, mlp_scheduler)
print("Training MLP Sq Relu:")
trained_mlp_sqrelu = train_model(mlp_sqrelu_model, train_loader, num_epochs, mlp_sqrelu_optimizer, mlp_sqrelu_scheduler)

# Test models
print("Testing FreeNet:")
freenet_results = test_model(trained_freenet, test_loader, degree, coefficient, get_test_metrics=True, get_distance_metrics=True)
print("\nTesting MLP:")
mlp_results = test_model(trained_mlp, test_loader, degree, coefficient, get_test_metrics=True, get_distance_metrics=True)
print("\nTesting MLP Square Relu:")
mlp_sqrelu_results = test_model(trained_mlp_sqrelu, test_loader, degree, coefficient, get_test_metrics=True, get_distance_metrics=True)


import datetime
# Plot results
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
figures_dir = os.path.join(project_root, 'figures')
os.makedirs(figures_dir, exist_ok=True)
sim=1
mlp_dim = "2,2"

x_plot, y_true, y_pred_freenet = get_plotter_values(trained_freenet, degree, coefficient)
draw_monomial_plot(x_plot, y_true, y_pred_freenet, f"FreeNet_", degree, coefficient, save_path=os.path.join(figures_dir, f"FreeNet_degree{degree}_hidden{hidden_dim}_sim{sim+1}_{timestamp}.png"))

x_plot, y_true, y_pred_mlp = get_plotter_values(trained_mlp, degree, coefficient)
draw_monomial_plot(x_plot, y_true, y_pred_mlp, f"MLP_", degree, coefficient, save_path=os.path.join(figures_dir, f"MLP_degree{degree}_hidden{mlp_dim}_sim{sim+1}_{timestamp}.png"))

x_plot, y_true, y_pred_mlp_sqrelu = get_plotter_values(trained_mlp_sqrelu, degree, coefficient)
draw_monomial_plot(x_plot, y_true, y_pred_mlp_sqrelu, f"MLPSqReLU_", degree, coefficient, save_path=os.path.join(figures_dir, f"MLPSqReLU_degree{degree}_hidden{mlp_dim}_sim{sim+1}_{timestamp}.png"))


# Print comparison
print("\nMetric Comparison:")
for metric in freenet_results.keys():
    if freenet_results[metric] is not None and mlp_results[metric] is not None:
        print(f"{metric}:")
        print(f"  FreeNet: {freenet_results[metric]:.6f}")
        print(f"  MLP:     {mlp_results[metric]:.6f}")
        print(f"  Ratio (FreeNet/MLP): {freenet_results[metric]/mlp_results[metric]:.6f}")
        print()