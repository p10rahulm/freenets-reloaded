import os
import sys
from pathlib import Path
import datetime
import json
import math

# Add project root to system path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import statistics
import torch
from torch.utils.data import TensorDataset, DataLoader
from models.freenet import FreeNet
from models.mlp import MLP
from models.mlp_sqrelu import MLP_SqReLU
from data_generators.sparse_polynomials import generate_sparse_polynomial_data
from utilities.data_utilities import split_data
from utilities.general_utilities import set_random_seed
from trainers.train_nn import train_model
from testers.test_nn import test_model
from optimizers.optimizer_params import get_optimizer_and_scheduler
from plotters.sparse_polynomial_plotter import save_sparse_polynomial_plots
import numpy as np

def run_experiment(d, k, num_sims=5):
    results = {
        'FreeNet': {'aggregate': {}, 'individual': []},
        'MLP': {'aggregate': {}, 'individual': []},
        'MLP_SqReLU': {'aggregate': {}, 'individual': []}
    }

    # Calculate hidden dimensions
    hidden_dim_freenet = int(2 * k * math.log2(d))
    hidden_dim_mlp = [hidden_dim_freenet, hidden_dim_freenet]  # Two layers with the same number of neurons

    for sim in range(num_sims):
        print(f"\nSimulation {sim + 1}/{num_sims}")
        set_random_seed(42 + sim)

        # Parameters
        num_data_points = 10000
        batch_size = 32
        num_epochs = 100
        learning_rate = 0.01
        optimizer_name = "adamw"
        percentage_test_split = 0.01

        # Generate data
        x, y, coefficients, degrees = generate_sparse_polynomial_data(d, k, num_data_points)

        # Split data
        x_train, x_test, y_train, y_test = split_data(x, y, test_size=percentage_test_split)

        # Create data loaders
        train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
        test_dataset = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # Initialize models
        input_dim = 1
        output_dim = 1

        freenet_model = FreeNet(input_dim, hidden_dim_freenet, output_dim)
        mlp_model = MLP(input_dim, hidden_dim_mlp, output_dim)
        mlp_sqrelu_model = MLP_SqReLU(input_dim, hidden_dim_mlp, output_dim)

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
        freenet_results = test_model(trained_freenet, test_loader, coefficients, degrees)
        print("\nTesting MLP:")
        mlp_results = test_model(trained_mlp, test_loader, coefficients, degrees)
        print("\nTesting MLP Square Relu:")
        mlp_sqrelu_results = test_model(trained_mlp_sqrelu, test_loader, coefficients, degrees)

        # Store results
        results['FreeNet']['individual'].append(freenet_results)
        results['MLP']['individual'].append(mlp_results)
        results['MLP_SqReLU']['individual'].append(mlp_sqrelu_results)

        # Generate plots
        x_plot = np.linspace(0, 1, 1000).reshape(-1, 1)
        y_true = np.zeros_like(x_plot)
        for coef, degree in zip(coefficients, degrees):
            y_true += coef * x_plot**degree
        
        y_pred_dict = {
            'FreeNet': trained_freenet(torch.FloatTensor(x_plot)).detach().numpy(),
            'MLP': trained_mlp(torch.FloatTensor(x_plot)).detach().numpy(),
            'MLP_SqReLU': trained_mlp_sqrelu(torch.FloatTensor(x_plot)).detach().numpy()
        }

        figures_dir = os.path.join(project_root, 'figures', 'sparse_polynomials', f'd{d}_k{k}_sim{sim+1}')
        save_sparse_polynomial_plots(x_plot, y_true, y_pred_dict, coefficients, degrees, figures_dir)

    # Compute aggregate metrics
    for model in results:
        for metric in results[model]['individual'][0].keys():
            values = [sim[metric] for sim in results[model]['individual'] if sim[metric] is not None]
            if values:
                results[model]['aggregate'][metric] = {
                    'mean': statistics.mean(values),
                    'min': min(values),
                    'max': max(values),
                    'std_dev': statistics.stdev(values) if len(values) > 1 else None
                }

    return results

def main():
    torch.autograd.set_detect_anomaly(True)

    configurations = [
        (4, 2), (8, 3), (16, 4), (32, 5), (64, 6)
    ]

    all_results = {}

    for d, k in configurations:
        print(f"\nRunning experiment: d={d}, k={k}")
        results = run_experiment(d, k)
        all_results[f"d{d}_k{k}"] = results

    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir = os.path.join(project_root, 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, f"sparse_polynomials_{timestamp}.json"), 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {os.path.join(output_dir, f'sparse_polynomials_{timestamp}.json')}")

if __name__ == "__main__":
    main()