import os
import sys
from pathlib import Path
import datetime
import json
import math

# Add project root to system path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import statistics
import torch
from torch.utils.data import TensorDataset, DataLoader
from models.freenet import FreeNetTrig as FreeNet
from models.mlp import MLPTrig as MLP
from models.mlp_sqrelu import MLP_SqReLUTrig as MLP_SqReLU
from data_generators.trig_polynomials import generate_sparse_fourier_data
from utilities.data_utilities import split_data, NumpyEncoder
from utilities.general_utilities import set_random_seed, get_device
from trainers.train_nn import train_model
from testers.test_nn import test_model
from optimizers.optimizer_params import get_optimizer_and_scheduler
from plotters.trig_polynomial_plotter import save_sparse_fourier_polynomial_plots

import torch
import numpy as np

def inference_function(model, coefficients, degrees, is_sine, num_points=1000, device=torch.device('cpu')):
    # Generate x values on CPU
    x = np.linspace(0, 2*np.pi, num_points).reshape(-1, 1)
    x_tensor = torch.from_numpy(x).float().to(device)
    
    # Ground truth calculation on GPU
    y_true = torch.zeros_like(x_tensor)
    for coef, degree, sine in zip(coefficients, degrees, is_sine):
        if sine:
            y_true += coef * torch.sin(degree * x_tensor)
        else:
            y_true += coef * torch.cos(degree * x_tensor)
    
    # Model predictions
    model.eval()
    with torch.no_grad():
        y_pred = model(x_tensor)
    
    # Move results back to CPU for further processing or plotting
    return x, y_true.cpu().numpy(), y_pred.cpu().numpy()


def run_experiment(n=5, k=2, hidden_dim_freenet=8, hidden_dim_mlp=None, num_sims=5):
    device = get_device()
    # device = 'cpu' # cpu is actually faster for training!
    print(f"Using device: {device}")
    
    if hidden_dim_mlp is None:
        hidden_dim_mlp = [int(hidden_dim_freenet/2), int(hidden_dim_freenet/2)]
    
    results = {
        'FreeNet': {'aggregate': {}, 'individual': []},
        'MLP': {'aggregate': {}, 'individual': []},
        'MLP_SqReLU': {'aggregate': {}, 'individual': []}
    }

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

        # Generate data (on CPU)
        x, y, coefficients, degrees, is_sine = generate_sparse_fourier_data(n, k, num_data_points)

        # Split data
        x_train, x_test, y_train, y_test = split_data(x, y, test_size=percentage_test_split)
        
        # Convert to PyTorch tensors
        x_train = torch.from_numpy(x_train).float()
        y_train = torch.from_numpy(y_train).float()
        x_test = torch.from_numpy(x_test).float()
        y_test = torch.from_numpy(y_test).float()

        # Create data loaders
        train_dataset = TensorDataset(x_train.to(device), y_train.to(device))
        test_dataset = TensorDataset(x_test.to(device), y_test.to(device))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # Initialize models
        input_dim = 1
        output_dim = 1

        freenet_model = FreeNet(input_dim, hidden_dim_freenet, output_dim).to(device)
        mlp_model = MLP(input_dim, hidden_dim_mlp, output_dim).to(device)
        mlp_sqrelu_model = MLP_SqReLU(input_dim, hidden_dim_mlp, output_dim).to(device)

        # Get optimizers and schedulers
        freenet_optimizer, freenet_scheduler = get_optimizer_and_scheduler(freenet_model, optimizer_name, learning_rate)
        mlp_optimizer, mlp_scheduler = get_optimizer_and_scheduler(mlp_model, optimizer_name, learning_rate)
        mlp_sqrelu_optimizer, mlp_sqrelu_scheduler = get_optimizer_and_scheduler(mlp_sqrelu_model, optimizer_name, learning_rate)

        # Train models
        print("Training FreeNet:")
        # Train models
        print("Training FreeNet:")
        trained_freenet = train_model(freenet_model, train_loader, num_epochs, freenet_optimizer, freenet_scheduler)
        print("Training MLP:")
        trained_mlp = train_model(mlp_model, train_loader, num_epochs, mlp_optimizer, mlp_scheduler)
        print("Training MLP Sq Relu:")
        trained_mlp_sqrelu = train_model(mlp_sqrelu_model, train_loader, num_epochs, mlp_sqrelu_optimizer, mlp_sqrelu_scheduler)

        # Test models
        inference_func = lambda model, num_points: inference_function(model, coefficients, degrees, is_sine, num_points, device)

        print("Testing FreeNet:")
        freenet_results = test_model(trained_freenet, test_loader, inference_func)
        print("\nTesting MLP:")
        mlp_results = test_model(trained_mlp, test_loader, inference_func)
        print("\nTesting MLP Square Relu:")
        mlp_sqrelu_results = test_model(trained_mlp_sqrelu, test_loader, inference_func)

        # Store results
        results['FreeNet']['individual'].append(freenet_results)
        results['MLP']['individual'].append(mlp_results)
        results['MLP_SqReLU']['individual'].append(mlp_sqrelu_results)

        # Generate plots
        x_plot = np.linspace(0, 2*np.pi, 1000).reshape(-1, 1)
        y_true = np.zeros_like(x_plot)
        for coef, degree, sine in zip(coefficients, degrees, is_sine):
            if sine:
                y_true += coef * np.sin(degree * x_plot)
            else:
                y_true += coef * np.cos(degree * x_plot)
        
        y_pred_dict = {
            'FreeNet': trained_freenet(torch.FloatTensor(x_plot).to(device)).cpu().detach().numpy(),
            'MLP': trained_mlp(torch.FloatTensor(x_plot).to(device)).cpu().detach().numpy(),
            'MLP_SqReLU': trained_mlp_sqrelu(torch.FloatTensor(x_plot).to(device)).cpu().detach().numpy()
        }

        figures_dir = os.path.join(project_root, 'figures', 'sparse_fourier_polynomials', f'n{n}_k{k}_sim{sim+1}')
        save_sparse_fourier_polynomial_plots(x_plot, y_true, y_pred_dict, coefficients, degrees, is_sine, figures_dir)

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
    torch.autograd.set_detect_anomaly(False)
    num_sims = 1
    
    # n, k, hidden_dim_freenet, hidden_dim_mlp
    configurations = [
        (4, 2, 2, [1,1]), 
        (4, 3, 4, [2,2]), 
        (4, 4, 6, [3,3]), 
        (8, 2, 6, [3,3]), 
        (8, 3, 8, [4,4]), 
        (8, 4, 12, [6,6]), 
        (16, 2, 8, [4,4]), 
        (16, 3, 12, [6,6]), 
        (16, 4, 16, [8,8]), 
        (32, 2, 10, [5,5]), 
        (32, 3, 16, [8,8]), 
        (32, 4, 20, [10,10]),
    ]
    configurations = [
        (16, 2, 8, [4,4]), 
    ]
    all_results = {}

    for n, k, h_free, h_mlp in configurations:
        print(f"\nRunning experiment: n={n}, k={k}, freenet hdim = {h_free}, mlp hdim = {h_mlp}")
        results = run_experiment(n, k, h_free, h_mlp, num_sims=num_sims)
        all_results[f"n{n}_k{k}"] = results

    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir = os.path.join(project_root, 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"sparse_fourier_polynomials_{timestamp}.json"), 'w') as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    
    print(f"\nResults saved to {os.path.join(output_dir, f'sparse_fourier_polynomials_{timestamp}.json')}")

if __name__ == "__main__":
    main()