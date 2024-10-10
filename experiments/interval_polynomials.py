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
from models.freenet import FreeNet
from models.mlp import MLP
from models.mlp_sqrelu import MLP_SqReLU
# from data_generators.sparse_polynomials import generate_sparse_polynomial_data
from data_generators.interval_sparse_polynomials import generate_interval_sparse_polynomial_data as generate_sparse_polynomial_data
from utilities.data_utilities import split_data, NumpyEncoder
from utilities.general_utilities import set_random_seed, get_device
from trainers.train_nn import train_model
from testers.test_nn import test_model
from optimizers.optimizer_params import get_optimizer_and_scheduler
from plotters.sparse_polynomial_plotter import save_sparse_polynomial_plots
from plotters.interval_sparse_polynomial_plotter import save_interval_sparse_polynomial_plots

import torch
import numpy as np


def inference_function(model, coefficients, degrees, interval, epsilon=3e-2, num_points=1000, device=torch.device('cpu')):
    # Generate x values
    x = np.linspace(0, 1, num_points).reshape(-1, 1)
    x_tensor = torch.from_numpy(x).float().to(device)
    
    # Compute the polynomial values using torch tensors
    polynomial_values = torch.zeros_like(x_tensor)
    for coef, degree in zip(coefficients, degrees):
        polynomial_values += coef * x_tensor.pow(degree)
    
    # Create smooth masks for the interval boundaries
    def smooth_transition(x, x0, direction='rising'):
        sharpness = 10 / epsilon  # Adjust sharpness based on epsilon
        if direction == 'rising':
            return 1 / (1 + torch.exp(-sharpness * (x - (x0 - epsilon))))
        elif direction == 'falling':
            return 1 / (1 + torch.exp(sharpness * (x - x0)))
        else:
            raise ValueError("Invalid direction. Use 'rising' or 'falling'.")
    
    # Compute the mask using torch tensors
    mask = torch.ones_like(x_tensor)
    mask *= smooth_transition(x_tensor, interval[0], direction='rising')
    mask *= smooth_transition(x_tensor, interval[1], direction='falling')
    
    # Apply the mask to the polynomial values to get y_true
    y_true = mask * polynomial_values
    
    # Model predictions
    model.eval()
    with torch.no_grad():
        y_pred = model(x_tensor)
    
    # Move results back to CPU for further processing or plotting
    return x, y_true.cpu().numpy(), y_pred.cpu().numpy()


def run_experiment(d=16, k=2, interval_start=0.25, interval_end=0.75, hidden_dim_freenet = 8, hidden_dim_mlp=None, num_sims=5):
    device = get_device()
    device = 'cpu' # cpu is actually faster for training!
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
        num_epochs = 250
        learning_rate = 0.01
        optimizer_name = "adamw"
        percentage_test_split = 0.01

        # Generate data (on CPU)
        x, y, coefficients, degrees = generate_sparse_polynomial_data(d, k, num_data_points, interval_start, interval_end)

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
        trained_freenet = train_model(freenet_model, train_loader, num_epochs, freenet_optimizer, freenet_scheduler)
        print("Training MLP:")
        trained_mlp = train_model(mlp_model, train_loader, num_epochs, mlp_optimizer, mlp_scheduler)
        print("Training MLP Sq Relu:")
        trained_mlp_sqrelu = train_model(mlp_sqrelu_model, train_loader, num_epochs, mlp_sqrelu_optimizer, mlp_sqrelu_scheduler)

        # Test models
        # inference_func = lambda model, num_points: inference_function(model, coefficients, degrees, num_points, device)
        inference_func = lambda model, num_points: inference_function(model, coefficients, degrees, 
                                                                      [interval_start, interval_end], 
                                                                      epsilon=3e-2, num_points=num_points, device=torch.device('cpu'))


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
        x_plot, y_true, _ = inference_func(trained_freenet, 1000)

        
        y_pred_dict = {
            'FreeNet': trained_freenet(torch.FloatTensor(x_plot).to(device)).cpu().detach().numpy(),
            'MLP': trained_mlp(torch.FloatTensor(x_plot).to(device)).cpu().detach().numpy(),
            'MLP_SqReLU': trained_mlp_sqrelu(torch.FloatTensor(x_plot).to(device)).cpu().detach().numpy()
        }

        figures_dir = os.path.join(project_root, 'figures', 'interval_sparse_polynomials', 
                           f'd{d}_k{k}_int{interval_start:.2f}_{interval_end:.2f}_sim{sim+1}')
        save_interval_sparse_polynomial_plots(x_plot, y_true, y_pred_dict, coefficients, degrees, [interval_start, interval_end], figures_dir)


    # Compute aggregate metrics
    for model in results:
        for metric in results[model]['individual'][0].keys():
            values = [sim[metric] for sim in results[model]['individual'] if sim[metric] is not None]
            # Convert values to NumPy array
            values = np.array(values, dtype=np.float64)
            # Remove NaN and Inf values
            values = values[~np.isnan(values) & ~np.isinf(values)]
            if values:
                results[model]['aggregate'][metric] = {
                    'mean': np.mean(values),
                    'min': min(values),
                    'max': max(values),
                    'std_dev': np.std(values, ddof=1) if len(values) > 1 else None
                }

    return results

def main():
    torch.autograd.set_detect_anomaly(True)
    # d, k, hidden_dim_freenet, hidden_dim_mlp
    num_sims = 5
    # configurations = [
    #     (4, 2, 2, [1,1] 0.17, 0.79), 
    #     (4, 3, 4, [2,2], 0.23, 0.59), 
    #     (4, 4, 6, [3,3], 0.25, 0.75), 
    #     (8, 2, 6, [3,3], 0.25, 0.75), 
    #     (8, 3, 8, [4,4], 0.42, 0.83), 
    #     (8, 4, 12, [6,6], 0.35, 0.58), 
    #     (16, 2, 8, [4,4], 0.56, 0.87), 
    #     (16, 3, 12, [6,6], 0.23, 0.69), 
    #     (16, 4, 16, [8,8], 0.11, 0.54), 
    #     (32, 2, 10, [5,5], 0.25, 0.88), 
    #     (32, 3, 16, [8,8], 0.36, 0.72), 
    #     (32, 4, 20, [10,10], 0.38, 0.79),
    # ]
    configurations = [
        (4, 2, 10, [5,5], 0.17, 0.79), 
        (4, 3, 12, [6,6], 0.23, 0.59), 
        (4, 4, 14, [7,7], 0.25, 0.75), 
        (8, 2, 14, [7,7], 0.25, 0.75), 
        (8, 3, 16, [8,8], 0.42, 0.83), 
        (8, 4, 20, [10,10], 0.35, 0.58), 
        (16, 2, 16, [8,8], 0.56, 0.87), 
        (16, 3, 20, [10,10], 0.23, 0.69), 
        (16, 4, 24, [12,12], 0.11, 0.54), 
        (32, 2, 18, [9,9], 0.25, 0.88), 
        (32, 3, 24, [12,12], 0.36, 0.72), 
        (32, 4, 28, [14,14], 0.38, 0.79),
    ]
    # configurations = [
    #     (20, [10, 10], 16, 2, 0.17, 0.79),
    #     (24, [12, 12], 16, 3, 0.23, 0.59),
    #     (28, [14, 14], 16, 4, 0.25, 0.75),
    #     (24, [12, 12], 32, 2, 0.42, 0.83),
    #     (28, [14, 14], 32, 3, 0.01, 0.38),
    #     (32, [16, 16], 32, 4, 0.56, 0.87),
    #     (28, [14, 14], 64, 2, 0.23, 0.69),
    #     (32, [16, 16], 64, 3, 0.11, 0.54),
    #     (36, [18, 18], 64, 4, 0.38, 0.79),
    # ]
    
    all_results = {}

    for d, k, h_free, h_mlp, interval_start, interval_end in configurations:
        print(f"\nRunning experiment: d={d}, k={k}, freenet hdim = {h_free}, mlp hdim = {h_mlp}")
        results = run_experiment(d, k, interval_start=interval_start, interval_end=interval_end, hidden_dim_freenet = h_free, hidden_dim_mlp=h_mlp,num_sims=num_sims)
        all_results[f"d{d}_k{k}"] = results

    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir = os.path.join(project_root, 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"sparse_polynomials_{timestamp}.json"), 'w') as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    
    print(f"\nResults saved to {os.path.join(output_dir, f'sparse_polynomials_{timestamp}.json')}")

if __name__ == "__main__":
    main()