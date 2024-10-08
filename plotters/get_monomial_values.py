import numpy as np
import torch

def get_plotter_values(model, degree, coefficient, num_points=100):
    x = np.linspace(0, 1, num_points)
    
    # Ground truth
    y_true = coefficient * x**degree
    
    # Model predictions
    model.eval()
    with torch.no_grad():
        x_tensor = torch.from_numpy(x).float().unsqueeze(1)
        y_pred = model(x_tensor).numpy().flatten()
    
    return x, y_true, y_pred