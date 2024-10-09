import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def compute_lp_metrics(y_true, y_pred, num_points=1000):
    diff = np.abs(y_true - y_pred)
    dx = 1 / num_points
    l1 = np.sum(diff) * dx
    l2 = np.sqrt(np.sum(diff**2) * dx)
    l3 = np.power(np.sum(diff**3) * dx, 1/3)
    l_inf = np.max(diff)
    return l1, l2, l3, l_inf

def test_model(model, test_loader, inference_function, get_test_metrics=True, get_distance_metrics=True, num_points=1000):
    model.eval()
    results = {
        'test_MSE': None, 'test_MAE': None, 'test_RMSE': None,
        'L1': None, 'L2': None, 'L3': None, 'L_inf': None
    }
    
    if get_test_metrics:
        y_true, y_pred = [], []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                outputs = model(x_batch.float())
                y_true.extend(y_batch.numpy())
                y_pred.extend(outputs.numpy())
        
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        
        results['test_MSE'] = mean_squared_error(y_true, y_pred)
        results['test_RMSE'] = np.sqrt(results['test_MSE'])
        results['test_MAE'] = mean_absolute_error(y_true, y_pred)
    
    if get_distance_metrics:
        x_plot, y_true, y_pred = inference_function(model, num_points)
        results['L1'], results['L2'], results['L3'], results['L_inf'] = compute_lp_metrics(y_true, y_pred, num_points=num_points)
    
    # Print results
    print("Test Results:")
    for metric, value in results.items():
        if value is not None:
            print(f"{metric}: {value:.6f}")
    
    return results