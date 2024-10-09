import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def draw_sparse_polynomial_plot(x, y_true, y_pred_dict, coefficients, degrees, save_path):
    # Set the style and color palette
    sns.set_style("whitegrid")
    sns.set_palette("deep")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot the ground truth
    sns.lineplot(x=x.flatten(), y=y_true.flatten(), label='Ground Truth', color='navy', linewidth=2)
    
    # Plot the predictions
    colors = ['crimson', 'green', 'orange']
    for (model_name, y_pred), color in zip(y_pred_dict.items(), colors):
        sns.lineplot(x=x.flatten(), y=y_pred.flatten(), label=f'{model_name} Prediction', color=color, linewidth=2, linestyle='--')
    
    # Set the title and labels
    polynomial_str = ' + '.join([f'{coef:.2f}x^{deg}' for coef, deg in zip(coefficients, degrees)])
    plt.title(f'Sparse Polynomial: {polynomial_str}', fontsize=20, fontweight='bold')
    plt.xlabel('x', fontsize=16)
    plt.ylabel('y', fontsize=16)
    
    # Customize the legend
    plt.legend(fontsize=14, loc='best')
    
    # Customize the tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    # Add a text box with model performance
    textstr = ''
    for model_name, y_pred in y_pred_dict.items():
        mse = np.mean((y_true - y_pred)**2)
        mae = np.mean(np.abs(y_true - y_pred))
        textstr += f'{model_name}:\nMSE: {mse:.4e}\nMAE: {mae:.4e}\n\n'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=props)
    
    # Adjust the layout and save the plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()

def save_sparse_polynomial_plots(x, y_true, y_pred_dict, coefficients, degrees, base_path):
    # Create subfolders
    for subfolder in ['freenet', 'mlp', 'mlp_sqrelu', 'combined']:
        os.makedirs(os.path.join(base_path, subfolder), exist_ok=True)
    
    # Save individual plots
    for model_name, y_pred in y_pred_dict.items():
        save_path = os.path.join(base_path, model_name.lower(), f'{model_name.lower()}_plot.pdf')
        draw_sparse_polynomial_plot(x, y_true, {model_name: y_pred}, coefficients, degrees, save_path)
    
    # Save combined plot
    save_path = os.path.join(base_path, 'combined', 'combined_plot.pdf')
    draw_sparse_polynomial_plot(x, y_true, y_pred_dict, coefficients, degrees, save_path)