import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re


def draw_monomial_plot(x, y_true, y_pred, model_name, degree, coefficient, save_path=None):
    # Set the style and color palette
    sns.set_style("whitegrid")
    sns.set_palette("deep")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot the ground truth
    sns.lineplot(x=x, y=y_true, label='Ground Truth', color='navy', linewidth=2)
    
    # Plot the prediction
    # cleaned_model_name = re.sub(r"_sim\d+", "", model_name)
    sns.lineplot(x=x, y=y_pred, label=f'{model_name} Prediction', color='crimson', linewidth=2, linestyle='--')
    
    # Fill the area between the curves
    plt.fill_between(x, y_true, y_pred, alpha=0.2, color='lightblue')
    
    # Set the title and labels
    plt.title(f'Monomial Function: {coefficient:.4f}x^{degree}', fontsize=20, fontweight='bold')
    plt.xlabel('x', fontsize=16)
    plt.ylabel('y', fontsize=16)
    
    # Customize the legend
    # plt.legend(fontsize=14, loc='best', bbox_to_anchor=(1, 1), bbox_transform=plt.gca().transAxes)
    plt.legend(fontsize=14, loc='best')
    # Customize the tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    # Add a text box with model performance
    mse = np.mean((y_true - y_pred)**2)
    mae = np.mean(np.abs(y_true - y_pred))
    textstr = f'MSE: {mse:.4e}\nMAE: {mae:.4e}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=props)
    
    # Adjust the layout and display or save the plot
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()