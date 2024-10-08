import matplotlib.pyplot as plt

def draw_monomial_plot(x, y_true, y_pred, model_name, degree, coefficient):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_true, label='Ground Truth', color='blue')
    plt.plot(x, y_pred, label=f'{model_name} Prediction', color='red', linestyle='--')
    plt.title(f'Monomial Function: {coefficient}x^{degree}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{model_name}_monomial_plot_degree_{degree}.png')
    plt.close()
    

