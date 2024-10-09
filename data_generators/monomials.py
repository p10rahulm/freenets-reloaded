import numpy as np

def generate_monomial_data(degree, num_data_points, coefficient=None):
    x = np.random.uniform(0, 1, (num_data_points, 1))
    
    if coefficient is None:
        coefficient = np.random.uniform(-1, 1)
    
    y = coefficient * x**degree
    
    return x, y, [coefficient], [degree]