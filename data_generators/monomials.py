import numpy as np

def generate_monomial_data(degree, num_data_points):
    x = np.random.uniform(-1, 1, num_data_points)
    coefficient = np.random.uniform(-1, 1)
    y = coefficient * x ** degree
    return x.reshape(-1, 1), y.reshape(-1, 1)
