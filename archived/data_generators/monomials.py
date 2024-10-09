import numpy as np

def generate_monomial_data(degree, num_data_points, coefficient=1):
    x = np.random.uniform(-1, 1, num_data_points)
    if not coefficient:
        coefficient = np.random.uniform(-1, 1)
    y = coefficient * x ** degree
    return x.reshape(-1, 1), y.reshape(-1, 1), coefficient
