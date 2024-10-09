import numpy as np

def generate_sparse_polynomial_data(d, k, num_data_points):
    """
    Generate data for a k-sparse polynomial of degree up to d.
    
    :param d: Maximum degree of the polynomial
    :param k: Number of non-zero terms in the polynomial
    :param num_data_points: Number of data points to generate
    :return: x, y, coefficients, degrees
    """
    x = np.random.uniform(0, 1, (num_data_points, 1))
    
    # Generate the highest degree term
    degrees = [d]
    
    # Generate k-1 more terms with degrees from 1 to d-1
    degrees.extend(np.random.randint(1, d, size=k-1))
    
    # Generate random coefficients
    coefficients = np.random.uniform(-1, 1, size=k)
    
    # Compute y
    y = np.zeros_like(x)
    for coef, degree in zip(coefficients, degrees):
        y += coef * x**degree
    
    return x, y, coefficients, degrees