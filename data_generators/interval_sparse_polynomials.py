import numpy as np

def generate_interval_sparse_polynomial_data(d, k, num_data_points, interval_start, interval_end):
    """
    Generate data for a k-sparse polynomial of degree up to d over a specific interval.
    
    :param d: Maximum degree of the polynomial
    :param k: Number of non-zero terms in the polynomial
    :param num_data_points: Number of data points to generate
    :param interval_start: Start of the interval where the polynomial is non-zero
    :param interval_end: End of the interval where the polynomial is non-zero
    :return: x, y, coefficients, degrees, interval
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
    mask = (x >= interval_start) & (x <= interval_end)
    for coef, degree in zip(coefficients, degrees):
        y[mask] += coef * x[mask]**degree
    
    return x, y, coefficients, degrees, (interval_start, interval_end)
