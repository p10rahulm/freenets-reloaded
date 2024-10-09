import numpy as np

def generate_sparse_fourier_data(n, k, num_data_points):
    """
    Generate data for a k-sparse Fourier polynomial with terms up to degree n.
    
    :param n: Maximum degree of the Fourier terms
    :param k: Number of non-zero terms in the polynomial
    :param num_data_points: Number of data points to generate
    :return: x, y, coefficients, degrees, is_sine
    """
    x = np.random.uniform(0, 2*np.pi, (num_data_points, 1))
    
    # Generate k terms with degrees from 1 to n
    degrees = np.random.randint(1, n+1, size=k)
    
    # Generate random coefficients
    coefficients = np.random.uniform(-1, 1, size=k)
    
    # Randomly choose between sine and cosine for each term
    is_sine = np.random.choice([True, False], size=k)
    
    # Compute y
    y = np.zeros_like(x)
    for coef, degree, sine in zip(coefficients, degrees, is_sine):
        if sine:
            y += coef * np.sin(degree * x)
        else:
            y += coef * np.cos(degree * x)
    
    return x, y, coefficients, degrees, is_sine