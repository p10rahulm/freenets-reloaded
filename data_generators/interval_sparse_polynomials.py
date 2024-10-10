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
    epsilon = 1e-2
    y = y + epsilon
    mask = (x >= interval_start) & (x <= interval_end)
    for coef, degree in zip(coefficients, degrees):
        y[mask] += coef * x[mask]**degree
    y = y + 1e-6
    x = np.clip(x, 1e-6, 1 - 1e-6)
    y = np.clip(y, -1e2, 1e2)

    return x, y, coefficients, degrees, (interval_start, interval_end)


import numpy as np

def generate_interval_sparse_polynomial_data(
    d, k, num_data_points, interval_start, interval_end, epsilon=3e-2):
    """
    Generate data for a k-sparse polynomial of degree up to d over a specific interval
    with smooth transitions at the interval boundaries.
    
    :param d: Maximum degree of the polynomial
    :param k: Number of non-zero terms in the polynomial
    :param num_data_points: Number of data points to generate
    :param interval_start: Start of the interval where the polynomial is active
    :param interval_end: End of the interval where the polynomial is active
    :param epsilon: Width of the transition zone for smoothness
    :return: x, y, coefficients, degrees, interval
    """
    x = np.random.uniform(0, 1, (num_data_points, 1))
    # Generate degrees and coefficients for the polynomial
    degrees = [d]  # Ensure the highest degree term is included
    degrees.extend(np.random.randint(1, d, size=k-1))  # Other degrees
    coefficients = np.random.uniform(-1, 1, size=k)
    
    # Compute the polynomial values
    polynomial_values = np.zeros_like(x)
    for coef, degree in zip(coefficients, degrees):
        polynomial_values += coef * x ** degree
    
    # Create smooth masks for the interval boundaries
    def smooth_transition(x, x0, direction='rising'):
        """
        Create a smooth transition using a sigmoid function.
        :param x: Input values
        :param x0: Transition point
        :param direction: 'rising' or 'falling'
        :return: Transition values between 0 and 1
        """
        # Control the sharpness of the transition
        sharpness = 10 / epsilon  # Adjust the denominator to control smoothness
        if direction == 'rising':
            return 1 / (1 + np.exp(-sharpness * (x - (x0 - epsilon))))
        elif direction == 'falling':
            return 1 / (1 + np.exp(sharpness * (x - x0)))
        else:
            raise ValueError("Invalid direction. Use 'rising' or 'falling'.")

    # Compute the smooth mask
    mask = np.ones_like(x)
    # Before interval_start
    mask *= smooth_transition(x, interval_start, direction='rising')
    # After interval_end
    mask *= smooth_transition(x, interval_end, direction='falling')
    
    # Apply the mask to the polynomial values
    y = mask * polynomial_values
    
    return x, y, coefficients, degrees
