# Freenets Function Approximation Experiments

This repository contains a collection of experiments that explore the ability of different neural network architectures to approximate various mathematical functions, including monomials, sparse polynomials, trigonometric polynomials, and interval-defined polynomials.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Getting Started](#getting-started)
- [Experiments](#experiments)
  - [Monomials](#monomials)
  - [Sparse Polynomials](#sparse-polynomials)
  - [Trigonometric Polynomials](#trigonometric-polynomials)
  - [Interval Sparse Polynomials](#interval-sparse-polynomials)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

## Introduction

The primary goal of these experiments is to investigate how different neural network architectures perform in approximating various mathematical functions. Specifically, we compare the performance of:

- **FreeNet**: A custom neural network architecture designed for function approximation.
- **MLP**: A standard Multi-Layer Perceptron with ReLU activation functions.
- **MLP_SqReLU**: An MLP with squared ReLU activation functions.

We generate datasets based on different types of functions and train the neural networks to learn these functions from data. The experiments aim to evaluate the models' ability to generalize and accurately approximate the target functions.

## Project Structure

The repository is organized as follows:

- `experiments/`: Contains the main scripts to run the experiments.
  - `monomials.py`
  - `sparse_polynomials.py`
  - `trig_polynomials.py`
  - `interval_polynomials.py`
- `models/`: Contains the implementations of the neural network architectures.
  - `freenet.py`
  - `mlp.py`
  - `mlp_sqrelu.py`
- `data_generators/`: Scripts for generating datasets.
  - `monomials.py`
  - `sparse_polynomials.py`
  - `trig_polynomials.py`
  - `interval_sparse_polynomials.py`
- `utilities/`: Utility functions for data handling, training, testing, and plotting.
  - `data_utilities.py`
  - `general_utilities.py`
  - `NumpyEncoder.py`
- `trainers/`: Training scripts for neural networks.
  - `train_nn.py`
- `testers/`: Testing and evaluation scripts.
  - `test_nn.py`
- `optimizers/`: Optimizer configurations.
  - `optimizer_params.py`
- `plotters/`: Scripts for generating plots of results.
  - `monomial_plotter.py`
  - `sparse_polynomial_plotter.py`
  - `trig_polynomial_plotter.py`
  - `interval_sparse_polynomial_plotter.py`
- `figures/`: Directory where generated plots are saved.
- `outputs/`: Directory where experiment results are saved as JSON files.

## Dependencies

The code is written in Python and relies on the following libraries:

- Python 3.7+
- NumPy
- PyTorch
- Matplotlib

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

**Note**: Ensure that you have a compatible version of PyTorch installed. You can install PyTorch by following the instructions on the [official website](https://pytorch.org/get-started/locally/).

## Getting Started

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. **Set up a virtual environment (optional but recommended)**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Experiments

Each experiment can be run independently from the base directory. Navigate to the base directory (the root of the repository) and execute the desired experiment script using Python.

### Monomials

**Script**: `experiments/monomials.py`

**Description**: Trains neural networks to approximate monomial functions of the form \( y = c x^d \).

**Usage**:

```bash
python experiments/monomials.py
```

**Configurations**:

The script runs experiments over various configurations defined in the `configurations` list within the `main()` function.

### Sparse Polynomials

**Script**: `experiments/sparse_polynomials.py`

**Description**: Trains neural networks to approximate sparse polynomials, i.e., polynomials with a small number of non-zero coefficients.

**Usage**:

```bash
python experiments/sparse_polynomials.py
```

### Trigonometric Polynomials

**Script**: `experiments/trig_polynomials.py`

**Description**: Trains neural networks to approximate trigonometric polynomials composed of sine and cosine functions.

**Usage**:

```bash
python experiments/trig_polynomials.py
```

### Interval Sparse Polynomials

**Script**: `experiments/interval_polynomials.py`

**Description**: Trains neural networks to approximate polynomials defined over specific intervals using smooth transitions (e.g., sigmoid functions) at the boundaries.

**Usage**:

```bash
python experiments/interval_polynomials.py
```

## Results

- **Outputs**: The experiment results are saved in the `outputs/` directory as JSON files with timestamps.
- **Figures**: Plots of the true functions versus the neural network approximations are saved in the `figures/` directory, organized by experiment.
