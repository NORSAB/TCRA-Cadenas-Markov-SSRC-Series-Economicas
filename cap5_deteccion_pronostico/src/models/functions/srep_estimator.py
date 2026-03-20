import numpy as np
from scipy.optimize import nnls

def srep_estimator(data, num_samples):
    """
    Estimates a column-stochastic transition matrix using NNLS (Standard Representation).
    Estimador de matriz de transición estocástica por columnas usando NNLS.
    
    Args:
        data (np.array): One-hot encoded time series data (k x T).
        num_samples (int): The number of time steps for estimation (T-1).

    Returns:
        np.array: The estimated column-stochastic transition matrix Pr (k x k).
    """
    num_states = data.shape[0]

    # 1. Construct the design matrix S0 and target vector S1
    S0_data = data[:, :num_samples]
    # np.kron for state duplication
    S0 = np.kron(S0_data, np.identity(num_states)).T
    S1 = S0.T @ (data[:, 1:(1 + num_samples)].T).reshape(num_samples * num_states)

    # 2. Construct the sum-to-one constraint matrix C
    C = np.kron(np.identity(num_states), np.ones((1, num_states)))

    # 3. Build the augmented matrix Mr
    Mr = np.zeros((num_states**2 + num_states, num_states**2))
    Mr[:num_states**2, :] = S0.T @ S0
    Mr[num_states**2:, :] = C

    # 4. Build the augmented right-hand side vector rhs
    rhs = np.zeros((num_states**2 + num_states))
    rhs[:num_states**2] = S1
    rhs[num_states**2:] = 1

    # 5. Solve the NNLS problem
    p_flat = nnls(Mr, rhs)[0]
    c = p_flat.reshape(-1, 1)

    # 6. Reshape and normalize to enforce stochasticity
    Pr = c.reshape(num_states, num_states).T
    col_sums = Pr.sum(axis=0)
    col_sums[col_sums == 0] = 1
    Pr = Pr / col_sums

    return Pr
