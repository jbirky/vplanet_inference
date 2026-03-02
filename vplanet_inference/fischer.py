import numpy as np
import pandas as pd
from typing import Callable, Optional, Union

__all__ = ["compute_fisher_information", "analyze_fisher_information"]

def compute_fisher_information(
    log_likelihood: Callable,
    params: np.ndarray,
    data: Union[np.ndarray, tuple],
    method: str = 'hessian',
    epsilon: float = 1e-5,
    per_observation: bool = False
) -> np.ndarray:
    """
    Compute the Fisher Information Matrix for a black-box likelihood function.
    
    Parameters
    ----------
    log_likelihood : callable
        Function that computes log-likelihood. 
        Should have signature: log_likelihood(params, data) -> float or array
        - If per_observation=False: returns scalar total log-likelihood
        - If per_observation=True: returns array of per-observation log-likelihoods
    params : np.ndarray
        Parameter values at which to evaluate Fisher information (shape: (n_params,))
    data : np.ndarray or tuple
        The observed data to pass to log_likelihood function
    method : str, default='hessian'
        Method to compute Fisher information:
        - 'hessian': Use negative Hessian of log-likelihood (observed FI)
        - 'score': Use outer product of score vectors (requires per_observation=True)
    epsilon : float, default=1e-5
        Step size for finite difference approximation
    per_observation : bool, default=False
        Whether log_likelihood returns per-observation values
        
    Returns
    -------
    fisher_info : np.ndarray
        Fisher Information Matrix (shape: (n_params, n_params))
        
    Examples
    --------
    >>> # Simple example: Normal distribution with unknown mean
    >>> def log_lik(params, data):
    ...     mu = params[0]
    ...     sigma = 1.0
    ...     return -0.5 * np.sum((data - mu)**2) / sigma**2
    >>> 
    >>> data = np.random.randn(100)
    >>> params = np.array([0.0])
    >>> FI = compute_fisher_information(log_lik, params, data)
    >>> print(f"Fisher Information: {FI[0,0]:.2f}")
    >>> print(f"Standard Error: {1/np.sqrt(FI[0,0]):.3f}")
    """
    params = np.atleast_1d(params)
    n_params = len(params)
    
    if method == 'hessian':
        return _fisher_from_hessian(log_likelihood, params, data, epsilon)
    elif method == 'score':
        if not per_observation:
            raise ValueError("method='score' requires per_observation=True")
        return _fisher_from_scores(log_likelihood, params, data, epsilon)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'hessian' or 'score'")


def _fisher_from_hessian(
    log_likelihood: Callable,
    params: np.ndarray,
    data: Union[np.ndarray, tuple],
    epsilon: float
) -> np.ndarray:
    """
    Compute observed Fisher Information as negative Hessian of log-likelihood.
    Uses central finite differences for numerical stability.
    """
    n_params = len(params)
    hessian = np.zeros((n_params, n_params))
    
    # Compute Hessian using central finite differences
    for i in range(n_params):
        for j in range(i, n_params):
            # Create perturbation vectors
            ei = np.zeros(n_params)
            ej = np.zeros(n_params)
            ei[i] = epsilon
            ej[j] = epsilon
            
            if i == j:
                # Diagonal elements: second derivative
                f_plus = log_likelihood(params + ei, data)
                f_center = log_likelihood(params, data)
                f_minus = log_likelihood(params - ei, data)
                
                hessian[i, i] = (f_plus - 2*f_center + f_minus) / (epsilon**2)
            else:
                # Off-diagonal elements: mixed partial derivatives
                f_pp = log_likelihood(params + ei + ej, data)
                f_pm = log_likelihood(params + ei - ej, data)
                f_mp = log_likelihood(params - ei + ej, data)
                f_mm = log_likelihood(params - ei - ej, data)
                
                hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * epsilon**2)
                hessian[j, i] = hessian[i, j]  # Symmetry
    
    # Fisher Information is negative of Hessian
    fisher_info = -hessian
    
    # Ensure positive semi-definite (numerical errors can violate this)
    fisher_info = _make_positive_semidefinite(fisher_info)
    
    return fisher_info


def _fisher_from_scores(
    log_likelihood: Callable,
    params: np.ndarray,
    data: Union[np.ndarray, tuple],
    epsilon: float
) -> np.ndarray:
    """
    Compute Fisher Information as outer product of score vectors.
    Requires per-observation log-likelihoods.
    """
    n_params = len(params)
    
    # Get per-observation log-likelihoods
    log_liks = log_likelihood(params, data)
    if np.isscalar(log_liks):
        raise ValueError("log_likelihood must return array when using score method")
    
    log_liks = np.atleast_1d(log_liks)
    n_obs = len(log_liks)
    
    # Compute gradient (score) for each observation
    scores = np.zeros((n_obs, n_params))
    
    for k in range(n_obs):
        # Define single-observation log-likelihood
        def single_obs_log_lik(p):
            all_liks = log_likelihood(p, data)
            return all_liks[k]
        
        # Compute gradient using central differences
        for i in range(n_params):
            ei = np.zeros(n_params)
            ei[i] = epsilon
            
            grad_i = (single_obs_log_lik(params + ei) - 
                     single_obs_log_lik(params - ei)) / (2 * epsilon)
            scores[k, i] = grad_i
    
    # Fisher Information = sum of outer products
    fisher_info = np.zeros((n_params, n_params))
    for k in range(n_obs):
        fisher_info += np.outer(scores[k], scores[k])
    
    fisher_info = _make_positive_semidefinite(fisher_info)
    
    return fisher_info


def _make_positive_semidefinite(matrix: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """
    Ensure matrix is positive semi-definite by clipping negative eigenvalues.
    This handles numerical errors in finite difference approximations.
    """
    # Check if already symmetric
    if not np.allclose(matrix, matrix.T):
        matrix = (matrix + matrix.T) / 2
    
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.maximum(eigenvalues, tol)
    
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


def analyze_fisher_information(
    fisher_info: np.ndarray,
    param_names: Optional[list] = None
) -> dict:
    """
    Analyze Fisher Information Matrix to extract useful statistics.
    
    Parameters
    ----------
    fisher_info : np.ndarray
        Fisher Information Matrix
    param_names : list, optional
        Names of parameters for reporting
        
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'standard_errors': Approximate parameter standard errors
        - 'correlation_matrix': Parameter correlation matrix
        - 'condition_number': Condition number (higher = more numerical issues)
        - 'determinant': Determinant (volume of confidence ellipsoid)
    """
    n_params = fisher_info.shape[0]
    
    if param_names is None:
        param_names = [f"θ_{i}" for i in range(n_params)]
    
    # Covariance matrix (inverse of Fisher Information)
    try:
        covariance = np.linalg.inv(fisher_info)
    except np.linalg.LinAlgError:
        print("Warning: Fisher Information is singular, using pseudo-inverse")
        covariance = np.linalg.pinv(fisher_info)
    
    # Standard errors
    standard_errors = np.sqrt(np.diag(covariance))
    
    # Correlation matrix
    std_matrix = np.outer(standard_errors, standard_errors)
    correlation = covariance / std_matrix
    
    # Condition number
    eigenvalues = np.linalg.eigvalsh(fisher_info)
    condition_number = np.max(eigenvalues) / max(np.min(eigenvalues), 1e-10)
    
    # Determinant
    determinant = np.linalg.det(fisher_info)
    
    results = {
        'standard_errors': dict(zip(param_names, standard_errors)),
        'correlation_matrix': correlation,
        'condition_number': condition_number,
        'determinant': determinant,
        'eigenvalues': eigenvalues
    }
    
    return results