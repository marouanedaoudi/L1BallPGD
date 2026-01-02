import numpy as np

def project_l1_ball(v: np.ndarray, t: float) -> np.ndarray:
    """
    Project vector v onto the L1 ball of radius t.
    
    Solves: argmin_z ||z - v||_2^2  s.t. ||z||_1 <= t
    
    Algorithm from Duchi et al. (2008).
    
    Parameters
    ----------
    v : np.ndarray, shape (p,)
        Input vector
    t : float
        L1 ball radius (must be positive)
        
    Returns
    -------
    z : np.ndarray, shape (p,)
        Projection of v onto the L1 ball
    """
    if t <= 0:
        raise ValueError("Radius t must be positive")
    
    # If already in the ball, return as is
    if np.sum(np.abs(v)) <= t:
        return v.copy()
    
    # Sort absolute values in descending order
    u = np.abs(v)
    u_sorted = np.sort(u)[::-1]
    
    # Find threshold theta
    cumsum = np.cumsum(u_sorted)
    rho = np.where(u_sorted > (cumsum - t) / (np.arange(len(u)) + 1))[0][-1]
    theta = (cumsum[rho] - t) / (rho + 1)
    
    # Apply soft thresholding
    z = np.sign(v) * np.maximum(u - theta, 0)
    
    return z
