import numpy as np
from numba import njit, float32, int32
from typing import Tuple, Optional

# Constants for better readability in the return tuple
# 0: Time, 1: S, 2: I, 3: R

@njit(fastmath=True)
def gillespie_step(
    S: float, 
    I: float, 
    R: float, 
    beta: float, 
    gamma: float, 
    N: float
) -> Tuple[float, float, float, float]:
    """
    Performs one step of the Gillespie Direct Method (Exact Stochastic Simulation).
    
    Physics:
    - Events are discrete (integer jumps).
    - Time between events is continuous (exponentially distributed).
    """
    
    # 1. Calculate Reaction Propensities (Rates)
    rate_infection = beta * S * I / N
    rate_recovery = gamma * I
    rate_total = rate_infection + rate_recovery

    # Safety: If no events can happen (e.g., I=0), return infinite time
    if rate_total <= 1e-12:
        return 1e8, S, I, R 

    # 2. Determine Time to Next Event (dt ~ Exp(rate_total))
    dt = -np.log(np.random.random()) / rate_total
    
    # 3. Determine Which Event Occurred
    # We use a single random number to select the event (Roulette Wheel selection)
    if np.random.random() * rate_total < rate_infection:
        # Infection Event: S -> I
        S -= 1.0
        I += 1.0
    else:
        # Recovery Event: I -> R
        I -= 1.0
        R += 1.0

    return dt, S, I, R

@njit
def simulate_epidemic(
    beta: float, 
    gamma: float, 
    N: float = 1000.0, 
    I0: float = 1.0, 
    max_time: float = 50.0,
    seed: int = -1
) -> np.ndarray:
    """
    Simulates one full stochastic epidemic trajectory.
    Args:
        seed: If -1, uses random entropy. If >=0, ensures reproducibility.
    Returns:
        Raw event history array of shape (Steps, 4) -> [Time, S, I, R]
    """
    if seed >= 0:
        np.random.seed(seed)
        
    S, I, R = N - I0, I0, 0.0
    t = 0.0
    
    # Pre-allocate memory (over-estimate size to avoid dynamic resizing)
    # A standard epidemic will have at most 2*N events (everyone gets infected + recovers)
    max_steps = int(2.5 * N) 
    history = np.zeros((max_steps, 4), dtype=np.float32)
    
    step = 0
    history[step] = [t, S, I, R]
    
    while t < max_time and I > 0 and step < max_steps - 1:
        dt, S, I, R = gillespie_step(S, I, R, beta, gamma, N)
        t += dt
        step += 1
        history[step] = [t, S, I, R]
        
    # Return valid portion of array
    return history[:step+1]

@njit
def interpolate_trajectory(
    raw_data: np.ndarray, 
    t_eval: np.ndarray, 
    N: float
) -> np.ndarray:
    """
    Research Optimization:
    Interpolates the raw Gillespie steps onto a fixed time grid (t_eval).
    
    Why: Neural ODEs require fixed time steps, but Gillespie produces variable ones.
    Doing this in Numba is ~50x faster than doing it in the Python training loop.
    
    Returns:
        Normalized trajectory [S, I, R] of shape (len(t_eval), 3)
    """
    # raw_data columns: 0=Time, 1=S, 2=I, 3=R
    raw_t = raw_data[:, 0]
    raw_s = raw_data[:, 1]
    raw_i = raw_data[:, 2]
    raw_r = raw_data[:, 3]
    
    # Numba supports np.interp (linear interpolation)
    s_interp = np.interp(t_eval, raw_t, raw_s)
    i_interp = np.interp(t_eval, raw_t, raw_i)
    r_interp = np.interp(t_eval, raw_t, raw_r)
    
    # Stack and Normalize by Population N
    # Shape: (len(t_eval), 3)
    result = np.stack((s_interp, i_interp, r_interp), axis=1) / N
    
    return result.astype(np.float32)