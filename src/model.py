import torch
import torch.nn as nn
from torchdiffeq import odeint

class SIRDerivativeNet(nn.Module):
    """
    Enforces Physical Constraints:
    1. Conservation of Mass (dS + dI + dR = 0)
    2. Non-negativity (Populations cannot be < 0)
    3. Monotonicity (S can only decrease)
    """
    def __init__(self):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.Tanh(),             
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 2)     
        )   
        # Initialize weights to small values to avoid exploding gradients early
        with torch.no_grad():
            self.net[-1].weight.fill_(0.0)
            self.net[-1].bias.fill_(0.0)

    def forward(self, t, y):
        # Physical Guard: Clamp states to [0, 1] to prevent halluncinating negative people
        y = torch.clamp(y, min=0.0, max=1.0)
        
        # S, I, R states
        S = y[:, 0:1]
        I = y[:, 1:2]
        
        d_out = self.net(y)
        
        # Physics Constraint 1: S can only decrease. 
        # We use -softplus to force dS/dt <= 0.
        # We also multiply by S*I because if either is 0, infection stops.
        d_S = -torch.nn.functional.softplus(d_out[:, 0:1]) * S * I
        
        # Physics Constraint 2: I's change depends on the infection gain and recovery loss.
        # We allow d_out[:, 1:2] to be the "net" change but scale by I to prevent negative I.
        d_I = d_out[:, 1:2] * I
        
        # Physics Constraint 3: Conservation of Mass.
        # dR is always the remainder to ensure the total sum of derivatives is 0.
        d_R = -(d_S + d_I)
        
        return torch.cat([d_S, d_I, d_R], dim=1)

class NeuralODE(nn.Module):
    def __init__(self, derivative_net, solver='rk4', step_size=0.1):
        super().__init__()
        self.derivative_net = derivative_net
        self.solver = solver
        self.step_size = step_size

    def forward(self, y0, t):
        y0 = y0.to(torch.float32)
        t = t.to(torch.float32)
        
        # We integrate the system using the physics-constrained derivative net
        return odeint(
            self.derivative_net, 
            y0, 
            t, 
            method=self.solver,
            options={'step_size': self.step_size}
        )