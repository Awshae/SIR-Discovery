import torch
import numpy as np
from gplearn.genetic import SymbolicRegressor
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import SIRDerivativeNet, NeuralODE

def discover_with_gp():
    derivative_net = SIRDerivativeNet()
    model = NeuralODE(derivative_net, solver='rk4', step_size=0.1)
    
    try:
        path = "data/best_sir_model.pth"
        if not os.path.exists(path): 
            path = "data/final_sir_model.pth"
        derivative_net.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        print(f"Model loaded successfully from {path}")
    except FileNotFoundError:
        print("Model file not found. Run train.py first.")
        return

    model.eval()

    # Diverse starting points ensure the GP sees the 'full' physics of the manifold
    y0_list = [
        torch.tensor([[0.99, 0.01, 0.00]], dtype=torch.float32),
        torch.tensor([[0.70, 0.30, 0.00]], dtype=torch.float32),
        torch.tensor([[0.40, 0.10, 0.50]], dtype=torch.float32)
    ]
    t_eval = torch.linspace(0, 40, 100)
    
    all_states, all_derivatives = [], []
    with torch.no_grad():
        for y0 in y0_list:
            y_pred = model(y0, t_eval)
            s_tensor = y_pred.squeeze(1)
            # We use the constrained derivative_net to get clean gradient data
            d_tensor = derivative_net(0, s_tensor)
            all_states.append(s_tensor.numpy()[:, :2]) # Track S and I
            all_derivatives.append(d_tensor.numpy()[:, :2])

    X = np.vstack(all_states)
    y_ds = np.vstack(all_derivatives)[:, 0]
    y_di = np.vstack(all_derivatives)[:, 1]

    # GP parameters to favor sparsity and physical interaction terms
    gp_params = {
        'population_size': 5000,
        'generations': 40,
        'function_set': ('mul', 'add', 'sub'),
        'parsimony_coefficient': 0.001, 
        'feature_names': ('S', 'I'),
        'stopping_criteria': 0.0001,
        'random_state': 42
    }

    est_ds = SymbolicRegressor(**gp_params)
    est_di = SymbolicRegressor(**gp_params)

    print("\nDiscovering equations...")
    est_ds.fit(X, y_ds)
    est_di.fit(X, y_di)

    print("\n" + "="*50)
    print("SYMBOLIC DISCOVERY RESULTS")
    print("="*50)
    print(f"(S)' = {est_ds._program}")
    print(f"(I)' = {est_di._program}")
    print("="*50)

    # Simulation to verify that the discovered math is stable
    def discovered_system(y, t):
        # State clamping within the ODE solver for added robustness
        S, I = np.clip(y[0], 0, 1), np.clip(y[1], 0, 1)
        state = np.array([[S, I]])
        return [est_ds.predict(state)[0], est_di.predict(state)[0]]

    t_plot = np.linspace(0, 40, 200)
    # Integrate starting from a standard outbreak state
    sol = odeint(discovered_system, [0.99, 0.01], t_plot)
    
    plt.figure(figsize=(10, 6))
    plt.plot(t_plot, sol[:, 0], 'b-', label='Discovered S (Trajectory)')
    plt.plot(t_plot, sol[:, 1], 'r-', label='Discovered I (Trajectory)')
    plt.ylim(-0.05, 1.05)
    plt.axhline(0, color='black', lw=1, alpha=0.3)
    plt.title("Epidemic Trajectory from Discovered Equations")
    plt.xlabel("Time")
    plt.ylabel("Population Fraction")
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/discovery_final_constrained.png")
    print("Final discovery plot saved to plots/discovery_final_constrained.png")
    plt.show()

if __name__ == "__main__":
    discover_with_gp()