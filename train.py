import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from src.gillespie import simulate_epidemic, interpolate_trajectory
from src.model import SIRDerivativeNet, NeuralODE

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print(f"Using MPS for hardware acceleration")
else:
    DEVICE = torch.device("cpu")

torch.set_default_dtype(torch.float32)

N_POPULATION = 1000.0 
EPOCHS = 2000             
LEARNING_RATE = 1e-3 
BATCH_SIZE = 64           
TRUE_BETA = 0.5
TRUE_GAMMA = 0.1

def get_training_batch(batch_size=64):
    batch_y = []
    t_eval = np.linspace(0, 40, 40) 

    count = 0
    while count < batch_size:
        raw_data = simulate_epidemic(TRUE_BETA, TRUE_GAMMA, N=N_POPULATION)
        max_infected = np.max(raw_data[:, 2])
        # Filtering for successful outbreaks only
        if max_infected < 50:
            continue
            
        trajectory = interpolate_trajectory(raw_data, t_eval, N_POPULATION)
        batch_y.append(trajectory)
        count += 1

    y_tensor = torch.tensor(np.array(batch_y), dtype=torch.float32).to(DEVICE)
    y_tensor = y_tensor.permute(1, 0, 2) # [Time, Batch, States]
    
    t_tensor = torch.tensor(t_eval, dtype=torch.float32).to(DEVICE)
    y0_tensor = y_tensor[0, :, :] 

    return t_tensor, y_tensor, y0_tensor

def validate_and_plot(model, epoch):
    model.eval()
    with torch.no_grad():
        t, y_true, y0 = get_training_batch(batch_size=1)
        y_pred = model(y0, t)
        
        t_cpu = t.cpu().numpy()
        y_true_cpu = y_true.permute(1,0,2).cpu().numpy()[0]
        y_pred_cpu = y_pred.permute(1,0,2).cpu().numpy()[0]
        
        plt.figure(figsize=(10, 6))
        plt.plot(t_cpu, y_true_cpu[:, 1], 'r.', alpha=0.3, label='Gillespie Data (Infected)')
        plt.plot(t_cpu, y_pred_cpu[:, 1], 'r-', linewidth=2, label='Neural ODE (Infected)')
        plt.plot(t_cpu, y_pred_cpu[:, 0], 'b--', alpha=0.7, label='Susceptible (Learned)')
        plt.plot(t_cpu, y_pred_cpu[:, 2], 'g--', alpha=0.7, label='Removed (Learned)')
        
        plt.title(f"Epoch {epoch}: Physics-Constrained Learning")
        plt.xlabel("Time")
        plt.ylabel("Population Fraction")
        plt.ylim(-0.05, 1.05)
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.savefig(f"plots/epoch_{epoch:03d}.png")
        plt.close()
    model.train()

def train():
    os.makedirs("data", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    derivative_net = SIRDerivativeNet().to(DEVICE)
    model = NeuralODE(derivative_net, solver='rk4', step_size=0.5).to(DEVICE)
    
    # Weight decay added to promote simpler internal representations for SINDy/GP
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.5)

    data_loss_fn = nn.HuberLoss()
    
    print(f"Starting physics-informed online learning on {DEVICE}...")
    start_time = time.time()
    best_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        t, y_true, y0 = get_training_batch(batch_size=BATCH_SIZE)
        y_pred = model(y0, t)
        
        # 1. Standard Data Fit Loss
        loss = data_loss_fn(y_pred, y_true)
        
        # 2. Physics Constraint Loss: S + I + R = 1
        # This penalizes the model if it tries to 'create' or 'destroy' people
        total_pop = torch.sum(y_pred, dim=2)
        conservation_loss = torch.mean((total_pop - 1.0)**2)
        
        total_loss = loss + 0.1 * conservation_loss
        
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(total_loss.item())
        
        if epoch % 50 == 0:
            current_val = total_loss.item()
            print(f"Epoch {epoch:04d} | Total Loss: {current_val:.6f}")
            validate_and_plot(model, epoch)
            
            if current_val < best_loss:
                best_loss = current_val
                torch.save(derivative_net.state_dict(), "data/best_sir_model.pth")

    print(f"Training Complete in {time.time() - start_time:.2f}s")
    torch.save(derivative_net.state_dict(), "data/final_sir_model.pth")

if __name__ == "__main__":
    train()