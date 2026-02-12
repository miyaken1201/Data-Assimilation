import numpy as np
from LETKF_fixed import EnKF_LETKF
from utils import rk4_step
import matplotlib.pyplot as plt
# --- Verification Experiment ---
# Parameters
N = 40
F = 8.0
dt = 0.05
spinup_steps = 1460
steps = 1460
obs_interval = 1

# True state generation
np.random.seed(1)
true_state = np.zeros(N)
true_state[0] = F # perturbation
for _ in range(spinup_steps): # spinup
    true_state = rk4_step(true_state, F, dt)

# Create KF
kf = EnKF_LETKF(N=N, F=F, dt=dt, ensemble_size=10, obs_error_std=1.0, inflation_factor=1.04, loc_radius=4)
kf.initialize_ensemble(true_state)

dfs_trace_history = []
dfs_svd_history = []
true_states = []

curr_state = true_state.copy()

for t in range(steps):
    # Forecast
    kf.forecast()
    curr_state = rk4_step(curr_state, F, dt)
    
    # Calculate DFS (Before Analysis)
    dfs_1 = kf.calculate_dfs_trace()
    dfs_2 = kf.calculate_dfs_svd()
    
    dfs_trace_history.append(dfs_1)
    dfs_svd_history.append(dfs_2)
    true_states.append(curr_state.copy())
    
    # Analysis
    obs = curr_state + np.random.normal(0, 1.0, N)
    kf.analysis(obs)

# Plotting
days = np.arange(steps) / 4.0  # 4ステップ=1日
day_ticks = np.arange(0, np.ceil(steps / 4.0) + 1, 50)

plt.figure(figsize=(10, 5))
plt.plot(days, dfs_trace_history, 'b-', linewidth=4, alpha=0.5, label='Trace Method')
plt.plot(days, dfs_svd_history, 'r--', linewidth=2, label='SVD Method')
plt.title('Comparison of DFS Calculation Methods in Lorenz-96 LETKF')
plt.xlabel('Days')
plt.ylabel('DFS')
plt.xticks(day_ticks, [f"{int(d)}" for d in day_ticks])
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('dfs_comparison.png')
# RMSEとSpreadも計算して表示
true_states = np.array(true_states)
rmse = kf.calculate_rmse(true_states)
spread = kf.calculate_spread()

# Plot RMSE & Spread
plt.figure(figsize=(10, 5))
plt.plot(days, rmse, 'g-', linewidth=2, label='RMSE')
plt.plot(days, spread, 'm--', linewidth=2, label='Spread')
plt.title('RMSE and Spread over 1 year Simulation')
plt.xlabel('Days')
plt.ylabel('Value')
plt.xticks(day_ticks, [f"{int(d)}" for d in day_ticks])
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('rmse_spread.png')


print(f"Mean DFS (Trace): {np.mean(dfs_trace_history):.4f}")
print(f"Mean DFS (SVD):   {np.mean(dfs_svd_history):.4f}")
print(f"Max Diff: {np.max(np.abs(np.array(dfs_trace_history) - np.array(dfs_svd_history))):.4e}")