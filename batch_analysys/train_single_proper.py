# train_single_subject_proper.py
"""
Proper single-subject training with full diagnostics
- Shows loss evolution in real-time
- Tracks stiffness and damping
- ONE plot at the end only
- No annoying figure messages
- FIXED: Very low physics weight to actually fit oscillations
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from balance_pinn import (
    load_balance_data, BalancePINN,
    compute_data_loss, compute_physics_loss,
    compute_initial_condition_loss, compute_smoothness_loss,
    compute_parameter_regularization
)
from sklearn.metrics import r2_score, mean_squared_error

# Close any existing plots
plt.close('all')

# Configuration
DATA_PATH = r"G:\human-balance-evaluation-database-1.0.0\BDS00001"
EPOCHS = 10000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# CRITICAL: Use very low learning rate and physics weight
LR_NET = 1e-4  # Low LR for small data scale
LR_PARAMS = 5e-4
W_PHYSICS = 0.0  # START WITH ZERO! Fit data first!

print("="*80)
print("PROPER SINGLE-SUBJECT TRAINING")
print("="*80)
print(f"Subject: {DATA_PATH}")
print(f"Epochs: {EPOCHS}")
print(f"Device: {DEVICE}")
print(f"Learning Rate (Net): {LR_NET}")
print(f"Learning Rate (Params): {LR_PARAMS}")
print(f"Physics Weight: {W_PHYSICS} (DISABLED - pure data fitting)")
print("="*80)

# Load data
print("\n[1/4] Loading data...")
time_data, cop_data, subject_info = load_balance_data(DATA_PATH)

print(f"  ✓ Loaded {len(time_data)} samples")
print(f"  ✓ Subject: {subject_info}")
print(f"  ✓ COP std: X={cop_data[:,0].std()*100:.3f}cm, Y={cop_data[:,1].std()*100:.3f}cm")

# Prepare training data
n_train = min(len(time_data), 10000)
indices = np.linspace(0, len(time_data)-1, n_train, dtype=int)

t_train = torch.tensor(time_data[indices].reshape(-1, 1), dtype=torch.float32).to(DEVICE)
cop_train = torch.tensor(cop_data[indices], dtype=torch.float32).to(DEVICE)

t0 = t_train[:1]
cop0 = cop_train[:1]

# Create model
print("\n[2/4] Creating model...")
model = BalancePINN(expected_stiffness=300.0, expected_damping=3.0).to(DEVICE)

# Optimizers
opt_net = torch.optim.Adam(model.net.parameters(), lr=LR_NET)
opt_params = torch.optim.Adam(
    [model.stiffness_x, model.stiffness_y, model.damping_x, model.damping_y],
    lr=LR_PARAMS
)

# Training history
history = {
    'epoch': [],
    'loss_total': [],
    'loss_data': [],
    'loss_physics': [],
    'loss_init': [],
    'stiffness_x': [],
    'stiffness_y': [],
    'damping_x': [],
    'damping_y': [],
}

# Training loop
print(f"\n[3/4] Training for {EPOCHS} epochs...")
print("-"*80)
print(f"{'Epoch':>6} | {'Total':>10} | {'Data':>10} | {'Physics':>10} | {'Kx':>8} | {'Ky':>8} | {'Cx':>8} | {'Cy':>8}")
print("-"*80)

for epoch in range(EPOCHS):
    model.train()

    # Zero gradients
    opt_net.zero_grad()
    opt_params.zero_grad()

    # Compute losses
    loss_data = compute_data_loss(model, t_train, cop_train)
    loss_physics = compute_physics_loss(model, t_train.clone(), subject_info)
    loss_init = compute_initial_condition_loss(model, t0, cop0)
    loss_smooth = compute_smoothness_loss(model, t_train.clone())
    loss_reg = compute_parameter_regularization(model)

    # Total loss - CRITICAL: Very low or zero physics weight
    loss_total = (
        1.0 * loss_data +
        W_PHYSICS * loss_physics +  # ZERO or very small!
        1.0 * loss_init +
        0.001 * loss_smooth +
        0.001 * loss_reg
    )

    # Backward
    loss_total.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Update
    opt_net.step()
    opt_params.step()

    # Record every 100 epochs
    if epoch % 100 == 0:
        history['epoch'].append(epoch)
        history['loss_total'].append(loss_total.item())
        history['loss_data'].append(loss_data.item())
        history['loss_physics'].append(loss_physics.item())
        history['loss_init'].append(loss_init.item())
        history['stiffness_x'].append(model.stiffness_x.item())
        history['stiffness_y'].append(model.stiffness_y.item())
        history['damping_x'].append(model.damping_x.item())
        history['damping_y'].append(model.damping_y.item())

    # Print every 500 epochs
    if epoch % 500 == 0:
        print(f"{epoch:6d} | {loss_total.item():10.6f} | {loss_data.item():10.6f} | "
              f"{loss_physics.item():10.6f} | {model.stiffness_x.item():8.2f} | "
              f"{model.stiffness_y.item():8.2f} | {model.damping_x.item():8.3f} | "
              f"{model.damping_y.item():8.3f}")

print("-"*80)
print("✓ Training complete!")

# Evaluation
print("\n[4/4] Evaluating...")
model.eval()

with torch.no_grad():
    # Full dataset evaluation
    t_full = torch.tensor(time_data.reshape(-1, 1), dtype=torch.float32).to(DEVICE)
    cop_pred_centered = model(t_full).cpu().numpy()

cop_true_centered = cop_data

# Restore means for visualization
cop_pred = cop_pred_centered.copy()
cop_pred[:, 0] += subject_info.cop_x_mean
cop_pred[:, 1] += subject_info.cop_y_mean

cop_true = cop_true_centered.copy()
cop_true[:, 0] += subject_info.cop_x_mean
cop_true[:, 1] += subject_info.cop_y_mean

# Metrics (on centered data)
rmse_x = np.sqrt(mean_squared_error(cop_true_centered[:, 0], cop_pred_centered[:, 0]))
rmse_y = np.sqrt(mean_squared_error(cop_true_centered[:, 1], cop_pred_centered[:, 1]))
r2_x = r2_score(cop_true_centered[:, 0], cop_pred_centered[:, 0])
r2_y = r2_score(cop_true_centered[:, 1], cop_pred_centered[:, 1])

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"R² X: {r2_x:.4f} | RMSE X: {rmse_x*100:.3f} cm")
print(f"R² Y: {r2_y:.4f} | RMSE Y: {rmse_y*100:.3f} cm")
print(f"\nIdentified Parameters:")
print(f"  Stiffness X: {model.stiffness_x.item():.2f} Nm/rad")
print(f"  Stiffness Y: {model.stiffness_y.item():.2f} Nm/rad")
print(f"  Damping X: {model.damping_x.item():.3f} Nm·s/rad")
print(f"  Damping Y: {model.damping_y.item():.3f} Nm·s/rad")
print("="*80)

# Create comprehensive visualization
print("\nCreating final visualization...")
plt.close('all')  # Close any existing figures

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Row 1: Trajectories
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(time_data, cop_true[:, 0]*100, 'b-', alpha=0.7, label='Measured', linewidth=1)
ax1.plot(time_data, cop_pred[:, 0]*100, 'r--', alpha=0.7, label='PINN', linewidth=1)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('COP X (cm)')
ax1.set_title(f'COP X Trajectory (R²={r2_x:.4f})')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(time_data, cop_true[:, 1]*100, 'b-', alpha=0.7, label='Measured', linewidth=1)
ax2.plot(time_data, cop_pred[:, 1]*100, 'r--', alpha=0.7, label='PINN', linewidth=1)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('COP Y (cm)')
ax2.set_title(f'COP Y Trajectory (R²={r2_y:.4f})')
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(cop_true[:, 0]*100, cop_true[:, 1]*100, 'b-', alpha=0.5, label='Measured', linewidth=0.5)
ax3.plot(cop_pred[:, 0]*100, cop_pred[:, 1]*100, 'r-', alpha=0.5, label='PINN', linewidth=0.5)
ax3.set_xlabel('COP X (cm)')
ax3.set_ylabel('COP Y (cm)')
ax3.set_title('2D COP Path')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_aspect('equal', adjustable='box')

# Row 2: Loss evolution
ax4 = fig.add_subplot(gs[1, 0])
ax4.semilogy(history['epoch'], history['loss_total'], 'k-', linewidth=2, label='Total')
ax4.semilogy(history['epoch'], history['loss_data'], 'b-', linewidth=1.5, label='Data')
ax4.semilogy(history['epoch'], history['loss_physics'], 'r-', linewidth=1, label='Physics')
ax4.semilogy(history['epoch'], history['loss_init'], 'g-', linewidth=1, label='Init Cond')
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Loss')
ax4.set_title('Training Loss Evolution')
ax4.legend()
ax4.grid(True, alpha=0.3)

ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(history['epoch'], history['stiffness_x'], 'b-', linewidth=2,
         label=f'Kx (final: {model.stiffness_x.item():.1f})')
ax5.plot(history['epoch'], history['stiffness_y'], 'r-', linewidth=2,
         label=f'Ky (final: {model.stiffness_y.item():.1f})')
ax5.set_xlabel('Epoch')
ax5.set_ylabel('Stiffness (Nm/rad)')
ax5.set_title('Stiffness Parameter Evolution')
ax5.legend()
ax5.grid(True, alpha=0.3)

ax6 = fig.add_subplot(gs[1, 2])
ax6.plot(history['epoch'], history['damping_x'], 'b-', linewidth=2,
         label=f'Cx (final: {model.damping_x.item():.2f})')
ax6.plot(history['epoch'], history['damping_y'], 'r-', linewidth=2,
         label=f'Cy (final: {model.damping_y.item():.2f})')
ax6.set_xlabel('Epoch')
ax6.set_ylabel('Damping (Nm·s/rad)')
ax6.set_title('Damping Parameter Evolution')
ax6.legend()
ax6.grid(True, alpha=0.3)

# Row 3: Detailed comparison (zoom in)
zoom_start = 0
zoom_end = min(2000, len(time_data))

ax7 = fig.add_subplot(gs[2, 0])
ax7.plot(time_data[zoom_start:zoom_end], cop_true[zoom_start:zoom_end, 0]*100,
         'b-', alpha=0.7, label='Measured', linewidth=1.5)
ax7.plot(time_data[zoom_start:zoom_end], cop_pred[zoom_start:zoom_end, 0]*100,
         'r--', alpha=0.9, label='PINN', linewidth=1.5)
ax7.set_xlabel('Time (s)')
ax7.set_ylabel('COP X (cm)')
ax7.set_title('Detailed View: First 20s (COP X)')
ax7.legend()
ax7.grid(True, alpha=0.3)

ax8 = fig.add_subplot(gs[2, 1])
ax8.plot(time_data[zoom_start:zoom_end], cop_true[zoom_start:zoom_end, 1]*100,
         'b-', alpha=0.7, label='Measured', linewidth=1.5)
ax8.plot(time_data[zoom_start:zoom_end], cop_pred[zoom_start:zoom_end, 1]*100,
         'r--', alpha=0.9, label='PINN', linewidth=1.5)
ax8.set_xlabel('Time (s)')
ax8.set_ylabel('COP Y (cm)')
ax8.set_title('Detailed View: First 20s (COP Y)')
ax8.legend()
ax8.grid(True, alpha=0.3)

# Error distribution
ax9 = fig.add_subplot(gs[2, 2])
error_x = (cop_pred_centered[:, 0] - cop_true_centered[:, 0]) * 100  # cm
error_y = (cop_pred_centered[:, 1] - cop_true_centered[:, 1]) * 100  # cm
ax9.hist(error_x, bins=50, alpha=0.5, label=f'X (std={error_x.std():.3f}cm)', color='blue')
ax9.hist(error_y, bins=50, alpha=0.5, label=f'Y (std={error_y.std():.3f}cm)', color='red')
ax9.set_xlabel('Prediction Error (cm)')
ax9.set_ylabel('Frequency')
ax9.set_title('Error Distribution')
ax9.legend()
ax9.grid(True, alpha=0.3, axis='y')

plt.suptitle(f'BDS00001 - Complete Training Results\nR²=[{r2_x:.4f}, {r2_y:.4f}] | '
             f'RMSE=[{rmse_x*100:.3f}, {rmse_y*100:.3f}]cm',
             fontsize=14, fontweight='bold')

# Save and close
output_file = 'BDS00001_complete_results.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
plt.close('all')  # IMPORTANT: Close to avoid the annoying message!

print(f"✓ Saved complete visualization to: {output_file}")

# Save history to CSV
df_history = pd.DataFrame(history)
df_history.to_csv('BDS00001_training_history.csv', index=False)
print(f"✓ Saved training history to: BDS00001_training_history.csv")

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)

if r2_x > 0.8 and r2_y > 0.8:
    print("✅ EXCELLENT! The model fitted the data very well.")
    print("   The oscillations should be captured in the plots.")
elif r2_x > 0.6:
    print("⚠️  MODERATE fit. To improve:")
    print("   1. Increase epochs to 15000-20000")
    print("   2. Decrease learning rate further (try 5e-5)")
    print("   3. Keep physics weight at 0 or use 0.00001")
else:
    print("❌ POOR fit. Check:")
    print("   1. Is the learning rate too high? (try 1e-5)")
    print("   2. Are there NaN/Inf in the data?")
    print("   3. Is the network architecture appropriate?")

print("="*80)
