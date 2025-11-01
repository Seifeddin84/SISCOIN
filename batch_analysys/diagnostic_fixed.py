# diagnostic_fixed.py
"""
FIXED DIAGNOSTIC - Tests with proper learning rates
The issue: Data scale is very small (~0.3cm), so learning rate must be tiny!
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from balance_pinn import load_balance_data, BalancePINN
from balance_pinn import compute_data_loss
from sklearn.metrics import r2_score, mean_squared_error

# Test file
DATA_PATH = r"G:\human-balance-evaluation-database-1.0.0\BDS00001"

print("="*80)
print("FIXED DIAGNOSTIC - Testing Different Learning Rates")
print("="*80)

# Load data
print("\n[1] Loading data...")
time_data, cop_data, subject_info = load_balance_data(DATA_PATH)

print(f"✓ Loaded {len(time_data)} samples")
print(f"  COP X std: {cop_data[:,0].std()*100:.3f} cm")
print(f"  COP Y std: {cop_data[:,1].std()*100:.3f} cm")
print(f"  → Data scale is VERY SMALL (sub-centimeter)")

# Subsample
indices = np.linspace(0, len(time_data)-1, 5000, dtype=int)
t_train = torch.tensor(time_data[indices].reshape(-1, 1), dtype=torch.float32)
cop_train = torch.tensor(cop_data[indices], dtype=torch.float32)

# Test multiple learning rates
learning_rates = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5]

print(f"\n[2] Testing {len(learning_rates)} different learning rates...")
print("="*80)

results = []

for lr in learning_rates:
    print(f"\nLearning Rate: {lr:.2e}")
    print("-"*40)

    # Create fresh model
    model = BalancePINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train for 2000 epochs
    epoch_losses = []

    for epoch in range(2000):
        optimizer.zero_grad()
        loss = compute_data_loss(model, t_train, cop_train)
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())

        if epoch % 500 == 0:
            print(f"  Epoch {epoch:4d}: Loss = {loss.item():.8f}")

    # Evaluate
    with torch.no_grad():
        cop_pred = model(t_train).numpy()
    cop_true = cop_train.numpy()

    r2_x = r2_score(cop_true[:, 0], cop_pred[:, 0])
    r2_y = r2_score(cop_true[:, 1], cop_pred[:, 1])
    rmse_x = np.sqrt(mean_squared_error(cop_true[:, 0], cop_pred[:, 0]))
    rmse_y = np.sqrt(mean_squared_error(cop_true[:, 1], cop_pred[:, 1]))

    # Check if diverged
    final_loss = epoch_losses[-1]
    initial_loss = epoch_losses[0]
    min_loss = min(epoch_losses)

    diverged = final_loss > initial_loss * 10

    status = "✗ DIVERGED" if diverged else ("✓ CONVERGED" if r2_x > 0.7 else "⚠ POOR FIT")

    print(f"\n  {status}")
    print(f"  R² X: {r2_x:.4f}, R² Y: {r2_y:.4f}")
    print(f"  RMSE: {rmse_x*100:.3f} cm, {rmse_y*100:.3f} cm")
    print(f"  Loss: initial={initial_loss:.8f}, min={min_loss:.8f}, final={final_loss:.8f}")

    results.append({
        'lr': lr,
        'r2_x': r2_x,
        'r2_y': r2_y,
        'rmse_x': rmse_x,
        'rmse_y': rmse_y,
        'diverged': diverged,
        'losses': epoch_losses,
        'cop_pred': cop_pred
    })

# Find best result
valid_results = [r for r in results if not r['diverged'] and r['r2_x'] > 0]
if valid_results:
    best = max(valid_results, key=lambda x: x['r2_x'])
    best_lr = best['lr']
    print(f"\n{'='*80}")
    print(f"BEST LEARNING RATE: {best_lr:.2e}")
    print(f"{'='*80}")
    print(f"  R² X: {best['r2_x']:.4f}")
    print(f"  R² Y: {best['r2_y']:.4f}")
    print(f"  RMSE: {best['rmse_x']*100:.3f} cm, {best['rmse_y']*100:.3f} cm")
else:
    print(f"\n{'='*80}")
    print("❌ NO LEARNING RATE WORKED!")
    print("All tested learning rates caused divergence or poor fit.")
    print("{'='*80}")
    best_lr = None

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Plot 1: Loss curves
ax = axes[0, 0]
for res in results:
    label = f"lr={res['lr']:.0e} ({'div' if res['diverged'] else 'ok'})"
    ax.semilogy(res['losses'], label=label, linewidth=1.5)
ax.set_title('Training Loss Curves')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 2: R² scores
ax = axes[0, 1]
lrs_str = [f"{r['lr']:.0e}" for r in results]
r2_scores = [r['r2_x'] for r in results]
colors = ['red' if r['diverged'] else ('green' if r['r2_x'] > 0.7 else 'orange') for r in results]
ax.bar(range(len(results)), r2_scores, color=colors)
ax.set_xticks(range(len(results)))
ax.set_xticklabels(lrs_str, rotation=45)
ax.set_ylabel('R² Score (X)')
ax.set_title('R² vs Learning Rate')
ax.axhline(y=0.8, color='g', linestyle='--', linewidth=1, label='Target')
ax.axhline(y=0.0, color='r', linestyle='--', linewidth=1)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Plot 3: RMSE
ax = axes[0, 2]
rmse_values = [r['rmse_x']*100 for r in results]
ax.bar(range(len(results)), rmse_values, color=colors)
ax.set_xticks(range(len(results)))
ax.set_xticklabels(lrs_str, rotation=45)
ax.set_ylabel('RMSE (cm)')
ax.set_title('RMSE vs Learning Rate')
ax.grid(True, alpha=0.3, axis='y')

# Plot 4-6: Best fit visualization
if best_lr:
    best_res = best
    cop_true = cop_train.numpy()
    cop_pred = best_res['cop_pred']

    # X trajectory
    ax = axes[1, 0]
    ax.plot(cop_true[:500, 0]*100, 'b-', label='True', linewidth=1, alpha=0.7)
    ax.plot(cop_pred[:500, 0]*100, 'r--', label=f'Pred (R²={best_res["r2_x"]:.3f})', linewidth=1)
    ax.set_title(f'Best Fit: lr={best_lr:.0e}')
    ax.set_xlabel('Sample')
    ax.set_ylabel('COP X (cm)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Y trajectory
    ax = axes[1, 1]
    ax.plot(cop_true[:500, 1]*100, 'b-', label='True', linewidth=1, alpha=0.7)
    ax.plot(cop_pred[:500, 1]*100, 'r--', label=f'Pred (R²={best_res["r2_y"]:.3f})', linewidth=1)
    ax.set_title(f'COP Y: lr={best_lr:.0e}')
    ax.set_xlabel('Sample')
    ax.set_ylabel('COP Y (cm)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2D scatter
    ax = axes[1, 2]
    ax.scatter(cop_true[::10, 0]*100, cop_true[::10, 1]*100, c='blue', s=1, alpha=0.5, label='True')
    ax.scatter(cop_pred[::10, 0]*100, cop_pred[::10, 1]*100, c='red', s=1, alpha=0.5, label='Pred')
    ax.set_xlabel('COP X (cm)')
    ax.set_ylabel('COP Y (cm)')
    ax.set_title('2D COP Path')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
else:
    for ax in [axes[1, 0], axes[1, 1], axes[1, 2]]:
        ax.text(0.5, 0.5, 'No valid fit', ha='center', va='center', transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
plt.savefig('learning_rate_diagnostic.png', dpi=150)
print(f"\n✓ Saved visualization to: learning_rate_diagnostic.png")

# Final recommendation
print(f"\n{'='*80}")
print("RECOMMENDATION")
print("="*80)

if best_lr:
    if best['r2_x'] > 0.8:
        print(f"✅ SUCCESS! Use learning rate: {best_lr:.0e}")
        print(f"\nUpdate your training config:")
        print(f"  lr_net = {best_lr:.0e}")
        print(f"  lr_params = {best_lr*2:.0e}  # Can be 2x network LR")
        print(f"\nThis should give R² > 0.8 even without physics constraints.")
    else:
        print(f"⚠️  Best LR ({best_lr:.0e}) gives R² = {best['r2_x']:.3f}")
        print(f"   This is better but still not great.")
        print(f"\n Possible actions:")
        print(f"   1. Try even lower learning rates (1e-5, 5e-6)")
        print(f"   2. Increase epochs to 5000-10000")
        print(f"   3. Increase network size: [256, 256, 128, 64]")
else:
    print("❌ CRITICAL: No learning rate worked!")
    print("\nThe problem is deeper than learning rate. Possible causes:")
    print("  1. Network initialization is bad")
    print("  2. Data preprocessing has issues")
    print("  3. Network architecture is wrong for this data")
    print("\nTry:")
    print("  - Increase network size drastically: [512, 512, 256, 128]")
    print("  - Try different activation (Sigmoid, ReLU instead of Tanh)")
    print("  - Normalize data to [-1, 1] range before training")

print("="*80)
