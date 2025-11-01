# deep_diagnostic.py
"""
DEEP DIAGNOSTIC - Find the root cause of poor PINN performance
Tests multiple hypotheses about why the model isn't fitting
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from balance_pinn import load_balance_data, BalancePINN, SubjectInfo
from balance_pinn import compute_data_loss, compute_physics_loss

# Test file
DATA_PATH = r"G:\human-balance-evaluation-database-1.0.0\BDS00001"

print("="*80)
print("DEEP DIAGNOSTIC - PINN Performance Investigation")
print("="*80)

# ===========================================
# TEST 1: Data Loading
# ===========================================
print("\n[TEST 1] Data Loading & Preprocessing")
print("-"*80)

try:
    time_data, cop_data, subject_info = load_balance_data(DATA_PATH)

    print(f"✓ Data loaded successfully")
    print(f"  Samples: {len(time_data)}")
    print(f"  Duration: {time_data[-1]:.1f}s")
    print(f"  Sample rate: {subject_info.fs} Hz")
    print(f"  Subject: {subject_info.weight}kg, {subject_info.height}m")
    print(f"\n  COP Statistics (after mean removal):")
    print(f"  X: mean={cop_data[:,0].mean()*100:.4f}cm, std={cop_data[:,0].std()*100:.3f}cm")
    print(f"     range=[{cop_data[:,0].min()*100:.2f}, {cop_data[:,0].max()*100:.2f}]cm")
    print(f"  Y: mean={cop_data[:,1].mean()*100:.4f}cm, std={cop_data[:,1].std()*100:.3f}cm")
    print(f"     range=[{cop_data[:,1].min()*100:.2f}, {cop_data[:,1].max()*100:.2f}]cm")

    if cop_data[:,0].std() < 0.001 or cop_data[:,1].std() < 0.001:
        print("\n  ❌ ERROR: Data has very low variability!")
        print("     The COP barely moves - this is likely bad data.")
        sys.exit(1)
    else:
        print(f"\n  ✓ Data variability looks reasonable")

except Exception as e:
    print(f"❌ Failed to load data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ===========================================
# TEST 2: Simple Data-Only Fitting (No Physics)
# ===========================================
print("\n\n[TEST 2] Pure Data Fitting (No Physics Loss)")
print("-"*80)
print("Testing if network can fit data WITHOUT physics constraints...")

# Subsample for faster testing
indices = np.linspace(0, len(time_data)-1, 5000, dtype=int)
t_train = torch.tensor(time_data[indices].reshape(-1, 1), dtype=torch.float32)
cop_train = torch.tensor(cop_data[indices], dtype=torch.float32)

model_test = BalancePINN(expected_stiffness=300.0, expected_damping=3.0)
optimizer = torch.optim.Adam(model_test.parameters(), lr=1e-2)

print("Training for 2000 epochs with ONLY data loss...")
losses = []

for epoch in range(2000):
    optimizer.zero_grad()

    # ONLY data loss - no physics!
    loss = compute_data_loss(model_test, t_train, cop_train)

    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if epoch % 500 == 0:
        print(f"  Epoch {epoch:4d}: Loss = {loss.item():.6f}")

# Evaluate
with torch.no_grad():
    cop_pred = model_test(t_train).numpy()
cop_true = cop_train.numpy()

from sklearn.metrics import r2_score
r2_x_data_only = r2_score(cop_true[:, 0], cop_pred[:, 0])
r2_y_data_only = r2_score(cop_true[:, 1], cop_pred[:, 1])

print(f"\nResults with DATA ONLY (no physics):")
print(f"  R² X: {r2_x_data_only:.4f}")
print(f"  R² Y: {r2_y_data_only:.4f}")

if r2_x_data_only < 0.7 or r2_y_data_only < 0.7:
    print("\n  ❌ CRITICAL: Network can't even fit data WITHOUT physics!")
    print("     Problem is likely:")
    print("     1. Network architecture too small")
    print("     2. Learning rate wrong")
    print("     3. Data has issues")
    print("     4. Initialization problem")
else:
    print("\n  ✓ Network CAN fit data when physics is disabled")
    print("     → Problem is with physics loss conflicting with data loss")

# ===========================================
# TEST 3: Check Physics Loss Magnitude
# ===========================================
print("\n\n[TEST 3] Physics Loss Magnitude Check")
print("-"*80)

# Create a model that fits data well
model_good = BalancePINN()
optimizer = torch.optim.Adam(model_good.parameters(), lr=1e-2)

# Train with ONLY data for 1000 epochs
for epoch in range(1000):
    optimizer.zero_grad()
    loss = compute_data_loss(model_good, t_train, cop_train)
    loss.backward()
    optimizer.step()

# Now check what physics loss looks like
with torch.no_grad():
    data_loss_val = compute_data_loss(model_good, t_train, cop_train).item()
    physics_loss_val = compute_physics_loss(model_good, t_train.clone(), subject_info).item()

print(f"After training to fit data:")
print(f"  Data loss:    {data_loss_val:.6f}")
print(f"  Physics loss: {physics_loss_val:.6f}")
print(f"  Ratio (Physics/Data): {physics_loss_val/data_loss_val:.1f}x")

if physics_loss_val / data_loss_val > 1000:
    print("\n  ❌ CRITICAL: Physics loss is MUCH larger than data loss!")
    print("     When these are combined, physics dominates and prevents data fitting.")
    print("     → Need to scale physics loss down or use MUCH lower weight")
elif physics_loss_val / data_loss_val > 10:
    print("\n  ⚠️  Physics loss is significantly larger than data loss")
    print("     → Need careful weighting (physics weight should be < 0.01)")
else:
    print("\n  ✓ Physics loss magnitude seems reasonable")

# ===========================================
# TEST 4: Check if Original Data Had Mean
# ===========================================
print("\n\n[TEST 4] Check Data Centering")
print("-"*80)

print(f"Mean values that were removed:")
print(f"  COP X mean: {subject_info.cop_x_mean*100:.3f} cm")
print(f"  COP Y mean: {subject_info.cop_y_mean*100:.3f} cm")

if abs(subject_info.cop_x_mean) > 0.05 or abs(subject_info.cop_y_mean) > 0.05:
    print(f"\n  ℹ️  Data had significant offset (>5cm)")
    print(f"     This is normal - data is now centered for physics model")
else:
    print(f"\n  ✓ Data was already near-centered")

# ===========================================
# TEST 5: Try Simplified Training
# ===========================================
print("\n\n[TEST 5] Simplified Training Strategy")
print("-"*80)
print("Testing with ultra-low physics weight...")

model_final = BalancePINN()

# Separate optimizers
opt_net = torch.optim.Adam(model_final.net.parameters(), lr=5e-3)
opt_params = torch.optim.Adam([model_final.stiffness_x, model_final.stiffness_y,
                                model_final.damping_x, model_final.damping_y], lr=1e-2)

# Ultra-simple training: mostly data fitting
print("Training 3000 epochs: 99% data, 1% physics")

for epoch in range(3000):
    opt_net.zero_grad()
    opt_params.zero_grad()

    loss_data = compute_data_loss(model_final, t_train, cop_train)
    loss_physics = compute_physics_loss(model_final, t_train.clone(), subject_info)

    # CRITICAL: Very low physics weight
    loss_total = loss_data + 0.0001 * loss_physics

    loss_total.backward()

    torch.nn.utils.clip_grad_norm_(model_final.parameters(), 1.0)

    opt_net.step()
    opt_params.step()

    if epoch % 1000 == 0:
        print(f"  Epoch {epoch:4d}: Data={loss_data:.6f}, Physics={loss_physics:.6f}")

# Final evaluation
with torch.no_grad():
    cop_pred_final = model_final(t_train).numpy()

r2_x_final = r2_score(cop_true[:, 0], cop_pred_final[:, 0])
r2_y_final = r2_score(cop_true[:, 1], cop_pred_final[:, 1])

print(f"\nFinal Results (w_physics=0.0001):")
print(f"  R² X: {r2_x_final:.4f}")
print(f"  R² Y: {r2_y_final:.4f}")
print(f"  Stiffness: [{model_final.stiffness_x.item():.1f}, {model_final.stiffness_y.item():.1f}] Nm/rad")
print(f"  Damping: [{model_final.damping_x.item():.2f}, {model_final.damping_y.item():.2f}] Nm·s/rad")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: Loss curves from data-only training
axes[0, 0].semilogy(losses)
axes[0, 0].set_title('Data-Only Training Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Data-only fit
axes[0, 1].plot(cop_true[:500, 0]*100, 'b-', label='True', linewidth=1)
axes[0, 1].plot(cop_pred[:500, 0]*100, 'r--', label=f'Pred (R²={r2_x_data_only:.3f})', linewidth=1)
axes[0, 1].set_title('Data-Only Fit (no physics)')
axes[0, 1].set_xlabel('Sample')
axes[0, 1].set_ylabel('COP X (cm)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Final fit
axes[1, 0].plot(cop_true[:500, 0]*100, 'b-', label='True', linewidth=1)
axes[1, 0].plot(cop_pred_final[:500, 0]*100, 'r--', label=f'Pred (R²={r2_x_final:.3f})', linewidth=1)
axes[1, 0].set_title('Final Fit (w_physics=0.0001)')
axes[1, 0].set_xlabel('Sample')
axes[1, 0].set_ylabel('COP X (cm)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Comparison
axes[1, 1].bar(['Data-Only', 'Final'], [r2_x_data_only, r2_x_final])
axes[1, 1].set_ylabel('R² Score')
axes[1, 1].set_title('Performance Comparison')
axes[1, 1].set_ylim([0, 1])
axes[1, 1].axhline(y=0.8, color='g', linestyle='--', label='Target')
axes[1, 1].axhline(y=0.6, color='orange', linestyle='--', label='Acceptable')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('diagnostic_results.png', dpi=150)
print("\n✓ Saved visualization to: diagnostic_results.png")

# ===========================================
# DIAGNOSIS SUMMARY
# ===========================================
print("\n\n" + "="*80)
print("DIAGNOSIS SUMMARY")
print("="*80)

print("\nTest Results:")
print(f"  [1] Data Loading:        {'✓ PASS' if cop_data[:,0].std() > 0.001 else '✗ FAIL'}")
print(f"  [2] Data-Only Fit:       R² = {r2_x_data_only:.3f} ({'✓ PASS' if r2_x_data_only > 0.7 else '✗ FAIL'})")
print(f"  [3] Physics/Data Ratio:  {physics_loss_val/data_loss_val:.1f}x ({'✓ OK' if physics_loss_val/data_loss_val < 100 else '✗ TOO HIGH'})")
print(f"  [5] Ultra-Low Physics:   R² = {r2_x_final:.3f} ({'✓ PASS' if r2_x_final > 0.7 else '✗ FAIL'})")

print("\n" + "-"*80)
print("RECOMMENDATION:")
print("-"*80)

if r2_x_data_only > 0.8 and r2_x_final > 0.8:
    print("✅ Network architecture is fine!")
    print("✅ Can fit data well with ultra-low physics weight")
    print("\n📋 ACTION: Use physics weight ≤ 0.0001 in all training")
    print("   The physics loss is too strong relative to data loss.")

elif r2_x_data_only > 0.8 and r2_x_final < 0.7:
    print("⚠️  Network CAN fit data, but physics is interfering")
    print("\n📋 ACTION: Physics model may be incompatible with data")
    print("   Options:")
    print("   1. Use w_physics = 0 (pure data fitting)")
    print("   2. Re-examine physics equations")
    print("   3. Check if inverted pendulum model is appropriate")

elif r2_x_data_only < 0.7:
    print("❌ Network CANNOT fit data even without physics!")
    print("\n📋 ACTION: Fundamental problem with network or data")
    print("   Options:")
    print("   1. Increase network size (try [256, 256, 128])")
    print("   2. Increase epochs (try 10000+)")
    print("   3. Check data preprocessing")
    print("   4. Try different learning rates")
else:
    print("❓ Mixed results - needs further investigation")

print("\n" + "="*80)
print("Next: Open diagnostic_results.png to visualize the issue")
print("="*80)
