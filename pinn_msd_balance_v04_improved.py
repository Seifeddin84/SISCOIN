# pinn_msd_balance_v04_improved.py
# PINN for human quiet balance (MSD-based physics loss) - v0.4 IMPROVED
#
# IMPROVEMENTS OVER v0.3:
# - FIXED: Deterministic file loading (sorted order, no randomization)
# - FIXED: Deterministic physics sampling (no random sampling during training)
# - IMPROVED: Better parameter initialization based on biomechanical priors
# - IMPROVED: Better physics loss formulation without over-normalization
# - IMPROVED: More aggressive training schedule for physics constraints
# - IMPROVED: Added deterministic seed control for all random operations
# - IMPROVED: Better mass/inertia estimation from anthropometric data
#
# Requirements: python3, torch, wfdb, numpy, pandas, matplotlib, scikit-learn, scipy

import os
import re
import wfdb
import math
import time as pytime
import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import HuberRegressor
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

# -------------------------
# CONFIG / HYPERPARAMS
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_EVERY = 200
PLOT_EVERY = 200
PHYS_SAMPLE = 1024  # Increased for better physics coverage
DATA_MAX_POINTS = 30000
DEFAULT_FS = 100

# IMPROVED: More aggressive physics schedule, start earlier with physics
PHASES = [
    (200, 1.0, 0.0, 0.0),      # warm-up: fit data only
    (200, 1.0, 0.02, 0.001),   # introduce physics earlier and stronger
    (300, 1.0, 0.1, 0.005),    # increase physics weight
    (400, 1.0, 0.3, 0.01),     # stronger physics
    (500, 1.0, 0.5, 0.02),     # balanced physics + data
    (400, 0.8, 0.8, 0.03)      # final: prioritize physics for parameter identification
]
LR = 1e-3
SEED = 42

# FIXED: Set all random seeds for reproducibility
def set_all_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # Make PyTorch operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_all_seeds(SEED)

# -------------------------
# UTIL: WFDB reading & COP extraction
# -------------------------
def read_wfdb_record(basepath):
    rec = wfdb.rdrecord(basepath)
    data = pd.DataFrame(rec.p_signal, columns=rec.sig_name)
    fs = getattr(rec, "fs", DEFAULT_FS)
    subject_info = {"weight": 75.0, "height": 1.70}
    hea_file = basepath + ".hea"
    if os.path.exists(hea_file):
        with open(hea_file, 'r') as f:
            content = f.read()
            h = re.search(r'#\s*Height[:=]?\s*([0-9]+\.*[0-9]*)', content)
            w = re.search(r'#\s*Weight[:=]?\s*([0-9]+\.*[0-9]*)', content)
            if h:
                try:
                    val = float(h.group(1))
                    subject_info['height'] = (val/100.0) if val > 3 else val
                except:
                    pass
            if w:
                try:
                    subject_info['weight'] = float(w.group(1))
                except:
                    pass
    return data, fs, subject_info

def ensure_cop_columns(df):
    # rename common alternatives to standard COPx, COPy
    colmap = {}
    for candidate in ['COPx', 'COP_x', 'copx', 'cop_x', 'X', 'COP X']:
        if candidate in df.columns:
            colmap[candidate] = 'COPx'; break
    for candidate in ['COPy', 'COP_y', 'copy', 'cop_y', 'Y', 'COP Y']:
        if candidate in df.columns:
            colmap[candidate] = 'COPy'; break
    df.rename(columns=colmap, inplace=True)
    if 'COPx' not in df.columns or 'COPy' not in df.columns:
        raise ValueError("Data missing COP channels. Available columns: " + ", ".join(df.columns))

# IMPROVED: Better inertia estimation from biomechanics literature
def estimate_body_inertia(weight_kg, height_m):
    """
    Estimate moment of inertia for human body about ankle.
    Based on biomechanics literature: I ≈ m * h^2 * k
    where k ≈ 0.3-0.4 for human body about ankle

    Returns: Inertia in kg*m^2
    """
    # Conservative estimate: treat body as inverted pendulum
    # Center of mass height typically ~55% of total height
    com_height = 0.55 * height_m
    # Moment of inertia about ankle
    inertia = weight_kg * (com_height ** 2)
    return max(inertia, 1.0)  # minimum 1.0 kg*m^2

# -------------------------
# MODEL
# -------------------------
class PINN_MSD(nn.Module):
    def __init__(self, hidden=128, nlayers=4, weight_kg=75.0, height_m=1.7):
        super().__init__()
        layers = []
        layers.append(nn.Linear(1, hidden)); layers.append(nn.Tanh())
        for _ in range(nlayers-1):
            layers.append(nn.Linear(hidden, hidden)); layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden, 2))  # outputs: normalized COPx, COPy
        self.net = nn.Sequential(*layers)

        # IMPROVED: Better initialization based on biomechanical priors
        # Typical damping for human balance: 20-100 Ns/m
        # Typical stiffness for human balance: 500-3000 N/m
        # Convert to effective values considering inertia
        I = estimate_body_inertia(weight_kg, height_m)

        # For normalized coordinates (std~1), estimate reasonable ranges
        # Natural frequency for balance: 0.3-1.5 Hz -> omega ~ 2-10 rad/s
        # omega = sqrt(k/I) -> k ~ I * omega^2
        # damping ratio: 0.1-0.5 -> d ~ 2*zeta*sqrt(k*I)

        typical_omega = 5.0  # rad/s - middle of physiological range
        typical_k_over_I = typical_omega ** 2  # ~25
        typical_zeta = 0.3  # damping ratio
        typical_d_over_I = 2 * typical_zeta * typical_omega  # ~3

        # Initialize in log-domain for positivity
        self.log_damping_x = nn.Parameter(torch.tensor(np.log(typical_d_over_I), dtype=torch.float32))
        self.log_stiffness_x = nn.Parameter(torch.tensor(np.log(typical_k_over_I), dtype=torch.float32))
        self.log_damping_y = nn.Parameter(torch.tensor(np.log(typical_d_over_I), dtype=torch.float32))
        self.log_stiffness_y = nn.Parameter(torch.tensor(np.log(typical_k_over_I), dtype=torch.float32))

        # Store inertia for physics calculations
        self.register_buffer('inertia', torch.tensor(I, dtype=torch.float32))

        # Small bias for DC offset in normalized space
        self.bias = nn.Parameter(torch.zeros(1, 2))

        # Better weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, t):
        out = self.net(t) + self.bias
        return out

    def phys_params(self):
        """Return physics parameters (d/I and k/I ratios)"""
        return {
            'd_x': torch.exp(self.log_damping_x),
            'k_x': torch.exp(self.log_stiffness_x),
            'd_y': torch.exp(self.log_damping_y),
            'k_y': torch.exp(self.log_stiffness_y)
        }

    def phys_params_absolute(self):
        """Return absolute physics parameters (d and k in SI units)"""
        params = self.phys_params()
        I = self.inertia.item()
        return {
            'd_x': params['d_x'].item() * I,
            'k_x': params['k_x'].item() * I,
            'd_y': params['d_y'].item() * I,
            'k_y': params['k_y'].item() * I,
            'I': I
        }

# -------------------------
# DERIVATIVES & PHYSICS LOSS
# -------------------------
def compute_derivatives(u, t):
    """Compute first and second derivatives of u with respect to t"""
    # u: (N,2), t requires_grad True
    du = []
    d2u = []
    for dim in range(u.shape[1]):
        ui = u[:, dim:dim+1]
        dui = torch.autograd.grad(ui.sum(), t, create_graph=True)[0]
        dui2 = torch.autograd.grad(dui.sum(), t, create_graph=True)[0]
        du.append(dui)
        d2u.append(dui2)
    du = torch.cat(du, dim=1)
    d2u = torch.cat(d2u, dim=1)
    return du, d2u

# IMPROVED: Better physics loss without over-normalization
def physics_msd_loss(model, t_samples, time_scale=1.0):
    """
    Compute physics-informed loss for MSD system.
    Equation: I * a + d * v + k * x = 0
    In normalized form: a + (d/I) * v + (k/I) * x = 0

    time_scale: factor to convert normalized time back to seconds
    """
    params = model.phys_params()

    t_samples.requires_grad_(True)
    preds = model(t_samples)  # normalized predictions
    vel, acc = compute_derivatives(preds, t_samples)

    # FIXED: Account for time scaling in derivatives
    # If t is normalized to [0,1] but represents T seconds:
    # du/dt_normalized = du/dt_real * T
    # d2u/dt2_normalized = d2u/dt2_real * T^2
    vel = vel / time_scale
    acc = acc / (time_scale ** 2)

    # Physics residuals (should be zero)
    # Equation: a + (d/I) * v + (k/I) * x = 0
    res_x = acc[:, 0:1] + params['d_x'] * vel[:, 0:1] + params['k_x'] * preds[:, 0:1]
    res_y = acc[:, 1:2] + params['d_y'] * vel[:, 1:2] + params['k_y'] * preds[:, 1:2]

    # Use mean squared residual
    loss_x = (res_x ** 2).mean()
    loss_y = (res_y ** 2).mean()

    return loss_x + loss_y

# IMPROVED: Better physics regularizer with physiologically reasonable bounds
def phys_regularizer(model):
    """
    Regularize physics parameters to stay in physiologically reasonable ranges.
    For balance control:
    - d/I: typically 1-20 (rad/s equivalent)
    - k/I: typically 5-200 (rad/s)^2
    """
    params = model.phys_params()
    loss = 0.0

    def soft_bound(x, low, high, sharpness=2.0):
        """Smooth penalty outside bounds"""
        below = torch.relu(low - x) ** sharpness
        above = torch.relu(x - high) ** sharpness
        return below + above

    # Physiological bounds for d/I (damping ratio)
    loss += soft_bound(params['d_x'], 0.5, 30.0)
    loss += soft_bound(params['d_y'], 0.5, 30.0)

    # Physiological bounds for k/I (natural frequency squared)
    loss += soft_bound(params['k_x'], 1.0, 400.0)
    loss += soft_bound(params['k_y'], 1.0, 400.0)

    return loss * 0.01

# -------------------------
# CLASSICAL M-ESTIMATION (Huber) FOR MSD PARAMETERS
# -------------------------
def classical_msd_estimate(cop, fs):
    """
    Classical robust estimation of MSD parameters using Huber regression.
    cop: numpy array (N,2) in original units (cm or m)
    Returns: estimated coefficients for each axis
    """
    dt = 1.0 / fs
    N = len(cop)
    # Compute derivatives using central differences
    vel = np.gradient(cop, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)

    results = {}
    for axis_idx, axis in enumerate(['x', 'y']):
        # Linear system: acc = -(d/I)*vel - (k/I)*pos
        # A = [vel, pos], y = acc, coefficients = [-(d/I), -(k/I)]
        A = np.vstack([vel[:, axis_idx], cop[:, axis_idx]]).T
        y = acc[:, axis_idx]

        try:
            hub = HuberRegressor(epsilon=1.5, max_iter=200)
            hub.fit(A, y)
            coef = hub.coef_
            # Extract d/I and k/I (negate because we formulated as negative)
            d_over_I = -coef[0]
            k_over_I = -coef[1]

            results[axis] = {
                'd_over_I': d_over_I,
                'k_over_I': k_over_I,
                'intercept': hub.intercept_,
                'score': hub.score(A, y)
            }
        except Exception as e:
            results[axis] = {'error': str(e)}

    return results

# -------------------------
# TRAINING PER RECORD
# -------------------------
def train_pinn_for_record(record_basepath, model=None, save_plots=False, out_folder="results",
                          phases=PHASES, max_points=DATA_MAX_POINTS, deterministic=True):
    """
    Train PINN on a single record.

    Args:
        deterministic: If True, use deterministic physics sampling (FIXED)
    """
    # Read data
    df, fs, subj = read_wfdb_record(record_basepath)
    ensure_cop_columns(df)
    cop = df[['COPx', 'COPy']].values.astype(np.float32)

    # Limit data points
    n = min(len(cop), max_points)
    cop = cop[:n, :]
    t = np.arange(n) / float(fs)
    t = t.reshape(-1, 1).astype(np.float32)

    # Normalization (per-record) - convert to meters if needed
    # Assume data is in cm, convert to m
    if np.abs(cop).max() > 10:  # likely in cm
        cop = cop / 100.0  # convert to meters

    mean_cop = cop.mean(axis=0, keepdims=True)
    std_cop = cop.std(axis=0, keepdims=True) + 1e-6
    cop_norm = (cop - mean_cop) / std_cop

    # Time normalization
    T_max = float(np.max(t)) if np.max(t) > 0 else 1.0
    t_norm = (t / T_max).astype(np.float32)

    device = DEVICE

    # Create model with subject-specific parameters
    if model is None:
        weight_kg = float(subj.get('weight', 75.0))
        height_m = float(subj.get('height', 1.7))
        model = PINN_MSD(weight_kg=weight_kg, height_m=height_m).to(device)
    else:
        model = model.to(device)

    t_tensor = torch.tensor(t_norm, dtype=torch.float32, device=device)
    cop_tensor = torch.tensor(cop_norm, dtype=torch.float32, device=device)

    # FIXED: Create deterministic physics sampling indices
    if deterministic:
        # Use evenly spaced indices for physics (no randomization)
        n_total = t_tensor.shape[0]
        phys_indices = np.linspace(0, n_total - 1, min(PHYS_SAMPLE, n_total), dtype=int)
        phys_indices = torch.tensor(phys_indices, dtype=torch.long, device=device)
    else:
        # Fallback to random (for comparison)
        phys_indices = None

    opt = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5,
                                                      patience=200, verbose=False)

    losses_log, data_loss_log, phys_loss_log, reg_loss_log = [], [], [], []
    damp_hist, stiff_hist = [], []

    # Live plotting setup
    plt.ion()
    fig = plt.figure(figsize=(14, 8))
    ax1 = plt.subplot2grid((3, 4), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((3, 4), (0, 2), colspan=2)
    ax3 = plt.subplot2grid((3, 4), (1, 0))
    ax4 = plt.subplot2grid((3, 4), (1, 1))
    ax5 = plt.subplot2grid((3, 4), (1, 2))
    ax6 = plt.subplot2grid((3, 4), (1, 3))
    ax7 = plt.subplot2grid((3, 4), (2, 0), colspan=4)

    step = 0
    total_epochs = sum([p[0] for p in phases])
    print(f"\nTraining {os.path.basename(record_basepath)}")
    print(f"Device: {device}, Epochs: {total_epochs}")
    print(f"Subject: weight={subj['weight']:.1f}kg, height={subj['height']:.2f}m")
    print(f"Estimated inertia: {model.inertia.item():.2f} kg*m^2")

    for phase_idx, (epochs, lambda_data, lambda_phys, lambda_reg) in enumerate(phases):
        print(f"\n--- Phase {phase_idx + 1}/{len(phases)}: λ_data={lambda_data}, λ_phys={lambda_phys}, λ_reg={lambda_reg} ---")

        for e in range(epochs):
            opt.zero_grad()
            model.train()

            # Forward pass
            t_tensor.requires_grad_(False)
            preds = model(t_tensor)
            data_loss = nn.MSELoss()(preds, cop_tensor)

            # FIXED: Deterministic physics sampling
            if deterministic:
                t_phys = t_tensor[phys_indices].detach().clone()
            else:
                phys_idx = torch.randint(0, t_tensor.shape[0],
                                        (min(PHYS_SAMPLE, t_tensor.shape[0]),),
                                        device=device)
                t_phys = t_tensor[phys_idx].detach().clone()

            t_phys.requires_grad_(True)
            phys_loss = physics_msd_loss(model, t_phys, time_scale=T_max)
            reg_loss = phys_regularizer(model)

            # Total loss
            loss = lambda_data * data_loss + lambda_phys * phys_loss + lambda_reg * reg_loss

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            scheduler.step(loss)

            # Logging
            losses_log.append(loss.item())
            data_loss_log.append(data_loss.item())
            phys_loss_log.append(phys_loss.item())
            reg_loss_log.append(reg_loss.item())

            p = model.phys_params()
            damp_hist.append((p['d_x'].item(), p['d_y'].item()))
            stiff_hist.append((p['k_x'].item(), p['k_y'].item()))
            step += 1

            # Print progress
            if step % PRINT_EVERY == 0 or (e == epochs - 1 and phase_idx == len(phases) - 1):
                d = model.phys_params()
                lr_current = opt.param_groups[0]['lr']
                print(f"Step {step:4d}/{total_epochs} | Loss={loss.item():.3e} "
                      f"(data={data_loss.item():.3e}, phys={phys_loss.item():.3e}) | "
                      f"d_x={d['d_x'].item():.2f}, k_x={d['k_x'].item():.1f} | lr={lr_current:.1e}")

            # Live plotting
            if step % PLOT_EVERY == 0 or (e == epochs - 1 and phase_idx == len(phases) - 1):
                model.eval()
                with torch.no_grad():
                    preds_full = model(t_tensor).cpu().numpy()

                # Un-normalize for plotting
                preds_unnorm = preds_full * std_cop + mean_cop
                true_unnorm = cop
                pred_x = preds_unnorm[:, 0]
                pred_y = preds_unnorm[:, 1]
                true_x = true_unnorm[:, 0]
                true_y = true_unnorm[:, 1]

                # Metrics
                rmse_x = np.sqrt(mean_squared_error(true_x, pred_x))
                r2_x = r2_score(true_x, pred_x)
                rmse_y = np.sqrt(mean_squared_error(true_y, pred_y))
                r2_y = r2_score(true_y, pred_y)

                # Update plots
                ax1.cla(); ax2.cla(); ax3.cla(); ax4.cla()
                ax5.cla(); ax6.cla(); ax7.cla()

                ax1.plot(t.squeeze(), true_x, label='True', linewidth=1)
                ax1.plot(t.squeeze(), pred_x, '--', label='PINN', linewidth=1)
                ax1.set_title(f"COP X (RMSE={rmse_x:.4f}m, R²={r2_x:.3f})")
                ax1.set_xlabel('Time (s)'); ax1.set_ylabel('COP (m)')
                ax1.legend(); ax1.grid(True, alpha=0.3)

                ax2.plot(t.squeeze(), true_y, label='True', linewidth=1)
                ax2.plot(t.squeeze(), pred_y, '--', label='PINN', linewidth=1)
                ax2.set_title(f"COP Y (RMSE={rmse_y:.4f}m, R²={r2_y:.3f})")
                ax2.set_xlabel('Time (s)'); ax2.set_ylabel('COP (m)')
                ax2.legend(); ax2.grid(True, alpha=0.3)

                ax3.scatter(true_x, pred_x, s=5, alpha=0.4)
                lims = [min(true_x.min(), pred_x.min()), max(true_x.max(), pred_x.max())]
                ax3.plot(lims, lims, 'k--', alpha=0.5)
                ax3.set_title('True vs Pred (X)')
                ax3.set_xlabel('True'); ax3.set_ylabel('Predicted')
                ax3.grid(True, alpha=0.3)
                ax3.axis('equal')

                ax4.plot(true_x, true_y, color='tab:blue', alpha=0.6, linewidth=1, label='True')
                ax4.plot(pred_x, pred_y, color='tab:red', alpha=0.8, linestyle='--',
                        linewidth=1, label='PINN')
                ax4.set_title('Stabilogram')
                ax4.set_xlabel('COP X (m)'); ax4.set_ylabel('COP Y (m)')
                ax4.legend(); ax4.axis('equal'); ax4.grid(True, alpha=0.3)

                ax5.plot(t.squeeze(), pred_x - true_x, label='err X', linewidth=0.8)
                ax5.plot(t.squeeze(), pred_y - true_y, label='err Y', linewidth=0.8)
                ax5.axhline(0, color='k', alpha=0.3, linestyle='--')
                ax5.set_title('Prediction Errors')
                ax5.set_xlabel('Time (s)'); ax5.set_ylabel('Error (m)')
                ax5.legend(); ax5.grid(True, alpha=0.3)

                # Frequency spectrum
                nperseg = min(1024, len(true_x))
                f, Pxx = welch(true_x, fs=fs, nperseg=nperseg)
                f2, Pxxp = welch(pred_x, fs=fs, nperseg=nperseg)
                ax6.semilogy(f, Pxx + 1e-15, label='True', linewidth=1)
                ax6.semilogy(f2, Pxxp + 1e-15, '--', label='PINN', linewidth=1)
                ax6.set_xlim(0, 10)
                ax6.set_title('Power Spectrum (X)')
                ax6.set_xlabel('Frequency (Hz)'); ax6.set_ylabel('PSD')
                ax6.legend(); ax6.grid(True, alpha=0.3)

                # Loss and parameters evolution
                ax7.plot(np.convolve(losses_log, np.ones(20) / 20, mode='valid'),
                        label='Total Loss', linewidth=1.5)
                ax7.set_ylabel('Loss', color='C0')
                ax7.tick_params(axis='y', labelcolor='C0')

                ax7_twin = ax7.twinx()
                dh = np.array(damp_hist)
                sh = np.array(stiff_hist)
                if len(dh) > 0:
                    ax7_twin.plot(dh[:, 0], label='d_x/I', color='C1', alpha=0.7, linewidth=1)
                    ax7_twin.plot(sh[:, 0] / 10, label='k_x/I ÷10', color='C2', alpha=0.7, linewidth=1)
                ax7_twin.set_ylabel('Physics Params', color='C1')
                ax7_twin.tick_params(axis='y', labelcolor='C1')

                ax7.set_title('Training Progress')
                ax7.set_xlabel('Training Step')
                ax7.grid(True, alpha=0.3)

                lines, labels = ax7.get_legend_handles_labels()
                lines2, labels2 = ax7_twin.get_legend_handles_labels()
                ax7.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=8)

                plt.tight_layout()
                plt.pause(0.01)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        preds_full = model(t_tensor).cpu().numpy()

    preds_unnorm = preds_full * std_cop + mean_cop
    true_unnorm = cop
    pred_x = preds_unnorm[:, 0]
    pred_y = preds_unnorm[:, 1]
    true_x = true_unnorm[:, 0]
    true_y = true_unnorm[:, 1]

    rmse_x = np.sqrt(mean_squared_error(true_x, pred_x))
    r2_x = r2_score(true_x, pred_x)
    rmse_y = np.sqrt(mean_squared_error(true_y, pred_y))
    r2_y = r2_score(true_y, pred_y)

    learned = model.phys_params_absolute()

    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    print(f"RMSE X: {rmse_x:.5f} m,  R² X: {r2_x:.4f}")
    print(f"RMSE Y: {rmse_y:.5f} m,  R² Y: {r2_y:.4f}")
    print(f"\nLearned Physics Parameters (absolute values):")
    print(f"  Inertia I = {learned['I']:.2f} kg*m²")
    print(f"  Damping   d_x = {learned['d_x']:.2f} Ns/m,  d_y = {learned['d_y']:.2f} Ns/m")
    print(f"  Stiffness k_x = {learned['k_x']:.1f} N/m,   k_y = {learned['k_y']:.1f} N/m")

    # Derived quantities
    omega_x = np.sqrt(learned['k_x'] / learned['I'])
    omega_y = np.sqrt(learned['k_y'] / learned['I'])
    freq_x = omega_x / (2 * np.pi)
    freq_y = omega_y / (2 * np.pi)
    zeta_x = learned['d_x'] / (2 * np.sqrt(learned['k_x'] * learned['I']))
    zeta_y = learned['d_y'] / (2 * np.sqrt(learned['k_y'] * learned['I']))

    print(f"\nDerived Quantities:")
    print(f"  Natural frequency: f_x = {freq_x:.3f} Hz,  f_y = {freq_y:.3f} Hz")
    print(f"  Damping ratio:     ζ_x = {zeta_x:.3f},     ζ_y = {zeta_y:.3f}")

    # Classical estimation
    print(f"\nClassical Huber Regression Estimates:")
    classical = classical_msd_estimate(true_unnorm, fs)
    for axis in ['x', 'y']:
        if 'error' not in classical[axis]:
            d_I = classical[axis]['d_over_I']
            k_I = classical[axis]['k_over_I']
            print(f"  Axis {axis}: d/I = {d_I:.2f}, k/I = {k_I:.1f}")
            if d_I > 0 and k_I > 0:
                omega_cl = np.sqrt(k_I)
                freq_cl = omega_cl / (2 * np.pi)
                zeta_cl = d_I / (2 * omega_cl)
                print(f"           freq = {freq_cl:.3f} Hz, ζ = {zeta_cl:.3f}")
        else:
            print(f"  Axis {axis}: Error - {classical[axis]['error']}")
    print("=" * 70)

    # Final static plot
    plt.ioff()
    fig2, axes = plt.subplots(2, 3, figsize=(15, 9))

    axes[0, 0].plot(t.squeeze(), true_x, 'b', label='True', linewidth=1)
    axes[0, 0].plot(t.squeeze(), pred_x, 'r--', label='PINN', linewidth=1)
    axes[0, 0].set_title(f'COP X (RMSE={rmse_x:.5f}m, R²={r2_x:.3f})')
    axes[0, 0].set_xlabel('Time (s)'); axes[0, 0].set_ylabel('COP (m)')
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(t.squeeze(), true_y, 'b', label='True', linewidth=1)
    axes[0, 1].plot(t.squeeze(), pred_y, 'r--', label='PINN', linewidth=1)
    axes[0, 1].set_title(f'COP Y (RMSE={rmse_y:.5f}m, R²={r2_y:.3f})')
    axes[0, 1].set_xlabel('Time (s)'); axes[0, 1].set_ylabel('COP (m)')
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].scatter(true_x, pred_x, s=10, alpha=0.5)
    lims = [min(true_x.min(), pred_x.min()), max(true_x.max(), pred_x.max())]
    axes[0, 2].plot(lims, lims, 'k--')
    axes[0, 2].set_title('True vs Predicted (X)')
    axes[0, 2].set_xlabel('True'); axes[0, 2].set_ylabel('Predicted')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].axis('equal')

    axes[1, 0].plot(true_x, true_y, label='True', linewidth=1)
    axes[1, 0].plot(pred_x, pred_y, '--', label='PINN', linewidth=1)
    axes[1, 0].set_title('Stabilogram')
    axes[1, 0].set_xlabel('COP X (m)'); axes[1, 0].set_ylabel('COP Y (m)')
    axes[1, 0].legend(); axes[1, 0].axis('equal'); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(t.squeeze(), pred_x - true_x, label='err X', linewidth=0.8)
    axes[1, 1].plot(t.squeeze(), pred_y - true_y, label='err Y', linewidth=0.8)
    axes[1, 1].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[1, 1].set_title('Prediction Errors')
    axes[1, 1].set_xlabel('Time (s)'); axes[1, 1].set_ylabel('Error (m)')
    axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

    nperseg = min(1024, len(true_x))
    f, Pxx = welch(true_x, fs=fs, nperseg=nperseg)
    f2, Pxxp = welch(pred_x, fs=fs, nperseg=nperseg)
    axes[1, 2].semilogy(f, Pxx + 1e-15, label='True X', linewidth=1)
    axes[1, 2].semilogy(f2, Pxxp + 1e-15, '--', label='PINN X', linewidth=1)
    axes[1, 2].set_xlim(0, 10)
    axes[1, 2].set_title('Power Spectrum (X)')
    axes[1, 2].set_xlabel('Frequency (Hz)'); axes[1, 2].set_ylabel('PSD')
    axes[1, 2].legend(); axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plots:
        os.makedirs(out_folder, exist_ok=True)
        filename = os.path.join(out_folder,
                               os.path.basename(record_basepath) + "_final_v04.png")
        fig2.savefig(filename, dpi=200, bbox_inches='tight')
        print(f"\nPlot saved to: {filename}")

    plt.show()

    metrics = {
        'rmse_x': rmse_x, 'r2_x': r2_x,
        'rmse_y': rmse_y, 'r2_y': r2_y,
        'learned': learned,
        'derived': {
            'freq_x_Hz': freq_x, 'freq_y_Hz': freq_y,
            'zeta_x': zeta_x, 'zeta_y': zeta_y
        },
        'classical': classical
    }

    return model, metrics

# -------------------------
# BATCH ANALYZE FOLDER
# -------------------------
def analyze_folder(folder_path, pattern_ext='.dat', save_plots=True, out_folder="results"):
    """
    FIXED: Process all records in folder with deterministic ordering.
    """
    # FIXED: Collect and SORT files for deterministic order
    candidates = []
    for f in os.listdir(folder_path):
        if f.endswith(pattern_ext):
            candidates.append(os.path.join(folder_path, f[:-len(pattern_ext)]))

    # CRITICAL FIX: Sort files alphabetically for reproducibility
    candidates.sort()

    print(f"\n{'=' * 70}")
    print(f"Found {len(candidates)} records to process")
    print(f"Processing in deterministic order:")
    for i, rec in enumerate(candidates[:5]):  # Show first 5
        print(f"  {i+1}. {os.path.basename(rec)}")
    if len(candidates) > 5:
        print(f"  ... and {len(candidates) - 5} more")
    print(f"{'=' * 70}\n")

    summary = []
    for idx, rec in enumerate(candidates):
        try:
            print(f"\n[{idx + 1}/{len(candidates)}] Processing: {os.path.basename(rec)}")
            print("-" * 70)

            # Reset seeds for each file to ensure reproducibility
            set_all_seeds(SEED + idx)

            model = None  # Let function create subject-specific model
            model, metrics = train_pinn_for_record(
                rec,
                model=model,
                save_plots=save_plots,
                out_folder=out_folder,
                deterministic=True  # FIXED: Use deterministic sampling
            )
            summary.append((rec, metrics))

        except Exception as e:
            print(f"\nERROR processing {os.path.basename(rec)}: {e}")
            import traceback
            traceback.print_exc()

    # Save summary table
    if summary:
        rows = []
        for rec, m in summary:
            row = {
                'record': os.path.basename(rec),
                'rmse_x_m': m['rmse_x'],
                'r2_x': m['r2_x'],
                'rmse_y_m': m['rmse_y'],
                'r2_y': m['r2_y'],
                'I_kgm2': m['learned']['I'],
                'd_x_Ns/m': m['learned']['d_x'],
                'k_x_N/m': m['learned']['k_x'],
                'd_y_Ns/m': m['learned']['d_y'],
                'k_y_N/m': m['learned']['k_y'],
                'freq_x_Hz': m['derived']['freq_x_Hz'],
                'freq_y_Hz': m['derived']['freq_y_Hz'],
                'zeta_x': m['derived']['zeta_x'],
                'zeta_y': m['derived']['zeta_y']
            }
            rows.append(row)

        os.makedirs(out_folder, exist_ok=True)
        df_summary = pd.DataFrame(rows)
        summary_file = os.path.join(out_folder, 'summary_metrics_v04.csv')
        df_summary.to_csv(summary_file, index=False, float_format='%.5f')

        print(f"\n{'=' * 70}")
        print(f"BATCH PROCESSING COMPLETE")
        print(f"{'=' * 70}")
        print(f"Processed {len(summary)}/{len(candidates)} records successfully")
        print(f"Summary saved to: {summary_file}")
        print(f"\nSummary Statistics:")
        print(df_summary.describe())
        print(f"{'=' * 70}")

    return summary

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    # Configuration
    data_folder = "/content/drive/MyDrive/human-balance-evaluation-database-1.0.0"

    # Single file example:
    # rec = os.path.join(data_folder, "subject01_recordname")  # without extension
    # train_pinn_for_record(rec, save_plots=True, out_folder="results_v04")

    # Batch process entire folder (FIXED: now deterministic)
    analyze_folder(
        data_folder,
        pattern_ext='.dat',
        save_plots=True,
        out_folder="results_v04"
    )
