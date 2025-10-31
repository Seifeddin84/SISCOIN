"""
PINN for Human Balance Analysis - Handles Non-Zero Mean COP

This module implements a Physics-Informed Neural Network (PINN) for analyzing
human balance data from Center of Pressure (COP) measurements. The model identifies
biomechanical parameters (stiffness and damping) while respecting the underlying
physics of an inverted pendulum model.

Key Features:
    - Handles non-zero mean COP by centering data around origin
    - Multi-phase training with adaptive loss weighting
    - Parameter identification for stiffness and damping coefficients
    - Comprehensive visualization and metrics

Example:
    >>> from balance_pinn import BalancePINNTrainer
    >>> trainer = BalancePINNTrainer(data_path="path/to/data")
    >>> model, results = trainer.train()
    >>> trainer.evaluate_and_plot()

Author: SISCOIN Team
Date: 2025-10-31
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wfdb
from scipy.signal import butter, filtfilt
from sklearn.metrics import mean_squared_error, r2_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Physical constants
GRAVITY = 9.81  # m/s^2

# Default parameters
DEFAULT_SAMPLE_RATE = 100  # Hz
DEFAULT_FILTER_CUTOFF = 10  # Hz
DEFAULT_FILTER_ORDER = 4

# Training defaults
DEFAULT_EPOCHS = 8000
DEFAULT_LEARNING_RATE_NET = 1e-3
DEFAULT_LEARNING_RATE_PARAMS = 5e-3
DEFAULT_GRADIENT_CLIP = 1.0

# Expected parameter ranges
EXPECTED_STIFFNESS = 300.0  # Nm/rad
EXPECTED_DAMPING = 3.0  # Nm·s/rad
MIN_STIFFNESS = 0.0
MAX_STIFFNESS = 1000.0
MIN_DAMPING = 0.0
MAX_DAMPING = 20.0


@dataclass
class SubjectInfo:
    """Container for subject-specific information and preprocessing parameters.

    Attributes:
        weight: Subject weight in kg
        height: Subject height in meters
        fs: Sampling frequency in Hz
        cop_x_mean: Mean COP X position (removed during centering) in meters
        cop_y_mean: Mean COP Y position (removed during centering) in meters
    """
    weight: float
    height: float
    fs: float
    cop_x_mean: float
    cop_y_mean: float

    @property
    def moment_of_inertia(self) -> float:
        """Calculate moment of inertia using point mass approximation."""
        return self.weight * self.height ** 2

    def __str__(self) -> str:
        return (f"Subject: {self.weight:.1f}kg, {self.height:.2f}m, "
                f"COP mean offset: ({self.cop_x_mean*100:.2f}, {self.cop_y_mean*100:.2f}) cm")


@dataclass
class TrainingConfig:
    """Configuration for PINN training.

    Attributes:
        epochs: Total number of training epochs
        n_train_points: Number of data points to use for training
        device: Device to train on ('cpu', 'cuda', or 'mps')
        lr_net: Learning rate for neural network parameters
        lr_params: Learning rate for physical parameters
        gradient_clip: Maximum gradient norm for clipping
        patience: Patience for learning rate scheduler
        scheduler_factor: Factor to reduce learning rate
        enable_early_stopping: Whether to enable early stopping
        early_stopping_patience: Patience for early stopping
        early_stopping_delta: Minimum change to qualify as improvement
    """
    epochs: int = DEFAULT_EPOCHS
    n_train_points: int = 10000
    device: str = 'cpu'
    lr_net: float = DEFAULT_LEARNING_RATE_NET
    lr_params: float = DEFAULT_LEARNING_RATE_PARAMS
    gradient_clip: float = DEFAULT_GRADIENT_CLIP
    patience: int = 500
    scheduler_factor: float = 0.7
    enable_early_stopping: bool = True
    early_stopping_patience: int = 1000
    early_stopping_delta: float = 1e-6

    def __post_init__(self):
        """Validate configuration and set device."""
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.n_train_points <= 0:
            raise ValueError(f"n_train_points must be positive, got {self.n_train_points}")

        # Auto-detect best available device if not specified
        if self.device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'

        logger.info(f"Training device: {self.device}")


@dataclass
class TrainingPhase:
    """Configuration for a single training phase.

    Attributes:
        epochs: Number of epochs for this phase
        w_data: Weight for data fitting loss
        w_physics: Weight for physics loss
        w_init: Weight for initial condition loss
        w_smooth: Weight for smoothness regularization
        w_reg: Weight for parameter regularization
    """
    epochs: int
    w_data: float = 1.0
    w_physics: float = 1.0
    w_init: float = 1.0
    w_smooth: float = 0.01
    w_reg: float = 0.01


# ============================================================================
# DATA LOADING
# ============================================================================

def load_balance_data(
    file_path: str,
    filter_cutoff: float = DEFAULT_FILTER_CUTOFF,
    filter_order: int = DEFAULT_FILTER_ORDER
) -> Tuple[np.ndarray, np.ndarray, SubjectInfo]:
    """Load and preprocess balance data from WFDB format.

    This function loads COP data, converts units to SI (meters), applies
    a low-pass Butterworth filter, and centers the data by removing the mean.

    Args:
        file_path: Path to the data file (without extension)
        filter_cutoff: Cutoff frequency for low-pass filter in Hz
        filter_order: Order of the Butterworth filter

    Returns:
        Tuple containing:
            - time: Time array in seconds, shape (N,)
            - cop_data_centered: Centered COP data in meters, shape (N, 2)
            - subject_info: SubjectInfo object with metadata

    Raises:
        FileNotFoundError: If data files are not found
        ValueError: If data format is invalid or missing required fields

    Example:
        >>> time, cop, info = load_balance_data("data/BDS00001")
        >>> print(f"Loaded {len(time)} samples for {info}")
    """
    try:
        # Load WFDB record
        record = wfdb.rdrecord(file_path)
        data = pd.DataFrame(record.p_signal, columns=record.sig_name)
        sample_rate = record.fs

        # Validate required columns
        if 'COPx' not in data.columns or 'COPy' not in data.columns:
            raise ValueError("Data must contain 'COPx' and 'COPy' columns")

        # Extract subject information from header
        header_path = f"{file_path}.hea"
        if not os.path.exists(header_path):
            raise FileNotFoundError(f"Header file not found: {header_path}")

        with open(header_path, 'r') as f:
            content = f.read()

        import re
        height_match = re.search(r'#Height:\s*(\d+\.?\d*)', content)
        weight_match = re.search(r'#Weight:\s*(\d+\.?\d*)', content)

        if not height_match or not weight_match:
            raise ValueError("Header file must contain Height and Weight information")

        height = float(height_match.group(1)) / 100.0  # cm to m
        weight = float(weight_match.group(1))  # kg

        # Extract and convert COP data (cm to m)
        cop_x = data['COPx'].values / 100.0
        cop_y = data['COPy'].values / 100.0

        # Apply Butterworth low-pass filter
        nyquist = 0.5 * sample_rate
        normalized_cutoff = filter_cutoff / nyquist

        if normalized_cutoff >= 1.0:
            logger.warning(f"Filter cutoff ({filter_cutoff} Hz) exceeds Nyquist frequency "
                         f"({nyquist} Hz). Skipping filtering.")
            cop_x_filtered = cop_x
            cop_y_filtered = cop_y
        else:
            b, a = butter(filter_order, normalized_cutoff, btype='low')
            cop_x_filtered = filtfilt(b, a, cop_x)
            cop_y_filtered = filtfilt(b, a, cop_y)

        # Center data by removing mean (critical for physics model)
        cop_x_mean = np.mean(cop_x_filtered)
        cop_y_mean = np.mean(cop_y_filtered)
        cop_x_centered = cop_x_filtered - cop_x_mean
        cop_y_centered = cop_y_filtered - cop_y_mean

        # Create time array
        time = np.linspace(0, len(data) / sample_rate, len(data))

        # Stack centered COP data
        cop_data_centered = np.column_stack([cop_x_centered, cop_y_centered])

        # Create subject info object
        subject_info = SubjectInfo(
            weight=weight,
            height=height,
            fs=sample_rate,
            cop_x_mean=cop_x_mean,
            cop_y_mean=cop_y_mean
        )

        logger.info(f"Loaded {len(time)} samples from {file_path}")
        logger.info(f"  {subject_info}")
        logger.info(f"  COP range (centered): X=[{cop_x_centered.min()*100:.2f}, "
                   f"{cop_x_centered.max()*100:.2f}] cm")

        return time, cop_data_centered, subject_info

    except Exception as e:
        logger.error(f"Failed to load data from {file_path}: {e}")
        raise


# ============================================================================
# NEURAL NETWORK MODEL
# ============================================================================

class BalancePINN(nn.Module):
    """Physics-Informed Neural Network for balance analysis.

    This network learns to predict COP trajectories while respecting the
    underlying physics of an inverted pendulum model. It simultaneously
    learns the trajectory and identifies biomechanical parameters.

    Architecture:
        - Input: Time (1D)
        - Hidden: 128 -> 128 -> 64 neurons with Tanh activation
        - Output: COP position (x, y) in meters
        - Parameters: Stiffness and damping coefficients (x, y directions)

    Attributes:
        net: Sequential neural network for trajectory prediction
        stiffness_x: Learnable stiffness parameter in X direction (Nm/rad)
        stiffness_y: Learnable stiffness parameter in Y direction (Nm/rad)
        damping_x: Learnable damping parameter in X direction (Nm·s/rad)
        damping_y: Learnable damping parameter in Y direction (Nm·s/rad)

    Example:
        >>> model = BalancePINN(expected_stiffness=300.0, expected_damping=3.0)
        >>> t = torch.tensor([[0.0], [0.1], [0.2]])
        >>> cop = model(t)  # Shape: (3, 2)
    """

    def __init__(
        self,
        expected_stiffness: float = EXPECTED_STIFFNESS,
        expected_damping: float = EXPECTED_DAMPING,
        hidden_sizes: List[int] = [128, 128, 64]
    ):
        """Initialize the PINN model.

        Args:
            expected_stiffness: Initial guess for stiffness (Nm/rad)
            expected_damping: Initial guess for damping (Nm·s/rad)
            hidden_sizes: List of hidden layer sizes
        """
        super().__init__()

        # Build network architecture
        layers = []
        input_size = 1
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.Tanh()
            ])
            input_size = hidden_size
        layers.append(nn.Linear(input_size, 2))  # Output: (x, y)

        self.net = nn.Sequential(*layers)

        # Initialize weights using Xavier initialization
        self._initialize_weights()

        # Initialize physical parameters as learnable
        self.stiffness_x = nn.Parameter(torch.tensor(expected_stiffness))
        self.stiffness_y = nn.Parameter(torch.tensor(expected_stiffness))
        self.damping_x = nn.Parameter(torch.tensor(expected_damping))
        self.damping_y = nn.Parameter(torch.tensor(expected_damping))

        logger.info(f"Initialized BalancePINN with architecture: {hidden_sizes}")
        logger.info(f"  Initial stiffness: {expected_stiffness:.1f} Nm/rad")
        logger.info(f"  Initial damping: {expected_damping:.2f} Nm·s/rad")

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

        # Scale output layer for small oscillations
        with torch.no_grad():
            self.net[-1].weight.data *= 0.01

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            t: Time tensor of shape (N, 1)

        Returns:
            COP predictions of shape (N, 2) where columns are (x, y)
        """
        return self.net(t)

    def get_parameters_dict(self) -> Dict[str, float]:
        """Get current physical parameters as a dictionary.

        Returns:
            Dictionary with keys: stiffness_x, stiffness_y, damping_x, damping_y
        """
        return {
            'stiffness_x': self.stiffness_x.item(),
            'stiffness_y': self.stiffness_y.item(),
            'damping_x': self.damping_x.item(),
            'damping_y': self.damping_y.item()
        }


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def compute_physics_loss(
    model: BalancePINN,
    t: torch.Tensor,
    subject_info: SubjectInfo
) -> torch.Tensor:
    """Compute physics-informed loss based on inverted pendulum dynamics.

    The physics loss enforces the equation of motion for an inverted pendulum:
        I·θ'' + c·θ' + (k - m·g·h)·θ = 0

    Where:
        - I: Moment of inertia
        - c: Damping coefficient
        - k: Stiffness coefficient
        - θ: Angular displacement (approximated by COP/height)

    Args:
        model: PINN model
        t: Time tensor of shape (N, 1), requires gradient
        subject_info: Subject information with mass and height

    Returns:
        Scalar physics loss value
    """
    t = t.requires_grad_(True)

    # Get position predictions
    pos = model(t)
    pos_x, pos_y = pos[:, 0:1], pos[:, 1:2]

    # Physical constants
    I = subject_info.moment_of_inertia
    g = GRAVITY
    h = subject_info.height

    # Compute first derivatives (velocities)
    vel_x = torch.autograd.grad(
        pos_x.sum(), t, create_graph=True, retain_graph=True
    )[0]
    vel_y = torch.autograd.grad(
        pos_y.sum(), t, create_graph=True, retain_graph=True
    )[0]

    # Compute second derivatives (accelerations)
    accel_x = torch.autograd.grad(
        vel_x.sum(), t, create_graph=True, retain_graph=True
    )[0]
    accel_y = torch.autograd.grad(
        vel_y.sum(), t, create_graph=True, retain_graph=True
    )[0]

    # Physics equations (centered data, equilibrium at origin)
    # I·θ'' + c·θ' + (k - m·g·h)·θ = 0
    physics_residual_x = (
        accel_x +
        (model.damping_x / I) * vel_x +
        (model.stiffness_x / I - g / h) * pos_x
    )
    physics_residual_y = (
        accel_y +
        (model.damping_y / I) * vel_y +
        (model.stiffness_y / I - g / h) * pos_y
    )

    # Return mean squared residual
    return torch.mean(physics_residual_x ** 2) + torch.mean(physics_residual_y ** 2)


def compute_data_loss(
    model: BalancePINN,
    t: torch.Tensor,
    cop_true: torch.Tensor
) -> torch.Tensor:
    """Compute data fitting loss (MSE between predictions and measurements).

    Args:
        model: PINN model
        t: Time tensor of shape (N, 1)
        cop_true: True COP measurements of shape (N, 2)

    Returns:
        Scalar MSE loss value
    """
    cop_pred = model(t)
    return torch.mean((cop_pred - cop_true) ** 2)


def compute_initial_condition_loss(
    model: BalancePINN,
    t0: torch.Tensor,
    cop0: torch.Tensor
) -> torch.Tensor:
    """Enforce initial condition constraints.

    Args:
        model: PINN model
        t0: Initial time tensor of shape (1, 1)
        cop0: Initial COP value of shape (1, 2)

    Returns:
        Scalar loss value
    """
    pred0 = model(t0)
    return torch.mean((pred0 - cop0) ** 2)


def compute_smoothness_loss(model: BalancePINN, t: torch.Tensor) -> torch.Tensor:
    """Regularization term to encourage smooth trajectories.

    Penalizes large velocities to prevent oscillatory artifacts.

    Args:
        model: PINN model
        t: Time tensor of shape (N, 1), requires gradient

    Returns:
        Scalar smoothness loss value
    """
    t = t.requires_grad_(True)
    pos = model(t)

    # Compute velocities
    vel_x = torch.autograd.grad(
        pos[:, 0].sum(), t, create_graph=True, retain_graph=True
    )[0]
    vel_y = torch.autograd.grad(
        pos[:, 1].sum(), t, create_graph=True, retain_graph=True
    )[0]

    # Penalize large velocities
    return 0.001 * (torch.mean(vel_x ** 2) + torch.mean(vel_y ** 2))


def compute_parameter_regularization(model: BalancePINN) -> torch.Tensor:
    """Regularization to keep parameters in physiologically reasonable ranges.

    Stiffness: 0-1000 Nm/rad
    Damping: 0-20 Nm·s/rad

    Args:
        model: PINN model

    Returns:
        Scalar regularization loss value
    """
    reg = torch.tensor(0.0, device=next(model.parameters()).device)

    # Penalize negative values (hard constraint)
    reg += torch.relu(-model.stiffness_x) ** 2
    reg += torch.relu(-model.stiffness_y) ** 2
    reg += torch.relu(-model.damping_x) ** 2
    reg += torch.relu(-model.damping_y) ** 2

    # Penalize values outside expected range (soft constraint)
    reg += 0.0001 * torch.relu(model.stiffness_x - MAX_STIFFNESS) ** 2
    reg += 0.0001 * torch.relu(model.stiffness_y - MAX_STIFFNESS) ** 2
    reg += 0.0001 * torch.relu(model.damping_x - MAX_DAMPING) ** 2
    reg += 0.0001 * torch.relu(model.damping_y - MAX_DAMPING) ** 2

    return reg


# ============================================================================
# TRAINING
# ============================================================================

class EarlyStopping:
    """Early stopping handler to prevent overfitting.

    Attributes:
        patience: Number of epochs to wait before stopping
        delta: Minimum change to qualify as improvement
        counter: Current count of epochs without improvement
        best_loss: Best loss value seen so far
        should_stop: Whether early stopping criteria is met
    """

    def __init__(self, patience: int = 1000, delta: float = 1e-6):
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait
            delta: Minimum improvement threshold
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False

    def __call__(self, loss: float) -> bool:
        """Check if training should stop.

        Args:
            loss: Current loss value

        Returns:
            True if training should stop, False otherwise
        """
        if loss < self.best_loss - self.delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def create_default_training_phases(total_epochs: int) -> List[TrainingPhase]:
    """Create default multi-phase training schedule.

    The training is divided into 4 phases with progressively increasing
    physics loss weight to balance data fitting and physics constraint.

    Phase 1: Focus on data fitting
    Phase 2: Balance data and physics
    Phase 3: Emphasize physics
    Phase 4: Strong physics enforcement

    Args:
        total_epochs: Total number of training epochs

    Returns:
        List of TrainingPhase objects
    """
    epochs_per_phase = total_epochs // 4

    return [
        TrainingPhase(
            epochs=epochs_per_phase,
            w_data=1.0, w_physics=0.1, w_init=10.0, w_smooth=0.1, w_reg=0.01
        ),
        TrainingPhase(
            epochs=epochs_per_phase,
            w_data=1.0, w_physics=1.0, w_init=5.0, w_smooth=0.05, w_reg=0.01
        ),
        TrainingPhase(
            epochs=epochs_per_phase,
            w_data=1.0, w_physics=5.0, w_init=1.0, w_smooth=0.01, w_reg=0.01
        ),
        TrainingPhase(
            epochs=total_epochs - 3 * epochs_per_phase,  # Remainder
            w_data=1.0, w_physics=10.0, w_init=0.5, w_smooth=0.01, w_reg=0.01
        ),
    ]


def train_pinn(
    model: BalancePINN,
    time_data: np.ndarray,
    cop_data: np.ndarray,
    subject_info: SubjectInfo,
    config: TrainingConfig,
    training_phases: Optional[List[TrainingPhase]] = None
) -> Tuple[BalancePINN, Dict[str, List]]:
    """Train the PINN model with multi-phase approach.

    Args:
        model: Initialized PINN model
        time_data: Time array of shape (N,)
        cop_data: COP data of shape (N, 2)
        subject_info: Subject information
        config: Training configuration
        training_phases: Optional custom training phases

    Returns:
        Tuple containing:
            - Trained model
            - Training history dictionary with loss curves and parameters

    Example:
        >>> config = TrainingConfig(epochs=8000, device='cuda')
        >>> model = BalancePINN()
        >>> model, history = train_pinn(model, time, cop, info, config)
    """
    device = torch.device(config.device)
    model = model.to(device)

    # Sample training points
    n_points = min(len(time_data), config.n_train_points)
    indices = np.linspace(0, len(time_data) - 1, n_points, dtype=int)

    # Prepare tensors
    t_train = torch.tensor(
        time_data[indices].reshape(-1, 1), dtype=torch.float32
    ).to(device)
    cop_train = torch.tensor(
        cop_data[indices], dtype=torch.float32
    ).to(device)

    # Initial condition
    t0 = t_train[:1]
    cop0 = cop_train[:1]

    # Setup optimizers
    opt_net = torch.optim.Adam(model.net.parameters(), lr=config.lr_net)
    opt_params = torch.optim.Adam(
        [model.stiffness_x, model.stiffness_y, model.damping_x, model.damping_y],
        lr=config.lr_params
    )

    # Learning rate schedulers
    scheduler_net = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_net, patience=config.patience, factor=config.scheduler_factor
    )
    scheduler_params = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_params, patience=config.patience, factor=config.scheduler_factor
    )

    # Early stopping
    early_stopping = None
    if config.enable_early_stopping:
        early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            delta=config.early_stopping_delta
        )

    # Training history
    history = {
        'epoch': [], 'total_loss': [], 'data_loss': [], 'physics_loss': [],
        'stiffness_x': [], 'stiffness_y': [], 'damping_x': [], 'damping_y': []
    }

    # Use default phases if not provided
    if training_phases is None:
        training_phases = create_default_training_phases(config.epochs)

    logger.info(f"\nStarting training for {config.epochs} epochs")
    logger.info(f"  Training points: {n_points}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Phases: {len(training_phases)}")

    epoch = 0
    for phase_idx, phase in enumerate(training_phases):
        logger.info(f"\nPhase {phase_idx + 1}/{len(training_phases)}: "
                   f"w_physics={phase.w_physics}")

        for _ in range(phase.epochs):
            model.train()

            # Zero gradients
            opt_net.zero_grad()
            opt_params.zero_grad()

            # Compute losses
            loss_data = compute_data_loss(model, t_train, cop_train)
            loss_physics = compute_physics_loss(
                model, t_train.clone(), subject_info
            )
            loss_init = compute_initial_condition_loss(model, t0, cop0)
            loss_smooth = compute_smoothness_loss(model, t_train.clone())
            loss_reg = compute_parameter_regularization(model)

            # Weighted total loss
            loss_total = (
                phase.w_data * loss_data +
                phase.w_physics * loss_physics +
                phase.w_init * loss_init +
                phase.w_smooth * loss_smooth +
                phase.w_reg * loss_reg
            )

            # Backward pass
            loss_total.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config.gradient_clip
            )

            # Update parameters
            opt_net.step()
            opt_params.step()

            # Update schedulers
            scheduler_net.step(loss_total)
            scheduler_params.step(loss_total)

            # Record history
            if epoch % 100 == 0:
                history['epoch'].append(epoch)
                history['total_loss'].append(loss_total.item())
                history['data_loss'].append(loss_data.item())
                history['physics_loss'].append(loss_physics.item())
                history['stiffness_x'].append(model.stiffness_x.item())
                history['stiffness_y'].append(model.stiffness_y.item())
                history['damping_x'].append(model.damping_x.item())
                history['damping_y'].append(model.damping_y.item())

                if epoch % 1000 == 0:
                    logger.info(
                        f"  Epoch {epoch:5d} | Data: {loss_data:.6f} | "
                        f"Physics: {loss_physics:.6f} | "
                        f"Kx: {model.stiffness_x.item():.1f} | "
                        f"Dx: {model.damping_x.item():.2f}"
                    )

            # Check early stopping
            if early_stopping is not None and early_stopping(loss_total.item()):
                logger.info(f"\nEarly stopping triggered at epoch {epoch}")
                return model, history

            epoch += 1

    logger.info(f"\nTraining completed after {epoch} epochs")
    return model, history


# ============================================================================
# EVALUATION
# ============================================================================

@dataclass
class EvaluationResults:
    """Container for model evaluation results.

    Attributes:
        rmse_x: Root mean squared error for X direction (meters)
        rmse_y: Root mean squared error for Y direction (meters)
        r2_x: R² score for X direction
        r2_y: R² score for Y direction
        stiffness_x: Identified stiffness in X direction (Nm/rad)
        stiffness_y: Identified stiffness in Y direction (Nm/rad)
        damping_x: Identified damping in X direction (Nm·s/rad)
        damping_y: Identified damping in Y direction (Nm·s/rad)
        omega_n_x: Natural frequency in X direction (Hz)
        omega_n_y: Natural frequency in Y direction (Hz)
        zeta_x: Damping ratio in X direction (dimensionless)
        zeta_y: Damping ratio in Y direction (dimensionless)
    """
    rmse_x: float
    rmse_y: float
    r2_x: float
    r2_y: float
    stiffness_x: float
    stiffness_y: float
    damping_x: float
    damping_y: float
    omega_n_x: float
    omega_n_y: float
    zeta_x: float
    zeta_y: float

    def to_dict(self) -> Dict[str, float]:
        """Convert results to dictionary."""
        return {
            'rmse_x': self.rmse_x,
            'rmse_y': self.rmse_y,
            'r2_x': self.r2_x,
            'r2_y': self.r2_y,
            'stiffness_x': self.stiffness_x,
            'stiffness_y': self.stiffness_y,
            'damping_x': self.damping_x,
            'damping_y': self.damping_y,
            'omega_n_x': self.omega_n_x,
            'omega_n_y': self.omega_n_y,
            'zeta_x': self.zeta_x,
            'zeta_y': self.zeta_y,
        }

    def __str__(self) -> str:
        """Format results as readable string."""
        return (
            "="*60 + "\n"
            "EVALUATION RESULTS\n"
            "="*60 + "\n"
            f"RMSE X: {self.rmse_x*100:.3f} cm | R² X: {self.r2_x:.4f}\n"
            f"RMSE Y: {self.rmse_y*100:.3f} cm | R² Y: {self.r2_y:.4f}\n"
            f"\nIdentified Parameters:\n"
            f"  Stiffness X: {self.stiffness_x:.2f} Nm/rad\n"
            f"  Stiffness Y: {self.stiffness_y:.2f} Nm/rad\n"
            f"  Damping X: {self.damping_x:.3f} Nm·s/rad\n"
            f"  Damping Y: {self.damping_y:.3f} Nm·s/rad\n"
            f"\nDerived Quantities:\n"
            f"  Natural Freq X: {self.omega_n_x:.3f} Hz | "
            f"Damping Ratio X: {self.zeta_x:.4f}\n"
            f"  Natural Freq Y: {self.omega_n_y:.3f} Hz | "
            f"Damping Ratio Y: {self.zeta_y:.4f}\n"
            "="*60
        )


def evaluate_model(
    model: BalancePINN,
    time_data: np.ndarray,
    cop_data_centered: np.ndarray,
    subject_info: SubjectInfo,
    device: str = 'cpu'
) -> Tuple[EvaluationResults, np.ndarray, np.ndarray]:
    """Evaluate trained model on full dataset.

    Args:
        model: Trained PINN model
        time_data: Time array of shape (N,)
        cop_data_centered: Centered COP data of shape (N, 2)
        subject_info: Subject information (contains mean offsets)
        device: Device for computation

    Returns:
        Tuple containing:
            - EvaluationResults object with metrics
            - Predicted COP with mean restored, shape (N, 2)
            - True COP with mean restored, shape (N, 2)
    """
    model.eval()
    device_obj = torch.device(device)

    # Prepare tensors
    t_test = torch.tensor(
        time_data.reshape(-1, 1), dtype=torch.float32
    ).to(device_obj)
    cop_test_centered = torch.tensor(
        cop_data_centered, dtype=torch.float32
    ).to(device_obj)

    # Get predictions
    with torch.no_grad():
        cop_pred_centered = model(t_test).cpu().numpy()

    cop_true_centered = cop_test_centered.cpu().numpy()

    # Restore means for display
    cop_pred = cop_pred_centered.copy()
    cop_pred[:, 0] += subject_info.cop_x_mean
    cop_pred[:, 1] += subject_info.cop_y_mean

    cop_true = cop_true_centered.copy()
    cop_true[:, 0] += subject_info.cop_x_mean
    cop_true[:, 1] += subject_info.cop_y_mean

    # Compute metrics on centered data
    rmse_x = np.sqrt(mean_squared_error(
        cop_true_centered[:, 0], cop_pred_centered[:, 0]
    ))
    rmse_y = np.sqrt(mean_squared_error(
        cop_true_centered[:, 1], cop_pred_centered[:, 1]
    ))
    r2_x = r2_score(cop_true_centered[:, 0], cop_pred_centered[:, 0])
    r2_y = r2_score(cop_true_centered[:, 1], cop_pred_centered[:, 1])

    # Extract parameters
    params = model.get_parameters_dict()

    # Calculate derived quantities
    I = subject_info.moment_of_inertia
    omega_n_x = np.sqrt(params['stiffness_x'] / I)
    omega_n_y = np.sqrt(params['stiffness_y'] / I)
    zeta_x = params['damping_x'] / (2 * np.sqrt(params['stiffness_x'] * I))
    zeta_y = params['damping_y'] / (2 * np.sqrt(params['stiffness_y'] * I))

    # Create results object
    results = EvaluationResults(
        rmse_x=rmse_x,
        rmse_y=rmse_y,
        r2_x=r2_x,
        r2_y=r2_y,
        stiffness_x=params['stiffness_x'],
        stiffness_y=params['stiffness_y'],
        damping_x=params['damping_x'],
        damping_y=params['damping_y'],
        omega_n_x=omega_n_x,
        omega_n_y=omega_n_y,
        zeta_x=zeta_x,
        zeta_y=zeta_y
    )

    logger.info(f"\n{results}")

    return results, cop_pred, cop_true


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(
    time_data: np.ndarray,
    cop_true: np.ndarray,
    cop_pred: np.ndarray,
    history: Dict[str, List],
    results: EvaluationResults,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Create comprehensive visualization of results.

    Args:
        time_data: Time array
        cop_true: True COP trajectories (with mean restored)
        cop_pred: Predicted COP trajectories (with mean restored)
        history: Training history
        results: Evaluation results
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # COP X trajectory
    ax = axes[0, 0]
    ax.plot(time_data, cop_true[:, 0] * 100, 'b-', alpha=0.7,
            label='Measured', linewidth=1)
    ax.plot(time_data, cop_pred[:, 0] * 100, 'r--', alpha=0.7,
            label='PINN', linewidth=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('COP X (cm)')
    ax.set_title(f'COP X Trajectory (R²={results.r2_x:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # COP Y trajectory
    ax = axes[0, 1]
    ax.plot(time_data, cop_true[:, 1] * 100, 'b-', alpha=0.7,
            label='Measured', linewidth=1)
    ax.plot(time_data, cop_pred[:, 1] * 100, 'r--', alpha=0.7,
            label='PINN', linewidth=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('COP Y (cm)')
    ax.set_title(f'COP Y Trajectory (R²={results.r2_y:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2D path
    ax = axes[0, 2]
    ax.plot(cop_true[:, 0] * 100, cop_true[:, 1] * 100, 'b-',
            alpha=0.5, label='Measured', linewidth=0.5)
    ax.plot(cop_pred[:, 0] * 100, cop_pred[:, 1] * 100, 'r-',
            alpha=0.5, label='PINN', linewidth=0.5)
    ax.set_xlabel('COP X (cm)')
    ax.set_ylabel('COP Y (cm)')
    ax.set_title('2D COP Path')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    # Training loss
    ax = axes[1, 0]
    ax.semilogy(history['epoch'], history['total_loss'], label='Total', linewidth=2)
    ax.semilogy(history['epoch'], history['data_loss'], label='Data', linewidth=1.5)
    ax.semilogy(history['epoch'], history['physics_loss'], label='Physics', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Stiffness evolution
    ax = axes[1, 1]
    ax.plot(history['epoch'], history['stiffness_x'],
            label=f"Kx (final: {results.stiffness_x:.1f})", linewidth=2)
    ax.plot(history['epoch'], history['stiffness_y'],
            label=f"Ky (final: {results.stiffness_y:.1f})", linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Stiffness (Nm/rad)')
    ax.set_title('Stiffness Parameters')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Damping evolution
    ax = axes[1, 2]
    ax.plot(history['epoch'], history['damping_x'],
            label=f"Dx (final: {results.damping_x:.2f})", linewidth=2)
    ax.plot(history['epoch'], history['damping_y'],
            label=f"Dy (final: {results.damping_y:.2f})", linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Damping (Nm·s/rad)')
    ax.set_title('Damping Parameters')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")

    return fig


# ============================================================================
# HIGH-LEVEL INTERFACE
# ============================================================================

class BalancePINNTrainer:
    """High-level interface for training and evaluating balance PINN.

    This class provides a convenient API for the entire workflow:
    loading data, training the model, evaluating results, and visualization.

    Example:
        >>> trainer = BalancePINNTrainer(
        ...     data_path="data/BDS00001",
        ...     config=TrainingConfig(epochs=8000, device='cuda')
        ... )
        >>> model, results = trainer.train()
        >>> trainer.plot()
        >>> trainer.save_model("model.pth")
    """

    def __init__(
        self,
        data_path: str,
        config: Optional[TrainingConfig] = None,
        model: Optional[BalancePINN] = None
    ):
        """Initialize trainer.

        Args:
            data_path: Path to data file (without extension)
            config: Training configuration (uses defaults if None)
            model: Pre-initialized model (creates new if None)
        """
        self.data_path = data_path
        self.config = config or TrainingConfig()

        # Load data
        logger.info(f"Loading data from {data_path}")
        self.time_data, self.cop_data_centered, self.subject_info = \
            load_balance_data(data_path)

        # Initialize model
        if model is None:
            self.model = BalancePINN()
        else:
            self.model = model

        # Placeholders for results
        self.history = None
        self.results = None
        self.cop_pred = None
        self.cop_true = None

    def train(self) -> Tuple[BalancePINN, EvaluationResults]:
        """Train the model and evaluate on full dataset.

        Returns:
            Tuple of (trained model, evaluation results)
        """
        # Train
        self.model, self.history = train_pinn(
            self.model,
            self.time_data,
            self.cop_data_centered,
            self.subject_info,
            self.config
        )

        # Evaluate
        self.results, self.cop_pred, self.cop_true = evaluate_model(
            self.model,
            self.time_data,
            self.cop_data_centered,
            self.subject_info,
            self.config.device
        )

        return self.model, self.results

    def plot(self, save_path: Optional[str] = None) -> plt.Figure:
        """Create visualization of results.

        Args:
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure

        Raises:
            RuntimeError: If called before training
        """
        if self.results is None:
            raise RuntimeError("Must call train() before plot()")

        return plot_results(
            self.time_data,
            self.cop_true,
            self.cop_pred,
            self.history,
            self.results,
            save_path
        )

    def save_model(self, path: str):
        """Save trained model to file.

        Args:
            path: Path to save model

        Raises:
            RuntimeError: If called before training
        """
        if self.model is None:
            raise RuntimeError("Must call train() before save_model()")

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'subject_info': self.subject_info,
            'results': self.results.to_dict() if self.results else None,
        }, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load trained model from file.

        Args:
            path: Path to model file
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(
    data_path: str,
    epochs: int = DEFAULT_EPOCHS,
    device: str = 'auto',
    save_model_path: Optional[str] = None,
    save_plot_path: Optional[str] = None
) -> Tuple[BalancePINN, EvaluationResults, Dict]:
    """Main execution function.

    Args:
        data_path: Path to data file (without extension)
        epochs: Number of training epochs
        device: Device for training ('cpu', 'cuda', 'mps', or 'auto')
        save_model_path: Optional path to save trained model
        save_plot_path: Optional path to save results plot

    Returns:
        Tuple of (trained model, evaluation results, training history)

    Example:
        >>> model, results, history = main(
        ...     "data/BDS00001",
        ...     epochs=8000,
        ...     device='cuda',
        ...     save_plot_path="results.png"
        ... )
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create trainer
    config = TrainingConfig(epochs=epochs, device=device)
    trainer = BalancePINNTrainer(data_path=data_path, config=config)

    # Train and evaluate
    model, results = trainer.train()

    # Visualize
    trainer.plot(save_path=save_plot_path)

    # Save model if requested
    if save_model_path:
        trainer.save_model(save_model_path)

    return model, results, trainer.history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train PINN for human balance analysis"
    )
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to data file (without extension)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cpu", "cuda", "mps", "auto"],
        help="Device for training (default: auto)"
    )
    parser.add_argument(
        "--save-model",
        type=str,
        default=None,
        help="Path to save trained model"
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default=None,
        help="Path to save results plot"
    )

    args = parser.parse_args()

    main(
        data_path=args.data_path,
        epochs=args.epochs,
        device=args.device,
        save_model_path=args.save_model,
        save_plot_path=args.save_plot
    )
