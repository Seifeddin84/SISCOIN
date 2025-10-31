# Physics-Informed Neural Network for Human Balance Analysis

A comprehensive implementation of a Physics-Informed Neural Network (PINN) for analyzing human balance data from Center of Pressure (COP) measurements. This implementation identifies biomechanical parameters (stiffness and damping coefficients) while respecting the underlying physics of an inverted pendulum model.

## Features

### Core Capabilities
- **Non-zero mean COP handling**: Automatically centers COP data around the origin for physics-based modeling
- **Multi-phase training**: Adaptive loss weighting strategy that progressively emphasizes physics constraints
- **Parameter identification**: Learns stiffness and damping coefficients for both X and Y directions
- **Comprehensive validation**: Includes extensive unit tests and integration tests
- **Production-ready code**: Type hints, logging, error handling, and configurable parameters

### Key Improvements Over Original Version

#### 1. **Code Structure & Organization**
- Modular design with clear separation of concerns
- Dataclasses for configuration and results
- High-level API (`BalancePINNTrainer`) for easy usage
- Type hints throughout for better IDE support and type checking

#### 2. **Performance Optimizations**
- Configurable training data sampling to reduce memory usage
- Gradient clipping to prevent exploding gradients
- Learning rate scheduling for adaptive optimization
- Early stopping to prevent overfitting
- GPU/MPS support with automatic device detection

#### 3. **Robustness & Error Handling**
- Comprehensive input validation
- Proper exception handling with informative error messages
- Logging instead of print statements for production use
- Parameter bounds checking and regularization

#### 4. **Documentation**
- Comprehensive docstrings for all functions and classes
- Usage examples in docstrings
- Detailed README with usage guide
- Inline comments for complex logic

#### 5. **Testing**
- 50+ unit tests covering all major components
- Integration tests for end-to-end workflow
- Physics constraint validation tests
- Mock-based tests for external dependencies

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd SISCOIN
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies

Core packages:
- `numpy` - Numerical computing
- `torch` - Deep learning framework
- `pandas` - Data manipulation
- `scipy` - Signal processing (Butterworth filter)
- `wfdb` - PhysioNet WFDB data format reader
- `scikit-learn` - Metrics (MSE, R²)
- `matplotlib` - Visualization

## Usage

### Quick Start (Command Line)

Run the PINN training on a data file:

```bash
python balance_pinn.py /path/to/data/BDS00001 \
    --epochs 8000 \
    --device cuda \
    --save-model trained_model.pth \
    --save-plot results.png
```

### Python API Usage

#### Basic Example

```python
from balance_pinn import BalancePINNTrainer, TrainingConfig

# Create trainer with default configuration
trainer = BalancePINNTrainer(data_path="data/BDS00001")

# Train model
model, results = trainer.train()

# Visualize results
trainer.plot(save_path="results.png")

# Save trained model
trainer.save_model("trained_model.pth")

# Print results
print(results)
```

#### Advanced Configuration

```python
from balance_pinn import (
    BalancePINNTrainer,
    TrainingConfig,
    BalancePINN,
    create_default_training_phases
)

# Custom training configuration
config = TrainingConfig(
    epochs=10000,
    n_train_points=5000,
    device='cuda',
    lr_net=5e-4,
    lr_params=1e-2,
    enable_early_stopping=True,
    early_stopping_patience=2000
)

# Custom model architecture
model = BalancePINN(
    expected_stiffness=350.0,
    expected_damping=4.0,
    hidden_sizes=[256, 256, 128]
)

# Create trainer with custom config and model
trainer = BalancePINNTrainer(
    data_path="data/BDS00001",
    config=config,
    model=model
)

# Train
model, results = trainer.train()
```

#### Manual Training (Low-Level API)

```python
from balance_pinn import (
    load_balance_data,
    BalancePINN,
    train_pinn,
    evaluate_model,
    plot_results,
    TrainingConfig
)

# Load data
time_data, cop_data_centered, subject_info = load_balance_data(
    "data/BDS00001"
)

# Create model
model = BalancePINN(expected_stiffness=300.0, expected_damping=3.0)

# Configure training
config = TrainingConfig(epochs=8000, device='cuda')

# Train
model, history = train_pinn(
    model,
    time_data,
    cop_data_centered,
    subject_info,
    config
)

# Evaluate
results, cop_pred, cop_true = evaluate_model(
    model,
    time_data,
    cop_data_centered,
    subject_info,
    device='cuda'
)

# Visualize
plot_results(
    time_data,
    cop_true,
    cop_pred,
    history,
    results,
    save_path="results.png"
)
```

### Custom Training Phases

```python
from balance_pinn import TrainingPhase

# Define custom training schedule
custom_phases = [
    TrainingPhase(epochs=2000, w_data=1.0, w_physics=0.01),
    TrainingPhase(epochs=2000, w_data=1.0, w_physics=0.1),
    TrainingPhase(epochs=2000, w_data=1.0, w_physics=1.0),
    TrainingPhase(epochs=2000, w_data=1.0, w_physics=10.0),
]

# Train with custom phases
model, history = train_pinn(
    model,
    time_data,
    cop_data_centered,
    subject_info,
    config,
    training_phases=custom_phases
)
```

## Data Format

The code expects data in PhysioNet WFDB format with:
- `.dat` file containing signal data
- `.hea` header file with metadata

### Required Data Columns
- `COPx`: Center of Pressure X coordinate (in cm)
- `COPy`: Center of Pressure Y coordinate (in cm)

### Required Metadata (in .hea file)
- `#Height: <value>` - Subject height in cm
- `#Weight: <value>` - Subject weight in kg

### Example Header Format
```
BDS00001 2 100 100000
# Age: 25
# Height: 175
# Weight: 70
# ...
```

## Model Architecture

### Network Structure
- **Input**: Time (1D scalar)
- **Hidden Layers**: 128 → 128 → 64 neurons (configurable)
- **Activation**: Tanh (smooth, differentiable)
- **Output**: COP position (x, y) in meters

### Learnable Physical Parameters
- `stiffness_x, stiffness_y`: Stiffness coefficients (Nm/rad)
- `damping_x, damping_y`: Damping coefficients (Nm·s/rad)

### Physics Model

The model is based on an inverted pendulum:

```
I·θ'' + c·θ' + (k - m·g·h)·θ = 0
```

Where:
- `I`: Moment of inertia (kg·m²)
- `c`: Damping coefficient (Nm·s/rad)
- `k`: Stiffness coefficient (Nm/rad)
- `θ`: Angular displacement (approximated by COP/height)
- `m`: Mass (kg)
- `g`: Gravitational acceleration (9.81 m/s²)
- `h`: Height (m)

### Loss Functions

The total loss is a weighted combination of:

1. **Data Loss**: MSE between predictions and measurements
2. **Physics Loss**: Residual of the differential equation
3. **Initial Condition Loss**: Enforces correct starting position
4. **Smoothness Loss**: Regularizes velocity for smooth trajectories
5. **Parameter Regularization**: Keeps parameters in physiological ranges

## Training Strategy

### Multi-Phase Training

Training proceeds in 4 phases with progressively increasing physics weight:

| Phase | Data Weight | Physics Weight | Focus |
|-------|-------------|----------------|-------|
| 1 | 1.0 | 0.1 | Data fitting |
| 2 | 1.0 | 1.0 | Balance |
| 3 | 1.0 | 5.0 | Physics emphasis |
| 4 | 1.0 | 10.0 | Strong physics |

This strategy ensures:
1. Initial convergence to data
2. Gradual incorporation of physics constraints
3. Final solution respects both data and physics

### Optimization

- Separate optimizers for network and physical parameters
- Adaptive learning rates with ReduceLROnPlateau scheduler
- Gradient clipping to prevent instability
- Optional early stopping to prevent overfitting

## Output and Results

### Evaluation Metrics
- **RMSE**: Root Mean Squared Error for X and Y directions (cm)
- **R²**: Coefficient of determination for trajectory fit
- **Stiffness**: Identified stiffness parameters (Nm/rad)
- **Damping**: Identified damping parameters (Nm·s/rad)
- **Natural Frequency**: ω_n = √(k/I) (Hz)
- **Damping Ratio**: ζ = c/(2√(k·I)) (dimensionless)

### Visualization

The `plot_results()` function generates a 6-panel figure:

1. **COP X Trajectory**: Time series comparison
2. **COP Y Trajectory**: Time series comparison
3. **2D COP Path**: Spatial trajectory comparison
4. **Training Loss**: Loss curves over epochs
5. **Stiffness Evolution**: Parameter convergence
6. **Damping Evolution**: Parameter convergence

## Testing

### Run All Tests

```bash
python test_balance_pinn.py
```

### Run Specific Test Suite

```python
python -m unittest test_balance_pinn.TestBalancePINN
```

### Test Coverage

The test suite includes:
- Data structure tests (SubjectInfo, TrainingConfig, etc.)
- Model architecture tests
- Loss function tests
- Training component tests (EarlyStopping, phases)
- Integration tests
- Physics constraint validation

## Performance Considerations

### Memory Usage
- Training uses subsampled data (default 10,000 points)
- Adjust `n_train_points` in TrainingConfig for memory constraints
- Evaluation uses full dataset

### Training Time
- Typical training: 5-15 minutes on GPU for 8000 epochs
- CPU training: 30-60 minutes
- Use fewer epochs for quick testing

### GPU Acceleration
```python
# Automatic GPU detection
config = TrainingConfig(device='auto')

# Force GPU
config = TrainingConfig(device='cuda')

# Force CPU
config = TrainingConfig(device='cpu')

# Apple Silicon MPS
config = TrainingConfig(device='mps')
```

## Troubleshooting

### Common Issues

**Issue**: Model parameters become negative
**Solution**: Increase `w_reg` parameter regularization weight

**Issue**: Poor physics loss convergence
**Solution**: Decrease initial `w_physics` in early phases

**Issue**: Oscillatory loss curves
**Solution**: Reduce learning rates or increase gradient clipping

**Issue**: Overfitting to data, ignoring physics
**Solution**: Increase `w_physics` in later training phases

**Issue**: NaN losses during training
**Solution**:
- Reduce learning rates
- Check data for NaN/Inf values
- Increase gradient clipping threshold

## Expected Parameter Ranges

### Physiological Ranges (Human Balance)
- **Stiffness**: 100-800 Nm/rad
- **Damping**: 1-20 Nm·s/rad
- **Natural Frequency**: 0.3-1.5 Hz
- **Damping Ratio**: 0.05-0.5 (underdamped)

### Data Ranges
- **COP X**: Typically ±2 cm from mean
- **COP Y**: Typically ±3 cm from mean
- **Sampling Rate**: 50-200 Hz

## Citation

If you use this code in your research, please cite:

```bibtex
@software{balance_pinn_2025,
  title={Physics-Informed Neural Network for Human Balance Analysis},
  author={SISCOIN Team},
  year={2025},
  url={https://github.com/your-repo/SISCOIN}
}
```

## License

[Add your license information here]

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: [Your contact information]

## Acknowledgments

This implementation builds upon physics-informed machine learning concepts and balance biomechanics research. Special thanks to the PhysioNet community for providing the Human Balance Evaluation Database.

## References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686-707.

2. Winter, D. A. (1995). Human balance and posture control during standing and walking. Gait & posture, 3(4), 193-214.

3. Goldberger, A. L., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation, 101(23), e215-e220.
