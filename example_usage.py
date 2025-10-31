"""
Example usage of Balance PINN module.

This script demonstrates various ways to use the balance_pinn module
for training and analyzing human balance data.
"""

import numpy as np
import torch
from balance_pinn import (
    BalancePINNTrainer,
    TrainingConfig,
    BalancePINN,
    TrainingPhase,
    SubjectInfo,
    load_balance_data,
    train_pinn,
    evaluate_model,
    plot_results
)


def example_1_quick_start():
    """Example 1: Quickest way to train a model."""
    print("="*60)
    print("Example 1: Quick Start")
    print("="*60)

    # Simplest usage - just provide data path
    data_path = "/path/to/your/data/BDS00001"  # Update this path

    try:
        trainer = BalancePINNTrainer(data_path=data_path)
        model, results = trainer.train()
        trainer.plot(save_path="example1_results.png")
        print(f"\n{results}")
    except FileNotFoundError:
        print(f"Data file not found: {data_path}")
        print("Please update the data_path variable with your actual data location")


def example_2_custom_config():
    """Example 2: Using custom configuration."""
    print("\n" + "="*60)
    print("Example 2: Custom Configuration")
    print("="*60)

    data_path = "/path/to/your/data/BDS00001"  # Update this path

    # Create custom configuration
    config = TrainingConfig(
        epochs=5000,  # Fewer epochs for faster training
        n_train_points=5000,  # Use fewer points
        device='auto',  # Automatically select best device
        lr_net=5e-4,  # Lower learning rate for network
        lr_params=1e-2,  # Higher learning rate for physics params
        enable_early_stopping=True,
        early_stopping_patience=1500
    )

    try:
        trainer = BalancePINNTrainer(data_path=data_path, config=config)
        model, results = trainer.train()

        # Save model
        trainer.save_model("example2_model.pth")

        print(f"\n{results}")
    except FileNotFoundError:
        print(f"Data file not found: {data_path}")


def example_3_custom_model():
    """Example 3: Using custom model architecture."""
    print("\n" + "="*60)
    print("Example 3: Custom Model Architecture")
    print("="*60)

    data_path = "/path/to/your/data/BDS00001"  # Update this path

    # Create custom model with different architecture
    model = BalancePINN(
        expected_stiffness=350.0,  # Different initial guess
        expected_damping=4.0,
        hidden_sizes=[256, 256, 128, 64]  # Deeper network
    )

    config = TrainingConfig(epochs=8000, device='auto')

    try:
        trainer = BalancePINNTrainer(
            data_path=data_path,
            config=config,
            model=model
        )

        model, results = trainer.train()
        print(f"\n{results}")
    except FileNotFoundError:
        print(f"Data file not found: {data_path}")


def example_4_custom_training_phases():
    """Example 4: Using custom training phases."""
    print("\n" + "="*60)
    print("Example 4: Custom Training Phases")
    print("="*60)

    data_path = "/path/to/your/data/BDS00001"  # Update this path

    # Define custom training schedule
    # More aggressive physics weight increase
    custom_phases = [
        TrainingPhase(epochs=1000, w_data=1.0, w_physics=0.01, w_init=20.0),
        TrainingPhase(epochs=2000, w_data=1.0, w_physics=0.5, w_init=10.0),
        TrainingPhase(epochs=2000, w_data=1.0, w_physics=5.0, w_init=5.0),
        TrainingPhase(epochs=3000, w_data=1.0, w_physics=20.0, w_init=1.0),
    ]

    try:
        # Load data manually
        time_data, cop_data, subject_info = load_balance_data(data_path)

        # Create model
        model = BalancePINN()

        # Train with custom phases
        config = TrainingConfig(epochs=8000, device='auto')
        model, history = train_pinn(
            model,
            time_data,
            cop_data,
            subject_info,
            config,
            training_phases=custom_phases
        )

        # Evaluate
        results, cop_pred, cop_true = evaluate_model(
            model,
            time_data,
            cop_data,
            subject_info,
            config.device
        )

        # Plot
        plot_results(
            time_data,
            cop_true,
            cop_pred,
            history,
            results,
            save_path="example4_results.png"
        )

        print(f"\n{results}")
    except FileNotFoundError:
        print(f"Data file not found: {data_path}")


def example_5_synthetic_data():
    """Example 5: Training on synthetic data."""
    print("\n" + "="*60)
    print("Example 5: Synthetic Data Example")
    print("="*60)

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Generate synthetic balance data
    fs = 100  # Hz
    duration = 60  # seconds
    n_samples = fs * duration

    t = np.linspace(0, duration, n_samples)

    # Simulate damped oscillations with noise
    omega = 2 * np.pi * 0.5  # 0.5 Hz natural frequency
    damping = 0.1
    amplitude = 0.01  # 1 cm

    cop_x = amplitude * np.exp(-damping * t) * np.sin(omega * t)
    cop_y = amplitude * np.exp(-damping * t) * np.cos(omega * t * 0.8)

    # Add noise
    cop_x += np.random.normal(0, 0.001, n_samples)
    cop_y += np.random.normal(0, 0.001, n_samples)

    cop_data = np.column_stack([cop_x, cop_y])

    # Create subject info
    subject_info = SubjectInfo(
        weight=70.0,
        height=1.75,
        fs=fs,
        cop_x_mean=0.0,  # Already centered
        cop_y_mean=0.0
    )

    print(f"Generated {n_samples} samples of synthetic data")
    print(f"COP X range: [{cop_x.min()*100:.2f}, {cop_x.max()*100:.2f}] cm")
    print(f"COP Y range: [{cop_y.min()*100:.2f}, {cop_y.max()*100:.2f}] cm")

    # Create and train model
    model = BalancePINN(expected_stiffness=300.0, expected_damping=3.0)

    config = TrainingConfig(
        epochs=3000,  # Fewer epochs for synthetic data
        n_train_points=5000,
        device='auto'
    )

    print("\nTraining on synthetic data...")
    model, history = train_pinn(
        model,
        t,
        cop_data,
        subject_info,
        config
    )

    # Evaluate
    results, cop_pred, cop_true = evaluate_model(
        model,
        t,
        cop_data,
        subject_info,
        config.device
    )

    # Plot
    plot_results(
        t,
        cop_true,
        cop_pred,
        history,
        results,
        save_path="example5_synthetic_results.png"
    )

    print(f"\n{results}")
    print("\nNote: Parameters won't match real biomechanics since this is synthetic data")


def example_6_parameter_sensitivity():
    """Example 6: Testing parameter sensitivity."""
    print("\n" + "="*60)
    print("Example 6: Parameter Sensitivity Analysis")
    print("="*60)

    # Test different initial parameter guesses
    stiffness_values = [200.0, 300.0, 400.0]
    damping_values = [2.0, 3.0, 4.0]

    print("\nTesting different initial parameter guesses:")
    print("(Using synthetic data for demonstration)")

    # Generate simple synthetic data
    t = np.linspace(0, 10, 1000)
    cop_x = 0.01 * np.sin(2 * np.pi * 0.5 * t)
    cop_y = 0.01 * np.cos(2 * np.pi * 0.5 * t)
    cop_data = np.column_stack([cop_x, cop_y])

    subject_info = SubjectInfo(70.0, 1.75, 100, 0.0, 0.0)

    results_list = []

    for k in stiffness_values:
        for c in damping_values:
            print(f"\nInitial guess: K={k:.1f}, C={c:.1f}")

            model = BalancePINN(expected_stiffness=k, expected_damping=c)
            config = TrainingConfig(epochs=500, n_train_points=500, device='cpu')

            model, history = train_pinn(model, t, cop_data, subject_info, config)

            params = model.get_parameters_dict()
            print(f"  Final: Kx={params['stiffness_x']:.1f}, "
                  f"Dx={params['damping_x']:.2f}")

            results_list.append(params)

    print("\n" + "-"*60)
    print("Parameter Sensitivity Summary:")
    print(f"Stiffness X range: [{min(r['stiffness_x'] for r in results_list):.1f}, "
          f"{max(r['stiffness_x'] for r in results_list):.1f}]")
    print(f"Damping X range: [{min(r['damping_x'] for r in results_list):.2f}, "
          f"{max(r['damping_x'] for r in results_list):.2f}]")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("Balance PINN Examples")
    print("="*70)

    # Example 5 uses synthetic data and will always work
    example_5_synthetic_data()

    # Example 6 demonstrates parameter sensitivity
    example_6_parameter_sensitivity()

    # Examples 1-4 require actual data files
    print("\n" + "="*70)
    print("\nExamples 1-4 require actual balance data files.")
    print("To run them, update the data_path variables in the example functions")
    print("with the path to your WFDB format balance data.")
    print("\nUncomment the following lines to run them:")
    print("  # example_1_quick_start()")
    print("  # example_2_custom_config()")
    print("  # example_3_custom_model()")
    print("  # example_4_custom_training_phases()")
    print("="*70)


if __name__ == "__main__":
    main()
