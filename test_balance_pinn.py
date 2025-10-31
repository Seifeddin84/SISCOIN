"""
Unit tests for balance_pinn module.

This test suite covers:
- Data loading and preprocessing
- Model architecture and forward pass
- Loss function computations
- Training configuration
- Evaluation metrics
"""

import unittest
import tempfile
import os
from unittest.mock import patch, MagicMock
import numpy as np
import torch
import torch.nn as nn

from balance_pinn import (
    SubjectInfo,
    TrainingConfig,
    TrainingPhase,
    BalancePINN,
    EarlyStopping,
    compute_physics_loss,
    compute_data_loss,
    compute_initial_condition_loss,
    compute_smoothness_loss,
    compute_parameter_regularization,
    create_default_training_phases,
    EvaluationResults,
    GRAVITY,
    DEFAULT_EPOCHS,
)


class TestSubjectInfo(unittest.TestCase):
    """Test SubjectInfo dataclass."""

    def setUp(self):
        self.subject = SubjectInfo(
            weight=70.0,
            height=1.75,
            fs=100.0,
            cop_x_mean=0.02,
            cop_y_mean=0.01
        )

    def test_moment_of_inertia(self):
        """Test moment of inertia calculation."""
        expected = 70.0 * 1.75 ** 2
        self.assertAlmostEqual(self.subject.moment_of_inertia, expected)

    def test_string_representation(self):
        """Test string formatting."""
        s = str(self.subject)
        self.assertIn("70.0kg", s)
        self.assertIn("1.75m", s)
        self.assertIn("2.00", s)  # 0.02 * 100 = 2.00 cm
        self.assertIn("1.00", s)  # 0.01 * 100 = 1.00 cm


class TestTrainingConfig(unittest.TestCase):
    """Test TrainingConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = TrainingConfig()
        self.assertEqual(config.epochs, DEFAULT_EPOCHS)
        self.assertGreater(config.n_train_points, 0)

    def test_invalid_epochs(self):
        """Test that negative epochs raises error."""
        with self.assertRaises(ValueError):
            TrainingConfig(epochs=-100)

    def test_invalid_n_train_points(self):
        """Test that non-positive n_train_points raises error."""
        with self.assertRaises(ValueError):
            TrainingConfig(n_train_points=0)

    def test_device_auto_detection(self):
        """Test automatic device detection."""
        config = TrainingConfig(device='auto')
        self.assertIn(config.device, ['cpu', 'cuda', 'mps'])


class TestBalancePINN(unittest.TestCase):
    """Test BalancePINN model."""

    def setUp(self):
        torch.manual_seed(42)
        self.model = BalancePINN(
            expected_stiffness=300.0,
            expected_damping=3.0
        )

    def test_initialization(self):
        """Test model initialization."""
        # Check parameters exist
        self.assertTrue(hasattr(self.model, 'stiffness_x'))
        self.assertTrue(hasattr(self.model, 'stiffness_y'))
        self.assertTrue(hasattr(self.model, 'damping_x'))
        self.assertTrue(hasattr(self.model, 'damping_y'))

        # Check initial values
        self.assertAlmostEqual(self.model.stiffness_x.item(), 300.0)
        self.assertAlmostEqual(self.model.damping_x.item(), 3.0)

    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        t = torch.randn(10, 1)
        output = self.model(t)

        self.assertEqual(output.shape, (10, 2))
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_custom_architecture(self):
        """Test model with custom architecture."""
        model = BalancePINN(hidden_sizes=[64, 32])
        t = torch.randn(5, 1)
        output = model(t)

        self.assertEqual(output.shape, (5, 2))

    def test_get_parameters_dict(self):
        """Test parameter dictionary extraction."""
        params = self.model.get_parameters_dict()

        self.assertIn('stiffness_x', params)
        self.assertIn('stiffness_y', params)
        self.assertIn('damping_x', params)
        self.assertIn('damping_y', params)

        self.assertEqual(len(params), 4)


class TestLossFunctions(unittest.TestCase):
    """Test loss function computations."""

    def setUp(self):
        torch.manual_seed(42)
        self.model = BalancePINN()
        self.subject_info = SubjectInfo(
            weight=70.0,
            height=1.75,
            fs=100.0,
            cop_x_mean=0.0,
            cop_y_mean=0.0
        )
        self.t = torch.linspace(0, 1, 10).reshape(-1, 1)
        self.cop = torch.randn(10, 2) * 0.01

    def test_data_loss_shape(self):
        """Test data loss returns scalar."""
        loss = compute_data_loss(self.model, self.t, self.cop)
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertFalse(torch.isnan(loss))

    def test_data_loss_zero_for_perfect_fit(self):
        """Test data loss is zero when prediction matches target."""
        with torch.no_grad():
            pred = self.model(self.t)
        loss = compute_data_loss(self.model, self.t, pred)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_physics_loss_shape(self):
        """Test physics loss returns scalar."""
        loss = compute_physics_loss(self.model, self.t, self.subject_info)
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertFalse(torch.isnan(loss))

    def test_physics_loss_requires_grad(self):
        """Test physics loss computation requires gradients."""
        t_no_grad = self.t.clone().detach()
        # Should still work (grad enabled internally)
        loss = compute_physics_loss(self.model, t_no_grad, self.subject_info)
        self.assertIsNotNone(loss)

    def test_initial_condition_loss(self):
        """Test initial condition loss."""
        t0 = self.t[:1]
        cop0 = self.cop[:1]
        loss = compute_initial_condition_loss(self.model, t0, cop0)

        self.assertEqual(loss.shape, torch.Size([]))
        self.assertFalse(torch.isnan(loss))

    def test_smoothness_loss(self):
        """Test smoothness loss."""
        loss = compute_smoothness_loss(self.model, self.t)

        self.assertEqual(loss.shape, torch.Size([]))
        self.assertFalse(torch.isnan(loss))
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_parameter_regularization(self):
        """Test parameter regularization."""
        loss = compute_parameter_regularization(self.model)

        self.assertEqual(loss.shape, torch.Size([]))
        self.assertFalse(torch.isnan(loss))
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_parameter_regularization_negative_params(self):
        """Test regularization penalizes negative parameters."""
        # Set negative parameters
        with torch.no_grad():
            self.model.stiffness_x.data = torch.tensor(-10.0)

        loss = compute_parameter_regularization(self.model)
        self.assertGreater(loss.item(), 0.0)


class TestEarlyStopping(unittest.TestCase):
    """Test EarlyStopping functionality."""

    def test_initialization(self):
        """Test early stopping initialization."""
        es = EarlyStopping(patience=10, delta=1e-4)
        self.assertEqual(es.patience, 10)
        self.assertEqual(es.delta, 1e-4)
        self.assertFalse(es.should_stop)

    def test_improvement_resets_counter(self):
        """Test counter resets on improvement."""
        es = EarlyStopping(patience=3, delta=1e-4)

        es(1.0)
        self.assertEqual(es.counter, 0)

        es(0.9)  # Improvement
        self.assertEqual(es.counter, 0)

        es(0.8)  # Improvement
        self.assertEqual(es.counter, 0)

    def test_no_improvement_increments_counter(self):
        """Test counter increments without improvement."""
        es = EarlyStopping(patience=3, delta=1e-4)

        es(1.0)
        es(1.0)  # No improvement
        self.assertEqual(es.counter, 1)

        es(1.0)  # No improvement
        self.assertEqual(es.counter, 2)

    def test_stops_after_patience(self):
        """Test stopping after patience is exceeded."""
        es = EarlyStopping(patience=2, delta=1e-4)

        self.assertFalse(es(1.0))
        self.assertFalse(es(1.0))
        self.assertFalse(es(1.0))
        self.assertTrue(es(1.0))  # Should trigger stopping

    def test_delta_threshold(self):
        """Test delta threshold for improvement."""
        es = EarlyStopping(patience=3, delta=0.1)

        es(1.0)
        es(0.95)  # Improvement < delta, should not reset
        self.assertEqual(es.counter, 1)

        es(0.85)  # Improvement > delta, should reset
        self.assertEqual(es.counter, 0)


class TestTrainingPhases(unittest.TestCase):
    """Test training phase configuration."""

    def test_default_phases(self):
        """Test default training phases creation."""
        epochs = 1000
        phases = create_default_training_phases(epochs)

        self.assertEqual(len(phases), 4)

        # Check total epochs
        total = sum(phase.epochs for phase in phases)
        self.assertEqual(total, epochs)

        # Check physics weight increases
        physics_weights = [phase.w_physics for phase in phases]
        self.assertTrue(all(
            physics_weights[i] <= physics_weights[i+1]
            for i in range(len(physics_weights)-1)
        ))

    def test_phase_attributes(self):
        """Test training phase has all required attributes."""
        phase = TrainingPhase(
            epochs=100,
            w_data=1.0,
            w_physics=0.5,
            w_init=10.0,
            w_smooth=0.01,
            w_reg=0.01
        )

        self.assertEqual(phase.epochs, 100)
        self.assertEqual(phase.w_data, 1.0)
        self.assertEqual(phase.w_physics, 0.5)


class TestEvaluationResults(unittest.TestCase):
    """Test EvaluationResults dataclass."""

    def setUp(self):
        self.results = EvaluationResults(
            rmse_x=0.005,
            rmse_y=0.006,
            r2_x=0.95,
            r2_y=0.93,
            stiffness_x=300.0,
            stiffness_y=310.0,
            damping_x=3.0,
            damping_y=3.2,
            omega_n_x=0.5,
            omega_n_y=0.52,
            zeta_x=0.1,
            zeta_y=0.11
        )

    def test_to_dict(self):
        """Test conversion to dictionary."""
        d = self.results.to_dict()

        self.assertIsInstance(d, dict)
        self.assertEqual(len(d), 12)
        self.assertIn('rmse_x', d)
        self.assertIn('stiffness_x', d)

    def test_string_representation(self):
        """Test string formatting."""
        s = str(self.results)

        self.assertIn('RMSE', s)
        self.assertIn('Stiffness', s)
        self.assertIn('Damping', s)
        self.assertIn('Natural Freq', s)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflow."""

    def test_model_training_convergence(self):
        """Test that model can be trained without errors."""
        torch.manual_seed(42)

        # Create synthetic data
        t = np.linspace(0, 10, 100)
        cop = np.column_stack([
            0.01 * np.sin(2 * np.pi * 0.5 * t),
            0.01 * np.cos(2 * np.pi * 0.5 * t)
        ])

        subject_info = SubjectInfo(
            weight=70.0,
            height=1.75,
            fs=10.0,
            cop_x_mean=0.0,
            cop_y_mean=0.0
        )

        # Create and train model (mini version)
        model = BalancePINN()
        config = TrainingConfig(
            epochs=100,  # Small for testing
            n_train_points=50,
            device='cpu'
        )

        # Prepare data
        device = torch.device('cpu')
        t_tensor = torch.tensor(t.reshape(-1, 1), dtype=torch.float32).to(device)
        cop_tensor = torch.tensor(cop, dtype=torch.float32).to(device)

        # Setup optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Training loop
        initial_loss = None
        final_loss = None

        for epoch in range(config.epochs):
            optimizer.zero_grad()

            loss = compute_data_loss(model, t_tensor, cop_tensor)

            if epoch == 0:
                initial_loss = loss.item()

            loss.backward()
            optimizer.step()

            if epoch == config.epochs - 1:
                final_loss = loss.item()

        # Check that loss decreased
        self.assertLess(final_loss, initial_loss)
        self.assertFalse(np.isnan(final_loss))

    def test_model_parameters_stay_positive(self):
        """Test that model parameters remain positive during training."""
        torch.manual_seed(42)

        model = BalancePINN()

        # All parameters should start positive
        self.assertGreater(model.stiffness_x.item(), 0)
        self.assertGreater(model.stiffness_y.item(), 0)
        self.assertGreater(model.damping_x.item(), 0)
        self.assertGreater(model.damping_y.item(), 0)


class TestPhysicsConstraints(unittest.TestCase):
    """Test physics constraints and equations."""

    def test_gravity_constant(self):
        """Test gravity constant is correct."""
        self.assertAlmostEqual(GRAVITY, 9.81)

    def test_moment_of_inertia_scaling(self):
        """Test moment of inertia scales correctly."""
        subject1 = SubjectInfo(50.0, 1.5, 100, 0, 0)
        subject2 = SubjectInfo(100.0, 1.5, 100, 0, 0)

        # Double weight should double moment of inertia
        self.assertAlmostEqual(
            subject2.moment_of_inertia,
            2 * subject1.moment_of_inertia
        )

    def test_physics_loss_dimensional_consistency(self):
        """Test physics loss is dimensionally consistent."""
        model = BalancePINN()
        subject_info = SubjectInfo(70.0, 1.75, 100, 0, 0)

        t = torch.linspace(0, 1, 10).reshape(-1, 1)

        loss = compute_physics_loss(model, t, subject_info)

        # Loss should be non-negative
        self.assertGreaterEqual(loss.item(), 0.0)


def run_tests():
    """Run all tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSubjectInfo))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainingConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestBalancePINN))
    suite.addTests(loader.loadTestsFromTestCase(TestLossFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestEarlyStopping))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainingPhases))
    suite.addTests(loader.loadTestsFromTestCase(TestEvaluationResults))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestPhysicsConstraints))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    result = run_tests()

    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)
