"""Tests for I/O utilities."""

import os
import tempfile

import jax.numpy as jnp
import numpy as np
import pytest

from pod_rbf.io import build_snapshot_matrix, load_model, save_model
from pod_rbf.types import ModelState


class TestSaveLoadModel:
    """Test model serialization."""

    @pytest.fixture
    def sample_state(self):
        """Create sample model state."""
        return ModelState(
            basis=jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            weights=jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            shape_factor=0.5,
            train_params=jnp.array([[1.0, 2.0, 3.0]]),
            params_range=jnp.array([2.0]),
            truncated_energy=0.99,
            cumul_energy=jnp.array([0.9, 0.99]),
            poly_coeffs=jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            poly_degree=2,
        )

    def test_save_load_roundtrip(self, sample_state):
        """Saved and loaded state should match."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            filename = f.name

        try:
            save_model(filename, sample_state)
            loaded = load_model(filename)

            assert jnp.allclose(loaded.basis, sample_state.basis)
            assert jnp.allclose(loaded.weights, sample_state.weights)
            assert loaded.shape_factor == sample_state.shape_factor
            assert jnp.allclose(loaded.train_params, sample_state.train_params)
            assert jnp.allclose(loaded.params_range, sample_state.params_range)
            assert loaded.truncated_energy == sample_state.truncated_energy
            assert jnp.allclose(loaded.cumul_energy, sample_state.cumul_energy)
            assert jnp.allclose(loaded.poly_coeffs, sample_state.poly_coeffs)
            assert loaded.poly_degree == sample_state.poly_degree
        finally:
            os.unlink(filename)

    def test_loaded_state_is_model_state(self, sample_state):
        """Loaded state should be ModelState instance."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            filename = f.name

        try:
            save_model(filename, sample_state)
            loaded = load_model(filename)

            assert isinstance(loaded, ModelState)
        finally:
            os.unlink(filename)


class TestBuildSnapshotMatrix:
    """Test snapshot matrix loading from CSV files."""

    @pytest.fixture
    def csv_dir(self):
        """Create temporary directory with CSV files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 3 CSV files with 5 data points each
            for i in range(3):
                data = np.arange(5) * (i + 1)  # [0,1,2,3,4] * (i+1)
                filepath = os.path.join(tmpdir, f"data_{i:03d}.csv")
                np.savetxt(filepath, data[:, None], delimiter=",", header="value", comments="")

            yield tmpdir

    def test_loads_all_files(self, csv_dir):
        """Should load all CSV files in directory."""
        snapshot = build_snapshot_matrix(csv_dir, verbose=False)

        assert snapshot.shape == (5, 3), f"Expected (5, 3), got {snapshot.shape}"

    def test_correct_values(self, csv_dir):
        """Should load correct values."""
        snapshot = build_snapshot_matrix(csv_dir, verbose=False)

        # File data_000.csv has [0,1,2,3,4]
        assert np.allclose(snapshot[:, 0], [0, 1, 2, 3, 4])
        # File data_001.csv has [0,2,4,6,8]
        assert np.allclose(snapshot[:, 1], [0, 2, 4, 6, 8])
        # File data_002.csv has [0,3,6,9,12]
        assert np.allclose(snapshot[:, 2], [0, 3, 6, 9, 12])

    def test_alphanumeric_order(self, csv_dir):
        """Files should be loaded in alphanumeric order."""
        snapshot = build_snapshot_matrix(csv_dir, verbose=False)

        # First file should be data_000.csv
        # Last file should be data_002.csv
        assert snapshot[1, 0] == 1  # data_000: index 1 = 1
        assert snapshot[1, 2] == 3  # data_002: index 1 = 3

    def test_empty_dir_raises(self):
        """Should raise ValueError for empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No CSV files"):
                build_snapshot_matrix(tmpdir, verbose=False)

    def test_inconsistent_samples_raises(self):
        """Should raise AssertionError for inconsistent sample counts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files with different row counts
            np.savetxt(os.path.join(tmpdir, "a.csv"), [1, 2, 3], delimiter=",", header="x", comments="")
            np.savetxt(os.path.join(tmpdir, "b.csv"), [1, 2], delimiter=",", header="x", comments="")

            with pytest.raises(AssertionError, match="Inconsistent"):
                build_snapshot_matrix(tmpdir, verbose=False)
