"""
File I/O utilities for POD-RBF.

Uses NumPy for file operations (not differentiable).
"""

import os
import pickle

import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from .types import ModelState


def build_snapshot_matrix(
    dirpath: str,
    skiprows: int = 1,
    usecols: int | tuple[int, ...] = 0,
    verbose: bool = True,
) -> np.ndarray:
    """
    Load snapshot matrix from CSV files in directory.

    Files are loaded in alphanumeric order. Ensure parameter array
    matches this ordering.

    Parameters
    ----------
    dirpath : str
        Directory containing CSV files.
    skiprows : int
        Number of header rows to skip in each file.
    usecols : int or tuple
        Column(s) to read from each file.
    verbose : bool
        Show progress bar.

    Returns
    -------
    np.ndarray
        Snapshot matrix, shape (n_samples, n_snapshots).
        Returns NumPy array - convert to JAX as needed.
    """
    files = sorted(
        [
            f
            for f in os.listdir(dirpath)
            if os.path.isfile(os.path.join(dirpath, f)) and f.endswith(".csv")
        ]
    )

    if not files:
        raise ValueError(f"No CSV files found in {dirpath}")

    # Get dimensions from first file
    first_data = np.loadtxt(
        os.path.join(dirpath, files[0]),
        delimiter=",",
        skiprows=skiprows,
        usecols=usecols,
    )
    n_samples = len(first_data) if first_data.ndim > 0 else 1
    n_snapshots = len(files)

    snapshot = np.zeros((n_samples, n_snapshots))

    iterator = tqdm(files, desc="Loading snapshots") if verbose else files
    for i, f in enumerate(iterator):
        data = np.loadtxt(
            os.path.join(dirpath, f),
            delimiter=",",
            skiprows=skiprows,
            usecols=usecols,
        )
        data_len = len(data) if data.ndim > 0 else 1
        assert data_len == n_samples, f"Inconsistent samples in {f}: got {data_len}, expected {n_samples}"
        snapshot[:, i] = data

    return snapshot


def save_model(filename: str, state: ModelState) -> None:
    """
    Save model state to file.

    Parameters
    ----------
    filename : str
        Output filename.
    state : ModelState
        Trained model state.
    """
    # Convert JAX arrays to NumPy for pickling
    state_dict = {
        "basis": np.asarray(state.basis),
        "weights": np.asarray(state.weights),
        "shape_factor": float(state.shape_factor),
        "train_params": np.asarray(state.train_params),
        "params_range": np.asarray(state.params_range),
        "truncated_energy": float(state.truncated_energy),
        "cumul_energy": np.asarray(state.cumul_energy),
        "poly_coeffs": np.asarray(state.poly_coeffs) if state.poly_coeffs is not None else None,
        "poly_degree": int(state.poly_degree),
    }
    with open(filename, "wb") as f:
        pickle.dump(state_dict, f)


def load_model(filename: str) -> ModelState:
    """
    Load model state from file.

    Parameters
    ----------
    filename : str
        Input filename.

    Returns
    -------
    ModelState
        Loaded model state with JAX arrays.
    """
    with open(filename, "rb") as f:
        state_dict = pickle.load(f)

    # Handle backward compatibility for models saved without poly fields
    poly_coeffs = state_dict.get("poly_coeffs")
    if poly_coeffs is not None:
        poly_coeffs = jnp.array(poly_coeffs)
    poly_degree = state_dict.get("poly_degree", 0)

    return ModelState(
        basis=jnp.array(state_dict["basis"]),
        weights=jnp.array(state_dict["weights"]),
        shape_factor=state_dict["shape_factor"],
        train_params=jnp.array(state_dict["train_params"]),
        params_range=jnp.array(state_dict["params_range"]),
        truncated_energy=state_dict["truncated_energy"],
        cumul_energy=jnp.array(state_dict["cumul_energy"]),
        poly_coeffs=poly_coeffs,
        poly_degree=poly_degree,
    )
