import os
import numpy as np
from scipy.optimize import minimize_scalar


class pod_rbf:
    def __init__(self, energy_threshold=0.99):
        self.energy_threshold = energy_threshold

    def _calcTruncatedPODBasis(self):
        """Calculate the truncated POD basis.

        Parameters
        ----------
        snapshot : ndarray
            The matrix containing data points for each parameter as columns.
        energyThreshold : float
            The minimum percent of energy to keep in the system.

        Returns
        -------
        ndarray
            The truncated POD basis.

        """

        # set memory limit (in gigabytes) to switch to a more efficient algorithm
        self.mem_limit = 16  # gigabytes

        # calculate the memory in gigabytes
        memory = self.snapshot.nbytes / 1e9

        if memory < self.mem_limit:
            # calculate the SVD of the snapshot
            U, S, _ = np.linalg.svd(self.snapshot, full_matrices=False)

            # calculate the truncated POD basis based on the amount of energy/POD modes required
            self.cumul_energy = np.cumsum(S) / np.sum(S)
            if self.energy_threshold >= 1:
                self.truncated_energy = 1
                trunc_id = len(S)
                return U
            elif self.energy_threshold < self.cumul_energy[0]:
                trunc_id = 0
            else:
                trunc_id = np.argmax(self.cumul_energy > self.energy_threshold)

            self.truncated_energy = self.cumul_energy[trunc_id]
            basis = U[:, : (trunc_id + 1)]
        else:
            print("using eig!!!!")
            # compute the covarience matrix and corresponding eigenvalues and eigenvectors
            cov = np.matmul(np.transpose(self.snapshot), self.snapshot)
            self.eig_vals, self.eig_vecs = np.linalg.eigh(cov)
            self.eig_vals = np.abs(self.eig_vals.real)
            self.eig_vecs = self.eig_vecs.real
            # rearrange eigenvalues and eigenvectors from largest -> smallest
            self.eig_vals = self.eig_vals[::-1]
            self.eig_vecs = self.eig_vecs[:, ::-1]

            # calculate the truncated POD basis based on the amount of energy/POD modes required
            self.cumul_energy = np.cumsum(self.eig_vals) / np.sum(self.eig_vals)
            if self.energy_threshold >= 1:
                self.truncated_cumul_energy = 1
                trunc_id = len(self.eig_vals)
            elif self.energy_threshold < self.cumul_energy[0]:
                trunc_id = 1
            else:
                trunc_id = np.argmax(self.cumul_energy > self.energy_threshold)

            self.truncated_energy = self.cumul_energy[trunc_id]
            self.eig_vals = self.eig_vals[: (trunc_id + 1)]
            self.eig_vecs = self.eig_vecs[:, : (trunc_id + 1)]

            # calculate the truncated POD basis
            basis = np.matmul(self.snapshot, self.eig_vecs) / np.sqrt(self.eig_vals)

        return basis

    def _buildCollocationMatrix(self, c):
        num_train_points = self.train_params.shape[1]
        num_params = self.train_params.shape[0]
        r2 = np.zeros((num_train_points, num_train_points))
        for i in range(num_params):
            I, J = np.meshgrid(
                self.train_params[i, :],
                self.train_params[i, :],
                indexing="ij",
                copy=False,
            )
            r2 += (I - J) ** 2
        # return 1 / np.sqrt(r2 + c ** 2)
        return 1 / np.sqrt((r2 / c ** 2) + 1)

    def _findOptimShapeParam(
        self, cond_range=[1e10, 1e11], factor_start=1, max_steps=1e5
    ):
        optim_c = 1
        found_optim_c = False
        k = 0
        factor = factor_start
        diff = np.diff(np.sort(self.train_params, axis=1), axis=1)
        avgDist = np.sqrt(np.sum(np.mean(diff, axis=1) ** 2))
        while found_optim_c is False:
            k += 1
            optim_c = factor * avgDist
            if optim_c < 0:
                ValueError("Shape parameter is negative.")
            C = self._buildCollocationMatrix(optim_c)
            cond = np.linalg.cond(C)
            if cond <= cond_range[0]:
                factor += 0.1
            if cond > cond_range[1]:
                factor -= 0.1
            if cond > cond_range[0] and cond < cond_range[1]:
                found_optim_c = True
            if cond < 0.1:
                found_optim_c = True
            if k > max_steps:
                print("WARNING: MAX STEPS")
                break

        return optim_c

    def _buildRBFInferenceMatrix(self, inf_params):
        """Make the Radial Basis Function (RBF) matrix using the
         Hardy Inverse Multi-Qualdrics (IMQ) function

        Parameters
        ----------
        train_params : ndarray
            The parameters used to generate the snapshot matrix.
        inf_params : ndarray
            The parameters to inference the RBF network on.
        shape_factor : float
            The shape factor to be used in the RBF network.

        Returns
        -------
        ndarray
            The RBF matrix.

        """

        num_params = self.train_params.shape[0]
        num_train_points = self.train_params.shape[1]
        r2 = np.zeros((num_params, num_train_points))
        for i in range(num_params):
            I, J = np.meshgrid(
                inf_params[i, :], self.train_params[i, :], indexing="ij", copy=False
            )
            r2 += (I - J) ** 2

        # return 1 / np.sqrt(r2 + self.shape_factor ** 2)
        return 1 / np.sqrt((r2 / self.shape_factor ** 2) + 1)

    def train(self, snapshot, train_params):
        """
        Train the Radial Basis Function (RBF) network

        Parameters
        ----------
        snapshot : ndarray
            The matrix containing data points for each parameter as columns.
        basis : ndarray
            The truncated POD basis.
        shape_factor : float
            The shape factor to be used in the RBF network.
        train_params : ndarray
            The parameters used to generate the snapshot matrix.

        Returns
        -------
        ndarray
            The weights/coefficients of the RBF network.

        """
        self.snapshot = snapshot
        if train_params.ndim < 2:
            self.train_params = np.expand_dims(train_params, axis=0)
        else:
            self.train_params = train_params

        self.shape_factor = self._findOptimShapeParam()
        self.basis = self._calcTruncatedPODBasis()

        # build the Radial Basis Function (RBF) matrix
        F = self._buildCollocationMatrix(self.shape_factor)

        # calculate the amplitudes (A) and weights/coefficients (B)
        A = np.matmul(np.transpose(self.basis), self.snapshot)
        try:
            self.weights = np.matmul(A, np.linalg.pinv(np.transpose(F)))
        except:
            print("failed!!!!!!")
            quit()

    def inference(self, inf_params):
        """Inference the RBF network with an unseen parameter.

        Parameters
        ----------
        basis : ndarray
            The truncated POD basis.
        weights : ndarray
            The weights/coefficients of the RBF network.
        shape_factor : float
            The shape factor to be used in the RBF network.
        train_params : ndarray
            The parameters used to generate the snapshot matrix.
        inf_params : ndarray
            The parameters to inference the RBF network on.

        Returns
        -------
        ndarray
            The output of the RBF network according to the inf_params argument.

        """
        if np.isscalar(inf_params):
            inf_params = np.array([[inf_params]])
        if inf_params.ndim < 2:
            inf_params = np.expand_dims(inf_params, axis=0)

        # build the Radial Basis Function (RBF) matrix
        F = self._buildRBFInferenceMatrix(inf_params)

        # calculate the inferenced solution
        A = np.matmul(self.weights, np.transpose(F))
        inference = np.matmul(self.basis, A)

        return inference[:, 0]


def buildSnapshotMatrix(mypath_pattern, skiprows=1, usecols=(0), split=1):
    """Assemble the snapshot matrix.

    Parameters
    ----------
    filename_pattern : string
        The full path of the files to be loaded. e.g. data%03d.csv for data001.csv, data002.csv...

    Returns
    -------
    ndarray
        The snapshot matrix.
    """

    split_path = os.path.split(mypath_pattern)
    dirpath = split_path[0]
    files = [
        f
        for f in sorted(os.listdir(dirpath))
        if os.path.isfile(os.path.join(dirpath, f))
    ]
    num_files = len(files)
    assert os.path.splitext(files[0])[1] == ".csv", "File is not a .csv - {}".format(
        files[0]
    )
    num_sample_points = len(
        np.loadtxt(
            os.path.join(dirpath, files[0]),
            delimiter=",",
            skiprows=skiprows,
            usecols=usecols,
            unpack=True,
        )
    )

    snapshot = np.zeros((num_sample_points, num_files))
    i = 0
    for f in files:
        assert os.path.splitext(f)[1] == ".csv", "File is not a .csv - {}".format(f)
        vals = np.loadtxt(
            os.path.join(dirpath, f),
            delimiter=",",
            skiprows=skiprows,
            usecols=usecols,
            unpack=True,
        )
        assert (
            len(vals) == num_sample_points
        ), "Number of sample points in {} is not consistent with the other files.".format(
            f
        )
        snapshot[:, i] = vals
        i += 1

    return snapshot
