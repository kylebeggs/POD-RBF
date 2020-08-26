import os
import numpy as np
from scipy.optimize import minimize_scalar


def mkSnapshotMatrix(mypath_pattern, skiprows=1, usecols=(0), split=1):
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
    files = [f for f in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, f))]
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
            The percent of energy to keep in the system.

        Returns
        -------
        ndarray
            The truncated POD basis.

        """

        # compute the covarience matrix and corresponding eigenvalues and eigenvectors
        cov = np.matmul(np.transpose(self.snapshot), self.snapshot)
        self.eig_vals, self.eig_vecs = np.linalg.eigh(cov)
        self.eig_vals = np.abs(self.eig_vals.real)
        self.eig_vecs = self.eig_vecs.real
        # rearrange eigenvalues and eigenvectors from largest -> smallest
        self.eig_vals = self.eig_vals[::-1]
        self.eig_vecs = self.eig_vecs[:, ::-1]

        # compute the energy in the system
        self.energy = self.eig_vals / np.sum(self.eig_vals)

        # truncate the eigenvalues and eigenvectors
        total_energy = 0
        self.trunc_id = None
        for i in range(len(self.energy)):
            total_energy += self.energy[i]
            if total_energy > self.energy_threshold:
                self.trunc_id = i + 1
                break
        if self.trunc_id is None:
            raise AttributeError(
                "trunc_id is None. energy_threshold is most likely set as a value higher then the energy contained in the first POD mode."
            )
        else:
            self.energy = self.energy[: self.trunc_id]
            self.eig_vals = self.eig_vals[: self.trunc_id]
            self.eig_vecs = self.eig_vecs[:, : self.trunc_id]

        # calculate the truncated POD basis
        basis = np.matmul(self.snapshot, self.eig_vecs)
        for i in range(len(self.eig_vals)):
            basis[:, i] = basis[:, i] / np.sqrt(self.eig_vals[i])

        return basis

    def _objectiveFuncShapeParam(self, shape_factor, sum, target_cond_num):
        return (
            np.linalg.cond(1 / np.sqrt(sum + shape_factor ** 2)) - target_cond_num
        ) ** 2

    def _findOptimShapeParam(self, target_cond_num=1e6):
        sum = np.zeros((len(self.train_params[0, :]), len(self.train_params[0, :])))
        for i in range(self.train_params.shape[0]):
            I, J = np.meshgrid(
                self.train_params[i, :],
                self.train_params[i, :],
                indexing="ij",
                copy=False,
            )
            sum += np.abs(I - J)
        res = minimize_scalar(
            self._objectiveFuncShapeParam,
            method="bounded",
            bounds=(1e-4, 1e4),
            args=(sum, target_cond_num),
        )
        return res.x

    def _mkRBFTrainMatrix(self):
        """Make the Radial Basis Function (RBF) matrix using the
         Hardy Inverse Multi-Qualdrics (IMQ) function

        Parameters
        ----------
        train_params : ndarray
            The parameters used to generate the snapshot matrix.
        shape_factor : float
            The shape factor to be used in the RBF network.

        Returns
        -------
        ndarray
            The RBF matrix.

        """

        sum = np.zeros((len(self.train_params[0, :]), len(self.train_params[0, :])))
        for i in range(self.train_params.shape[0]):
            I, J = np.meshgrid(
                self.train_params[i, :],
                self.train_params[i, :],
                indexing="ij",
                copy=False,
            )
            sum += np.abs(I - J)

        return 1 / np.sqrt(sum + self.shape_factor ** 2)

    def _mkRBFInferenceMatrix(self, inf_params):
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

        sum = np.zeros((1, len(self.train_params[0, :])))
        for i in range(self.train_params.shape[0]):
            I, J = np.meshgrid(
                inf_params[i], self.train_params[i, :], indexing="ij", copy=False
            )
            sum += np.abs(I - J)

        return 1 / np.sqrt(sum + self.shape_factor ** 2)

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
        F = self._mkRBFTrainMatrix()

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
        F = self._mkRBFInferenceMatrix(inf_params)

        # calculate the inferenced solution
        A = np.matmul(self.weights, np.transpose(F))
        inference = np.matmul(self.basis, A)

        return inference[:, 0]
