import numpy as np
from scipy.optimize import minimize_scalar


class pod_rbf:
    def __init__(self, energy_threshold=0.2):
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
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        eig_vals = np.abs(eig_vals.real)
        eig_vecs = eig_vecs.real

        # compute the energy in the system
        energy = eig_vals / np.sum(eig_vals)

        # truncate the eigenvalues and eigenvectors
        totalEnergy = 0
        for i in range(len(energy)):
            totalEnergy += energy[i]
            if totalEnergy > self.energy_threshold:
                truncId = i + 1
                break
        if truncId is not None:
            eig_vals = eig_vals[:truncId]
            energy = energy[:truncId]
        eig_vecs = eig_vecs[:, :truncId]

        # calculate the truncated POD basis
        basis = np.matmul(self.snapshot, eig_vecs)
        for i in range(len(eig_vals)):
            basis[:, i] = basis[:, i] / np.sqrt(eig_vals[i])

        return basis

    def _objectiveFuncShapeParam(self, shape_factor, sum, target_cond_num):
        return (
            np.linalg.cond(1 / np.sqrt(sum + shape_factor ** 2)) - target_cond_num
        ) ** 2

    def _findOptimShapeParam(self, target_cond_num=1e12):
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
            bounds=(1e-2, 1000),
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
            The output of the RBF netowkr according to the inf_params argument.

        """

        # build the Radial Basis Function (RBF) matrix
        F = self._mkRBFInferenceMatrix(inf_params)

        # calculate the inferenced solution
        A = np.matmul(self.weights, np.transpose(F))
        inference = np.matmul(self.basis, A)

        return inference
