"""
Class and methods to implement the Proper Orthogonal Decomposition - Radial 
Basis Function (POD-RBF) method.

Original Author: Kyle Beggs
"""
import os
import numpy as np
from tqdm import tqdm
import pickle


class pod_rbf:
    def __init__(self, energy_threshold=0.99):
        self.energy_threshold = energy_threshold

    def _calcTruncatedPODBasis(self):
        """
        Calculate the truncated POD basis.

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
            basis = U[:, :(trunc_id + 1)]
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
            self.cumul_energy = np.cumsum(self.eig_vals) / np.sum(
                self.eig_vals)
            if self.energy_threshold >= 1:
                self.truncated_cumul_energy = 1
                trunc_id = len(self.eig_vals)
            elif self.energy_threshold < self.cumul_energy[0]:
                trunc_id = 1
            else:
                trunc_id = np.argmax(self.cumul_energy > self.energy_threshold)

            self.truncated_energy = self.cumul_energy[trunc_id]
            self.eig_vals = self.eig_vals[:(trunc_id + 1)]
            self.eig_vecs = self.eig_vecs[:, :(trunc_id + 1)]

            # calculate the truncated POD basis
            basis = (self.snapshot @ self.eig_vecs) / np.sqrt(self.eig_vals)

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
            r2 += ((I - J) / self.params_range[i])**2
        return 1 / np.sqrt(r2 / (c**2) + 1)

    def _findOptimShapeParam(self, cond_range=[1e11, 1e12], max_steps=1e5):
            # find lower bound of c for bisection
            c_low = 0.011
            found_c_low = False
            C = self._buildCollocationMatrix(c_low)
            cond = np.linalg.cond(C)
            if cond <= cond_range[1]:
                found_c_low = True
            else:
                raise ValueError("Shape factor cannot be less than 0. shape_factor={}".format(c_low))

            # find upper bound of c for bisection
            c_high = 1
            found_c_high = False
            k = 0
            while found_c_high is False:
                k += 1
                C = self._buildCollocationMatrix(c_high)
                cond = np.linalg.cond(C)
                if cond < cond_range[0]:
                    c_high += 0.01
                else:
                    found_c_high = True
                if k > max_steps:
                    print("WARNING: MAX STEPS")
                    break

            # start bisection algorithm
            if found_c_low and found_c_high:
                found_optim_c = False
                k = 0
                while found_optim_c is False:
                    k += 1
                    optim_c = (c_low + c_high) / 2.0
                    C = self._buildCollocationMatrix(optim_c)
                    cond = np.linalg.cond(C)
                    if cond <= cond_range[0]:
                        c_low = optim_c
                    elif cond > cond_range[1]:
                        c_high = optim_c
                    else:
                        found_optim_c = True

                    if k > max_steps:
                        print("WARNING: MAX STEPS")
                        break
            else:
                raise ValueError("Could not find c {}".format(optim_c))

                        
            return optim_c

    def _buildRBFInferenceMatrix(self, inf_params):
        """Make the Radial Basis Function (RBF) matrix using the
         Hardy Inverse Multi-Qualdrics (IMQ) function

        Parameters
        ----------
        inf_params : ndarray
            The parameters to inference the RBF network on.

        Returns
        -------
        ndarray
            The RBF matrix.

        """

        inf_params = np.transpose(inf_params)
        assert inf_params.shape[0] == self.train_params.shape[0]

        num_params = self.train_params.shape[0]
        num_train_points = self.train_params.shape[1]
        r2 = np.zeros((num_params, num_train_points))
        for i in range(num_params):
            I, J = np.meshgrid(
                inf_params[i, :],
                self.train_params[i, :],
                indexing="ij",
                copy=False,
            )
            r2 += ((I - J) / self.params_range[i])**2
        return 1 / np.sqrt(r2 / (self.shape_factor**2) + 1)

    def train(self, snapshot, train_params, shape_factor=None):
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
        if train_params.ndim < 2:
            assert (
                snapshot.shape[1] == train_params.shape[0]
            ), "Number of parameter points ({}) and snapshots ({}) not the same".format(
                train_params.shape[1], snapshot.shape[1])
            self.params_range = np.array([np.ptp(train_params, axis=0)])
        else:
            assert (
                snapshot.shape[1] == train_params.shape[1]
            ), "Number of parameter points ({}) and snapshots ({}) not the same".format(
                train_params.shape[1], snapshot.shape[1])
            self.params_range = np.ptp(train_params, axis=1)
        self.snapshot = snapshot
        if train_params.ndim < 2:
            self.train_params = np.expand_dims(train_params, axis=0)
        else:
            self.train_params = train_params

        if shape_factor is None:
            self.shape_factor = self._findOptimShapeParam()
        else:
            self.shape_factor = shape_factor
        self.basis = self._calcTruncatedPODBasis()

        # build the Radial Basis Function (RBF) matrix
        F = self._buildCollocationMatrix(self.shape_factor)

        # calculate the amplitudes (A) and weights/coefficients (B)
        A = np.matmul(np.transpose(self.basis), self.snapshot)
        self.weights = A @ np.linalg.pinv(F.T)

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


def buildSnapshotMatrix(mypath_pattern,
                        skiprows=1,
                        usecols=(0),
                        split=1,
                        verbose=False):
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
    print(dirpath)
    files = [
        f for f in sorted(os.listdir(dirpath))
        if os.path.isfile(os.path.join(dirpath, f))
    ]
    num_files = len(files)
    assert (
        os.path.splitext(files[0])[1] == ".csv"
    ), "You have a file in this directory that is not a .csv named {}. All files in the directory must be .csv. Make sure to check for hidden files with a dot (.) prepended".format(
        files[0])
    num_sample_points = len(
        np.loadtxt(
            os.path.join(dirpath, files[0]),
            delimiter=",",
            skiprows=skiprows,
            usecols=usecols,
            unpack=True,
        ))

    snapshot = np.zeros((num_sample_points, num_files))
    i = 0
    for f in tqdm(files, desc="Loading snapshot .csv files"):
        assert os.path.splitext(
            f)[1] == ".csv", "File is not a .csv - {}".format(f)
        assert (
            os.path.splitext(f)[1] == ".csv"
        ), "You have a file in this directory that is not a .csv named {}. All files in the directory must be .csv. Make sure to check for hidden files with a dot (.) prepended".format(
            f)

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
            f)
        snapshot[:, i] = vals
        i += 1

    return snapshot


def save_model(filename, model):
    """Save the model. Uses Pythons pickle module.

    Parameters
    ----------
    filename_pattern : string
        The full path of the model to be saved.

    Returns
    -------
    None
    """
    file = open(filename, 'ab')
    pickle.dump(model, file)
    file.close()
    

def load_model(filename):
    """Load the model.

    Parameters
    ----------
    filename_pattern : string
        The full path of the file that has the model saved.

    Returns
    -------
    object
        The model object.
    """
    file = open(filename, 'rb')
    model = pickle.load(file)
    file.close()
    return model
