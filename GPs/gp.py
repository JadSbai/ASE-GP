import GPy
import numpy as np
import matplotlib.pyplot as plt
from GPy.models import GPRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, ConstantKernel, RBF
from sklearn.metrics import mean_squared_error
from skopt import gp_minimize
from skopt.space import Real


class GP:
    def __init__(self, simulation, descriptor="soap"):
        self.simulation = simulation
        self.descriptor = descriptor
        self.feature_size = 1
        self.train_positions, self.validate_positions = self.get_positions()
        self.train_energy, self.validate_energy = simulation.get_energies()
        params = self.optimise_hyperparameters()
        self.kernel = self.get_kernel(params=params)
        self.model = self.get_model()
        self.predictions = []

    def get_kernel(self, params=None):
        if self.descriptor == "soap":
            # params = [100, 0.12]
            return DotProduct(sigma_0=params[0]) + RBF(length_scale=params[1])
        elif self.descriptor == "distance":
            # params = [10, 10, 1]
            periodic_kernel = GPy.kern.PeriodicExponential(input_dim=1, active_dims=[0], variance=params[0])
            for dim in range(1, 20):
                periodic_kernel += GPy.kern.PeriodicExponential(input_dim=1, active_dims=[dim], variance=params[0])
            return GPy.kern.RBF(input_dim=self.feature_size, lengthscale=params[1], variance=params[2]) + periodic_kernel
        else:
            # params = [10]
            periodic_kernel = GPy.kern.PeriodicExponential(input_dim=1, active_dims=[0], variance=params[0])
            for dim in range(1, 20):
                periodic_kernel += GPy.kern.PeriodicExponential(input_dim=1, active_dims=[dim], variance=params[0])
            return periodic_kernel

    def get_positions(self):
        if self.descriptor == "soap":
            train_positions, validate_positions = self.simulation.calculate_soap_descriptors()
            train_positions = [soap.flatten() for soap in train_positions]
            validate_positions = [soap.flatten() for soap in validate_positions]
        elif self.descriptor == "distance":
            train_positions, validate_positions = self.simulation.calculate_distance_descriptors()
            train_target_size = max(len(array) for array in train_positions)
            validate_target_size = max(len(array) for array in validate_positions)
            self.feature_size = max(train_target_size, validate_target_size)
            for i in range(len(train_positions)):
                train_positions[i] = np.array(train_positions[i]).flatten()
                train_positions[i] = np.pad(train_positions[i],
                                                 (0, self.feature_size - len(train_positions[i])), 'constant')
            for i in range(len(validate_positions)):
                validate_positions[i] = np.array(validate_positions[i]).flatten()
                validate_positions[i] = np.pad(validate_positions[i],
                                                    (0, self.feature_size - len(validate_positions[i])), 'constant')
            train_positions = np.vstack(train_positions)
            validate_positions = np.vstack(validate_positions)
        else:
            train_positions = np.array([at.get_positions().flatten() for at in self.simulation.train_positions])
            validate_positions = np.array([at.get_positions().flatten() for at in self.simulation.validate_positions])

        return train_positions, validate_positions

    def fit_model_energy(self):
        self.model = self.get_model()
        if self.descriptor == "soap":
            self.model.fit(self.train_positions, self.train_energy)
            y_pred = self.model.predict(self.validate_positions)
            mse = mean_squared_error(self.validate_energy, y_pred)
        else:
            validate_energy = np.array(self.validate_energy).reshape(-1, 1)
            y_pred, _ = self.model.predict(self.validate_positions)
            mse = np.mean((validate_energy - y_pred) ** 2)
        self.predictions = y_pred
        print(f"MSE: {mse}")
        return mse

    def fit_model_force(self):
        pass

    def get_model(self):
        if self.descriptor == "soap":
            return GaussianProcessRegressor(kernel=self.kernel)
        else:
            train_energy = np.array(self.train_energy).reshape(-1, 1)
            return GPRegression(self.train_positions, train_energy, self.kernel)

    def objective_function(self, params):
        self.kernel = self.get_kernel(params)
        return self.fit_model_energy()

    def get_space(self):
        if self.descriptor == "soap":
            return [
                Real(1e-2, 1e2, name='dot_sigma'),
                Real(1e-10, 1, name='noise_level')
            ]
        elif self.descriptor == "distance":
            return [
                Real(1, 1e1, name='periodic_variance'),
                Real(1e-1, 1e1, name='rbf_length_scale'),
                Real(1, 1e1, name='rbf_variance'),
            ]
        else:
            return [
                Real(10, 1e2, name='periodic_variance')
            ]

    def optimise_hyperparameters(self):
        num_evaluations = 20
        space = self.get_space()
        result = gp_minimize(
            self.objective_function,  # the function to minimize
            space,  # the bounds on each dimension of x
            acq_func="EI",  # the acquisition function
            n_calls=num_evaluations,  # the number of evaluations of f
            n_random_starts=10,  # the number of random initialization points
            random_state=123  # the random seed
        )
        print("Optimal hyperparameters: ", result.x)
        return result.x

    def plot(self):
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(self.validate_energy)), self.validate_energy, 'o')
        plt.plot(range(len(self.predictions)), self.predictions, 'x')
        plt.title(f'Energy Prediction using {self.descriptor} descriptor')
        plt.legend(['True', 'Predicted'])
        plt.xlabel('Step')
        plt.ylabel('Energy (eV)')
        plt.show()
