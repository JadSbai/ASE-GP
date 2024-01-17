import GPy
import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, WhiteKernel
from sklearn.metrics import mean_squared_error
from skopt import gp_minimize
from skopt.space import Real


class ForceGP:
    def __init__(self, simulation, descriptor="soap"):
        self.simulation = simulation
        self.descriptor = descriptor
        self.train_positions, self.validate_positions = self.get_positions()
        train_values, validate_values = self.simulation.get_forces()
        self.train_values = np.array(train_values)
        self.validate_values = np.array(validate_values)
        self.predictions = []
        self.variance = []
        self.kernel = None
        # self.params = self.optimise_hyperparameters()


    def get_positions(self):
        if self.descriptor == "soap":
            train_positions, validate_positions = self.simulation.calculate_soap_descriptors()
            train_positions = np.array(train_positions)
            validate_positions = np.array(validate_positions)
        else:
            train_positions = np.array([at.get_positions() for at in self.simulation.train_positions])
            validate_positions = np.array([at.get_positions() for at in self.simulation.validate_positions])
        return train_positions, validate_positions

    def get_kernel(self, params=None, dimension=0):
        if self.descriptor == "soap":
            if dimension == 0 or dimension == 1:
                return GPy.kern.Linear(input_dim=2101, variances=[params[0]]) + GPy.kern.White(input_dim=2101, variance=params[1])
            else:
                return GPy.kern.Matern32(input_dim=2101, variance=params[0], lengthscale=params[1])
        else:
            # params = [1, 1, 0.5]
            return GPy.kern.Matern32(input_dim=2101, variance=params[0], lengthscale=params[1]) + GPy.kern.White(
                input_dim=2101, variance=params[2])

    def fit_dimension(self, dimension):
        train_positions = self.train_positions[:, 0, :]
        train_forces = self.train_values[:, 0, dimension].reshape(-1, 1)

        if self.descriptor == "soap":
            model = GPy.models.GPRegression(train_positions, train_forces, self.kernel)
            model.optimize()

        else:
            model = GaussianProcessRegressor(kernel=self.kernel)
            model.fit(train_positions, train_forces)
        return model

    def predict_dimension(self, dimension):
        model = self.fit_dimension(dimension)
        mse = 0
        n_atoms = len(self.validate_values[0])
        predictions = np.zeros((len(self.validate_values), len(self.validate_values[0])))
        for n in range(n_atoms):
            y_pred, Y_var = model.predict(self.validate_positions[:, :, dimension][:, n].reshape(-1, 1))
            mse += mean_squared_error(self.validate_values[:, :, dimension][:, n].reshape(-1, 1), y_pred)
            predictions[:, n] = y_pred.flatten()
        return predictions, mse

    def predict_soap_dimension(self, dimension):
        model = self.fit_dimension(dimension)
        mse = 0
        n_atoms = len(self.validate_values[0])
        predictions = np.zeros((len(self.validate_values), len(self.validate_values[0])))
        std_devs = np.zeros((len(self.validate_values), len(self.validate_values[0])))
        for n in range(n_atoms):
            y_pred, Y_var = model.predict(self.validate_positions[:, n, :])
            mse += mean_squared_error(self.validate_values[:, n, dimension].reshape(-1, 1), y_pred)
            predictions[:, n] = y_pred.flatten()
            std_devs[:, n] = np.sqrt(Y_var.flatten())
        return predictions, mse, std_devs

    def fit_model(self):
        params = [40, 0.9]
        self.kernel = self.get_kernel(params=params, dimension=0)
        x_predictions, x_mse, std_devs_x = self.predict_soap_dimension(0)
        print(f"x_MSE: {x_mse}")
        params = [41, 0.6]
        self.kernel = self.get_kernel(params=params, dimension=1)
        y_predictions, y_mse, std_devs_y = self.predict_soap_dimension(1)
        print(f"y_MSE: {y_mse}")
        params = [12, 0.4]
        self.kernel = self.get_kernel(params=params, dimension=2)
        z_predictions, z_mse, std_devs_z = self.predict_soap_dimension(2)
        print(f"z_MSE: {z_mse}")

        self.predictions = np.array([x_predictions, y_predictions, z_predictions])
        self.variance = np.array([std_devs_x, std_devs_y, std_devs_z])

    def objective_function(self, params):
        self.kernel = self.get_kernel(params=params)
        _, mse, std = self.predict_soap_dimension(1)
        print(f"MSE: {mse}")
        return mse

    def get_space(self):
        if self.descriptor == "soap":
            return [
                Real(1e-5, 1e3, name='dot_sigma'),
                Real(1e-5, 1e3, name='noise_level')
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
        x_pred_aggregated = np.max(self.predictions[0], axis=1)
        y_pred_aggregated = np.max(self.predictions[1], axis=1)
        z_pred_aggregated = np.max(self.predictions[2], axis=1)

        x_true_aggregated = np.max(self.validate_values[:, :, 0], axis=1)
        y_true_aggregated = np.max(self.validate_values[:, :, 1], axis=1)
        z_true_aggregated = np.max(self.validate_values[:, :, 2], axis=1)

        x_std_aggregated = np.min(self.variance[0], axis=1)
        y_std_aggregated = np.min(self.variance[1], axis=1)
        z_std_aggregated = np.min(self.variance[2], axis=1)

        time_steps = np.arange(len(x_pred_aggregated))

        # Plotting
        plt.figure(figsize=(15, 5))

        # X Component
        plt.subplot(1, 3, 1)
        plt.plot(time_steps, x_true_aggregated, label='True X Force', marker='o')
        plt.plot(time_steps, x_pred_aggregated, label='Predicted X Force', marker='x')
        plt.fill_between(time_steps, x_pred_aggregated - x_std_aggregated, x_pred_aggregated + x_std_aggregated,
                         alpha=0.2)
        plt.title('Cumulative X Force Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Force')
        plt.legend()

        # Y Component
        plt.subplot(1, 3, 2)
        plt.plot(time_steps, y_true_aggregated, label='True Y Force', marker='o')
        plt.plot(time_steps, y_pred_aggregated, label='Predicted Y Force', marker='x')
        plt.fill_between(time_steps, y_pred_aggregated - y_std_aggregated, y_pred_aggregated + y_std_aggregated,
                         alpha=0.2)
        plt.title('Cumulative Y Force Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Force')
        plt.legend()

        # Z Component
        plt.subplot(1, 3, 3)
        plt.plot(time_steps, z_true_aggregated, label='True Z Force', marker='o')
        plt.plot(time_steps, z_pred_aggregated, label='Predicted Z Force', marker='x')
        plt.fill_between(time_steps, z_pred_aggregated - z_std_aggregated, z_pred_aggregated + z_std_aggregated,
                         alpha=0.2)
        plt.title('Cumulative Z Force Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Force')
        plt.legend()

        plt.tight_layout()
        plt.show()
