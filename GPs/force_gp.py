import GPy
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error


class ForceGP:
    def __init__(self, simulation, descriptor="soap"):
        self.simulation = simulation
        self.descriptor = descriptor
        self.train_positions, self.validate_positions = self.get_positions()
        train_values, validate_values = self.simulation.get_forces()
        self.train_values = np.array(train_values)
        self.validate_values = np.array(validate_values)
        self.predictions = []

    def get_positions(self):
        train_positions = np.array([at.get_positions() for at in self.simulation.train_positions])
        validate_positions = np.array([at.get_positions() for at in self.simulation.validate_positions])

        return train_positions, validate_positions

    def get_kernel(self, dimension, params=None):
        if dimension == 0:
            return GPy.kern.sde_Matern32(input_dim=1, ARD=True)
        elif dimension == 1:
            return GPy.kern.sde_Matern32(input_dim=1, ARD=True) + GPy.kern.White(input_dim=1)
        elif dimension == 2:
            return GPy.kern.sde_Matern32(input_dim=1, ARD=True) + GPy.kern.White(input_dim=1)

    def fit_dimension(self, dimension):
        train_positions = self.train_positions[:, :, dimension][:, 0].reshape(-1, 1)
        train_forces = self.train_values[:, :, dimension][:, 0].reshape(-1, 1)
        kernel = self.get_kernel(dimension)

        model = GPy.models.GPRegression(train_positions, train_forces, kernel)
        model.optimize()
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

    def fit_model(self):
        x_predictions, x_mse = self.predict_dimension(0)
        y_predictions, y_mse = self.predict_dimension(1)
        z_predictions, z_mse = self.predict_dimension(2)

        self.predictions = np.array([x_predictions, y_predictions, z_predictions])

    def plot(self):

        x_pred_aggregated = np.sum(self.predictions[0], axis=1)
        y_pred_aggregated = np.sum(self.predictions[1], axis=1)
        z_pred_aggregated = np.sum(self.predictions[2], axis=1)

        x_true_aggregated = np.sum(self.validate_values[:, :, 0], axis=1)
        y_true_aggregated = np.sum(self.validate_values[:, :, 1], axis=1)
        z_true_aggregated = np.sum(self.validate_values[:, :, 2], axis=1)

        # Plotting
        plt.figure(figsize=(15, 5))

        # X Component
        plt.subplot(1, 3, 1)
        plt.plot(x_true_aggregated, label='True X Force', marker='o')
        plt.plot(x_pred_aggregated, label='Predicted X Force', marker='x')
        plt.title('Cumulative X Force Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Force')
        plt.legend()

        # Y Component
        plt.subplot(1, 3, 2)
        plt.plot(y_true_aggregated, label='True Y Force', marker='o')
        plt.plot(y_pred_aggregated, label='Predicted Y Force', marker='x')
        plt.title('Cumulative Y Force Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Force')
        plt.legend()

        # Z Component
        plt.subplot(1, 3, 3)
        plt.plot(z_true_aggregated, label='True Z Force', marker='o')
        plt.plot(z_pred_aggregated, label='Predicted Z Force', marker='x')
        plt.title('Cumulative Z Force Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Force')
        plt.legend()

        plt.tight_layout()
        plt.show()

