from base_model import Model # Import base Model class
from random import random # Import random function
class ARIMA(Model): # General model class
    def __init__(self):
        self.const, self.ar_coef, self.diff, self.ma_coef = 0, [0], 0, [0] # Initialize coefficients
    def train(self, x, ar_order, diff_order, ma_order, iterationLimit = 1000): # Train parameters of model to given data with given orders
        def get_residual(x, predictions, i):
            return x[i] - self.const - sum([self.ar_coef[j] * (predictions[i - j] if i - j >= 0 else x[i - j]) for j in range(1, len(self.ar_coef))]) - sum([self.ma_coef[j] * (predictions[i - j] - x[i - j]) for j in range(1, len(self.ma_coef)) if 0 <= i - j < len(x)])
        self.ar_coef = [random()] * (ar_order + 1) # Initialize random AR coefficients
        self.diff = diff_order
        self.ma_coef = [random()] * (ma_order + 1) # Initialize random MA coefficients
        x_diff, _ = self.difference(x, diff_order) # Apply differencing
        for _ in range(iterationLimit):
            predictions = self.predict(x, forecasts = 0, forecasts_only = False) # Get predictions for current parameters
            for i in range(len(self.ar_coef)):
                self.ar_coef[i] -= -2 * sum([((predictions[j] - x[j]) * x_diff[j]) for j in range(len(x))])
                self.const -= -2 * sum([(predictions[j] - x[j]) for j in range(len(x))])
                self.ma_coef[i] -= -2 * sum([((predictions[j] - x[j]) * get_residual(x, predictions, j)) for j in range(len(x))])
    def difference(self, x, d): # Apply differencing to input list
        x_lasts = [] # Store last values for reverse differencing
        for _ in range(d):
            x_lasts.append(x[-1]) # Store last value
            x = [x[i] - x[i - 1] for i in range(1, len(x))] # Apply differencing
        return x, x_lasts
    def reverse_difference(self, x_diff, x_lasts): # Reverse differencing of input list
        def remove_differencing_step(x_diff, initial):
            x = [initial] # Start at initial value
            for i in range(len(x_diff)):
                x.append(x[-1] + x_diff[i]) # Reverse differencing
            return x[1:] # Remove initial value
        for d in range(self.diff):
            x_diff = remove_differencing_step(x_diff, x_lasts[-1 - d]) # Reverse differencing
        return x_diff
    def predict(self, x, forecasts = 10, forecasts_only = True): # Return model prediction for given inputs list
        x, x_lasts = self.difference(x, self.diff) # Apply differencing
        predictions_diff = []
        for i in range(len(x) + forecasts):
            predictions_diff.append(self.const + sum([self.ar_coef[j] * (predictions_diff[i - j] if i - j >= len(x) else x[i - j]) for j in range(1, len(self.ar_coef)) if i - j >= 0]) + sum([self.ma_coef[j] * (predictions_diff[i - j] - x[i - j]) for j in range(1, len(self.ma_coef)) if 0 <= i - j < len(x)]))
        if forecasts > 0:
            forecasts = self.reverse_difference(predictions_diff[-forecasts:], x_lasts) # Separate forecasts and invert differencing
        return ([] if forecasts_only else (predictions_diff[:-forecasts] if forecasts > 0 else predictions_diff)) + forecasts
    def loss(self, x):
        predictions = self.predict(x, forecasts = 0, forecasts_only = False)
        return sum([(x[i] - predictions[i])**2 for i in range(len(x))]) / len(x) # Return MSE loss