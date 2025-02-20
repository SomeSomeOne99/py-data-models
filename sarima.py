from base_model import Model # Import base Model class
from grid_search import grid_search
from random import random # Import random function
from math import isnan # Import isnan function
import matplotlib.pyplot as plt
class SARIMA(Model): # General model class
    def __init__(self, const = 0, ar_coef = [0], diff = 0, ma_coef = [0], m = 1): # Initialize model with given parameters
        self.const, self.ar_coef, self.diff, self.ma_coef, self.m = const, ar_coef, diff, ma_coef, m # Set parameters
    def train(self, x, ar_order, diff_order, ma_order, m, iterationLimit = 1000): # Train parameters of model to given data with given orders
        def clip(x, a = -.1, b = .1):
            return max(a, min(b, x))
        self.const = random()
        self.ar_coef = [random() for _ in range(ar_order)] # Initialize random AR coefficients
        self.diff = diff_order
        self.ma_coef = [random() for _ in range(ma_order)] # Initialize random MA coefficients
        self.m = m
        x_diff, _ = self.difference(x, diff_order) # Apply differencing
        for _ in range(iterationLimit):
            predictions = self.predict(x, forecasts_num = 0, forecasts_only = False) # Get predictions for current parameters
            for i in range(len(self.ar_coef)):
                self.ar_coef[i] -= clip(2 * sum([((predictions[j] - x[j]) * x_diff[j - ((i + 1) * self.m)]) for j in range(len(x))]) * 1e-5)
                self.ar_coef[i] = clip(self.ar_coef[i], -1, 1)
                if isnan(self.ar_coef[i]):
                    print("nan!!")
                    self.ar_coef[i] = random() # Re-initialise parameter
            self.const -= clip(2 * sum([(predictions[j] - x[j]) for j in range(len(x))]) * 1e-5)
            if isnan(self.const):
                print("nan!!")
                self.const = random() # Re-initialise parameter
            for i in range(len(self.ma_coef)):
                self.ma_coef[i] -= clip(2 * sum([((predictions[j] - x[j]) * (predictions[j] - x[j - ((i + 1) * self.m)])) for j in range(len(x))]) * 1e-5)
                self.ma_coef[i] = clip(self.ma_coef[i], -1, 1)
                if isnan(self.ma_coef[i]):
                    print("nan!!")
                    self.ma_coef[i] = random() # Re-initialise parameter
    def difference(self, x, d): # Apply differencing to input list
        x_initials = [] # Store last values for reverse differencing
        for _ in range(d):
            x_initials.append(x[:self.m + 1]) # Store first season values
            x = [0] + [x[i] - x[i - self.m] for i in range(1, len(x))] # Apply differencing
        return x, x_initials
    def reverse_difference(self, x_diff, x_initials): # Reverse differencing of input list
        def remove_differencing_step(x_diff, initial):
            x = initial.copy() # Start at initial season values
            for i in range(len(x_diff)):
                x.append(x[-1 - self.m] + x_diff[i] - self.const) # Reverse differencing
            return x[1:] # Remove initial value
        for d in range(len(x_initials)):
            x_diff = remove_differencing_step(x_diff, x_initials[-1 - d]) # Reverse differencing
        return x_diff
    def normalise(self, x): # Normalise input list
        x_min = min(x)
        x_range = max(x) - min(x)
        return [(x[i] - x_min) / x_range for i in range(len(x))] # Normalise values
    def predict(self, x, forecasts_num = 10, forecasts_only = True): # Return model prediction for given inputs list
        x, x_initials = self.difference(x, self.diff) # Apply differencing
        predictions_diff = []
        for i in range(len(x) + forecasts_num):
            predictions_diff.append(self.const + sum([self.ar_coef[j] * (predictions_diff[i - j * self.m] if i - j >= len(x) else x[i - j * self.m]) for j in range(1, len(self.ar_coef)) if i - j * self.m >= 0]) + sum([self.ma_coef[j] * (predictions_diff[i - j * self.m] - x[i - j * self.m]) for j in range(1, len(self.ma_coef)) if 0 <= i - j * self.m < len(x)]))
        return self.reverse_difference(([] if forecasts_only else ((predictions_diff[:-forecasts_num] if forecasts_num > 0 else predictions_diff))) + (predictions_diff[-forecasts_num:] if forecasts_num > 0 else []), x_initials)
    def predict_diff(self, x, forecasts_num = 10, forecasts_only = True): # Return predictions without reversing difference
        x, _ = self.difference(x, self.diff) # Apply differencing and discard x_initials
        predictions_diff = []
        for i in range(len(x) + forecasts_num):
            predictions_diff.append(self.const + sum([self.ar_coef[j] * (predictions_diff[i - j * self.m] if i - j >= len(x) else x[i - j * self.m]) for j in range(1, len(self.ar_coef)) if i - j * self.m >= 0]) + sum([self.ma_coef[j] * (predictions_diff[i - j * self.m] - x[i - j * self.m]) for j in range(1, len(self.ma_coef)) if 0 <= i - j * self.m < len(x)]))
        return ([] if forecasts_only else (predictions_diff[:-forecasts_num] if forecasts_num > 0 else predictions_diff)) + (predictions_diff[-forecasts_num:] if forecasts_num > 0 else [])
    def loss(self, x, targets = None):
        if targets is None:
            targets = x # If no targets given, use inputs as targets
        predictions = self.predict(x, forecasts_num = len(targets) - len(x), forecasts_only = False) # Calculate predictions with forecasts as necessary to match length of targets
        return sum([(targets[i] - predictions[i])**2 for i in range(len(targets))]) / len(targets) # Return MSE loss
if __name__ == "__main__":
    sarimaModel = SARIMA() # Create ARIMA model
    inputs = [random() * 0.05]
    for _ in range(19):
        inputs.append(inputs[-1] + random() * 0.1) # Generate random inputs for season of 20
    for x in range(20, 500):
        inputs.append(inputs[x - 20] + random() * 0.01 - 0.005)
    inputs = sarimaModel.normalise(inputs) # Normalise input data

    sarimaModel.train(inputs, 2, 0, 0, 20) # Train model on full inputs
    predictions = sarimaModel.predict(inputs, 50, forecasts_only = False) # Predict data from full inputs
    print(sarimaModel.loss(inputs))

    sarimaModel.train(inputs, 2, 0, 1, 20) # Train model on full inputs
    predictions2 = sarimaModel.predict(inputs, 50, forecasts_only = False) # Predict data from full inputs
    print(sarimaModel.loss(inputs))

    sarimaModel.train(inputs, 3, 0, 1, 20) # Train model on full inputs
    predictions3 = sarimaModel.predict(inputs, 50, forecasts_only = False) # Predict data from full inputs
    print(sarimaModel.loss(inputs))

    loss, hyperparameters = grid_search(sarimaModel, inputs, None, 20, list(range(15)), list(range(3)), list(range(10)), [20])
    sarimaModel.train(inputs, *hyperparameters) # Train model on full inputs
    predictions_best = sarimaModel.predict(inputs, 50, forecasts_only = False) # Predict data from full inputs
    print(sarimaModel.loss(inputs), hyperparameters)

    # Plot predictions
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(inputs)), inputs, label='Input Data')
    plt.plot(range(len(predictions)), predictions, label='Predictions [2, 0, 0, 20]', linestyle='--')
    plt.plot(range(len(predictions2)), predictions2, label='Predictions [2, 0, 1, 20]', linestyle='dotted')
    plt.plot(range(len(predictions3)), predictions3, label='Predictions [3, 0, 1, 20]', linestyle='-')
    plt.plot(range(len(predictions_best)), predictions_best, label='Predictions (Best) ' + str(hyperparameters), linestyle='solid')
    plt.legend()
    plt.title('SARIMA Model Predictions')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.show()