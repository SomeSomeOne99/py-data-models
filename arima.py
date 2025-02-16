from base_model import Model # Import base Model class
from random import random # Import random function
from math import isnan # Import isnan function
import matplotlib.pyplot as plt
class ARIMA(Model): # General model class
    def __init__(self, const = 0, ar_coef = [0], diff = 0, ma_coef = [0]): # Initialize model with given parameters
        self.const, self.ar_coef, self.diff, self.ma_coef = const, ar_coef, diff, ma_coef # Set parameters
    def train(self, x, ar_order, diff_order, ma_order, iterationLimit = 1000): # Train parameters of model to given data with given orders
        def clip(x, a = -.1, b = .1):
            return max(a, min(b, x))
        self.const = random()
        self.ar_coef = [random() for _ in range(ar_order)] # Initialize random AR coefficients
        self.diff = diff_order
        self.ma_coef = [random() for _ in range(ma_order)] # Initialize random MA coefficients
        x_diff, _ = self.difference(x, diff_order) # Apply differencing
        for _ in range(iterationLimit):
            predictions = self.predict(x, forecasts_num = 0, forecasts_only = False) # Get predictions for current parameters
            for i in range(len(self.ar_coef)):
                self.ar_coef[i] -= clip(2 * sum([((predictions[j] - x[j]) * x_diff[j]) for j in range(len(x))]) * 1e-5)
                self.ar_coef[i] = clip(self.ar_coef[i], -1, 1)
                if isnan(self.ar_coef[i]):
                    print("nan!!")
                    self.ar_coef[i] = random() # Re-initialise parameter
            self.const -= clip(2 * sum([(predictions[j] - x[j]) for j in range(len(x))]) * 1e-5)
            if isnan(self.const):
                print("nan!!")
                self.const = random() # Re-initialise parameter
            for i in range(len(self.ma_coef)):
                self.ma_coef[i] -= clip(2 * sum([((predictions[j] - x[j]) ** 2) for j in range(len(x))]) * 1e-5)
                self.ma_coef[i] = clip(self.ma_coef[i], -1, 1)
                if isnan(self.ma_coef[i]):
                    print("nan!!")
                    self.ma_coef[i] = random() # Re-initialise parameter
    def difference(self, x, d): # Apply differencing to input list
        x_initials = [] # Store last values for reverse differencing
        for _ in range(d):
            x_initials.append(x[0]) # Store first value
            x = [0] + [x[i] - x[i - 1] for i in range(1, len(x))] # Apply differencing
        return x, x_initials
    def reverse_difference(self, x_diff, x_initials): # Reverse differencing of input list
        def remove_differencing_step(x_diff, initial):
            x = [initial] # Start at initial value
            for i in range(len(x_diff)):
                x.append(x[-1] + x_diff[i] - self.const) # Reverse differencing
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
            predictions_diff.append(self.const + sum([self.ar_coef[j] * (predictions_diff[i - j] if i - j >= len(x) else x[i - j]) for j in range(1, len(self.ar_coef)) if i - j >= 0]) + sum([self.ma_coef[j] * (predictions_diff[i - j] - x[i - j]) for j in range(1, len(self.ma_coef)) if 0 <= i - j < len(x)]))
        return self.reverse_difference(([] if forecasts_only else ((predictions_diff[:-forecasts_num] if forecasts_num > 0 else predictions_diff))) + (predictions_diff[-forecasts_num:] if forecasts_num > 0 else []), x_initials)
    def predict_diff(self, x, forecasts_num = 10, forecasts_only = True): # Return predictions without reversing difference
        x, _ = self.difference(x, self.diff) # Apply differencing and discard x_initials
        predictions_diff = []
        for i in range(len(x) + forecasts_num):
            predictions_diff.append(self.const + sum([self.ar_coef[j] * (predictions_diff[i - j] if i - j >= len(x) else x[i - j]) for j in range(1, len(self.ar_coef)) if i - j >= 0]) + sum([self.ma_coef[j] * (predictions_diff[i - j] - x[i - j]) for j in range(1, len(self.ma_coef)) if 0 <= i - j < len(x)]))
        return ([] if forecasts_only else (predictions_diff[:-forecasts_num] if forecasts_num > 0 else predictions_diff)) + (predictions_diff[-forecasts_num:] if forecasts_num > 0 else [])
    def loss(self, x):
        predictions = self.predict(x, forecasts_num = 0, forecasts_only = False)
        return sum([(x[i] - predictions[i])**2 for i in range(len(x))]) / len(x) # Return MSE loss

arimaModel = ARIMA() # Create ARIMA model

inputs = [random()] # Generate random inputs
for x in range(1, 500):
    inputs.append(inputs[x - 1] + random() * 0.1 - 0.05)
inputs = arimaModel.normalise(inputs) # Normalise input data

arimaModel.train(inputs, 2, 0, 0) # Train model on full inputs
predictions = arimaModel.predict(inputs, 50, forecasts_only = False) # Predict data from full inputs
print(arimaModel.loss(inputs))

arimaModel.train(inputs, 2, 0, 1) # Train model on full inputs
predictions2 = arimaModel.predict(inputs, 50, forecasts_only = False) # Predict data from full inputs
print(arimaModel.loss(inputs))

arimaModel.train(inputs, 2, 0, 2) # Train model on full inputs
predictions3 = arimaModel.predict(inputs, 50, forecasts_only = False) # Predict data from full inputs
print(arimaModel.loss(inputs))

# Plot predictions
plt.figure(figsize=(12, 6))
plt.plot(range(len(inputs)), inputs, label='Input Data')
plt.plot(range(len(predictions)), predictions, label='Predictions', linestyle='--')
plt.plot(range(len(predictions2)), predictions2, label='Predictions2', linestyle='dotted')
plt.plot(range(len(predictions3)), predictions3, label='Predictions3', linestyle='-')
plt.legend()
plt.title('ARIMA Model Predictions')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()