from base_model import Model # Import base Model class
class ARIMA(Model): # General model class
    def __init__(self):
        self.const, self.ar_coef, self.diff, self.ma_coef = 0, [0], 0, [0] # Initialize coefficients
    #def train(self, y): # Train parameters of model to given data
    #    pass
    def predict(self, x, forecasts = 10, forecasts_only = True): # Return model prediction for given inputs list
        def remove_differencing(x_diff, initial):
            x = [initial] # Start at initial value
            for i in range(len(x_diff)):
                x.append(x[-1] + x_diff[i]) # Reverse differencing
            return x[1:] # Remove initial value
        # Apply differencing
        x_lasts = [] # Store last values for reverse differencing
        for _ in range(self.diff):
            x_lasts.append(x[-1]) # Store last value
            x = [x[i] - x[i - 1] for i in range(1, len(x))]
        predictions_diff = []
        for i in range(len(x) + forecasts):
            predictions_diff.append(self.const + sum([self.ar_coef[j] * (predictions_diff[i - j] if i - j >= len(x) else x[i - j]) for j in range(1, len(self.ar_coef)) if i - j >= 0]) + sum([self.ma_coef[j] * (predictions_diff[i - j] - x[i - j]) for j in range(1, len(self.ma_coef)) if 0 <= i - j < len(x)]))
        # Apply inverse differencing
        if forecasts > 0:
            forecasts = predictions_diff[-forecasts:] # Separate forecasts
            for d in range(self.diff):
                forecasts = remove_differencing(forecasts, x_lasts[-1 - d]) # Reverse differencing
        return ([] if forecasts_only else (predictions_diff[:-forecasts] if forecasts > 0 else predictions_diff)) + forecasts
    def loss(self, x):
        predictions = self.predict(x, forecasts = 0, forecasts_only = False)
        return sum([(x[i] - predictions[i])**2 for i in range(len(x))]) / len(x) # Return MSE loss