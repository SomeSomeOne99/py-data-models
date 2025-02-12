from base_model import Model # Import base Model class
class ARIMA(Model): # General model class
    def __init__(self):
        self.const, self.ar_coef, self.diff, self.ma_coef = 0, [0], [0] # Initialize coefficients
    #def train(self, y): # Train parameters of model to given data
    #    pass
    def predict(self, x, predictions = 0): # Return model prediction for given inputs list
        # Apply differencing
        for _ in range(self.diff):
            x = [x[i] - x[i - 1] for i in range(1, len(x))]
        predictions = []
        for i in range(len(x) + predictions):
            predictions.append(self.const + sum([self.ar_coef[j] * (predictions[i - j] if i - j >= len(x) else x[i - j]) for j in range(1, len(self.ar_coef)) if i - j >= 0]) + sum([self.ma_coef[j] * (predictions[i - j] - x[i - j]) for j in range(1, len(self.ma_coef)) if 0 <= i - j < len(x)]))
        return predictions
    def loss(self, x):
        predictions = self.predict(x)
        return sum([(x[i] - predictions[i])**2 for i in range(len(x))]) / len(x) # Return MSE loss