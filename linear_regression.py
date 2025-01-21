from base_model import Model # Import base Model class
from matrix import Matrix # Import Matrix class
class LinearRegression(Model):
    def __init__(self, w = None): # No initial weights by default
        self.w = w
    def train(self, X, y): # Train model weights on given data
        self.w = (X.T() @ X).I() @ X.T() @ y
    def predict(self, input): # Predict output for given input
        if self.w is None:
            return None # No learned weights
        return input @ self.w # Apply weights to input
linModel = LinearRegression()
linModel.train(Matrix([[1, 2], [3, 4]]), Matrix([[3], [7]]))
print(linModel.predict(Matrix([[10, 20]])))