from base_model import Model # Import base Model class
from matrix import Matrix # Import Matrix class
class PolynomialRegression(Model):
    def __init__(self, w = None): # No initial weights by default
        self.w = w
    def train(self, x, y, maxPower = 10): # Train model weights on given data
        if type(x) != list and type(x) != list: # Invalid type
            raise TypeError("Input must be a vector as a list")
        X = Matrix([[x[i]**p for p in range(maxPower)] for i in range(len(x))])
        self.w = (X.T() @ X).I() @ X.T() @ y
    def predict(self, x): # Predict output for given input
        if self.w is None:
            return None # No learned weights
        if type(x) != int and type(x) != float: # Invalid type
            raise TypeError("Input must be a numeric type")
        return Matrix([x**p for p in range(len(self.w))]) @ self.w # Apply weights to input
    def loss(self, x, y):
        return sum([sum([(y[i][j] - pred)**2 for j, pred in enumerate(self.predict(x[i]).data)]) / len(y[i]) for i in range(len(x))]) / len(x)
polyModel = PolynomialRegression()
targetModel = PolynomialRegression(Matrix([0, 1]))
polyModel.train([x for x in range(10)], Matrix([x**2 for x in range(10)]), maxPower = 10)
print(polyModel.predict(5))
print(polyModel.w.data)
print(polyModel.loss([x for x in range(10)], [[x**2] for x in range(10)]))