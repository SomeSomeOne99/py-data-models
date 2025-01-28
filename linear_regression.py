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
    def loss(self, x, y):
        return sum([sum([(y[i][j] - pred)**2 for j, pred in enumerate(self.predict(x[i]).data)]) / len(y[i]) for i in range(len(x))]) / len(x)
def example_train():
    linModel = LinearRegression()
    linModel.train(Matrix([[1, 2], [3, 4]]), Matrix([[3], [7]]))
    print("1,2 :", linModel.predict(Matrix([[1, 2]])), "->", 3)
    print("10,20 :", linModel.predict(Matrix([[10, 20]])), "->", 30)
    print("loss", linModel.loss([Matrix([1, 2]), Matrix([3, 4])], [[3], [7]]))