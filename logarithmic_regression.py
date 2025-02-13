from base_model import Model, time_function # Import base Model class
from math import log # Import ln()
class LogarithmicRegression(Model):
    def __init__(self, a = None, b = None): # No initial weights by default
        self.a, self.b = a, b
    def train(self, x, y, iterationLimit = 10000): # Train model weights on given data
        self.a = y[0] # Set weight to first data point
        self.b = y[1] - y[0] / log(x[1]) # Set initial weight value using first value
        bestA, bestB = self.a, self.b
        loss = self.loss(x, y) # Calculate initial MSE loss
        minLoss = float("inf")
        iteration = 0
        while loss > 0 and iteration < iterationLimit and minLoss != loss: # Continue until correct parameter value found, iteration limit reached or model stagnation
            if minLoss > loss:
                minLoss = loss
                bestA, bestB = self.a, self.b # New best parameters
            # Use Newton-Raphson method to iteratively improve parameters
            newA = self.a - loss / (2 * sum([(self.predict(x[i]) - y[i]) for i in range(len(x))])) # Prevent change until all gradients calculated
            self.b -= loss / (2 * sum([(log(x[i]) * (self.predict(x[i]) - y[i])) for i in range(len(x))]))
            self.a = newA
            loss = self.loss(x, y) # Calculate new MSE loss
            iteration += 1
        if loss > minLoss:
            self.a, self.b = bestA, bestB
    def predict(self, x): # Predict output for given input
        if self.a is None or self.b is None:
            return None # No learned weights
        return self.a + self.b * log(x) # Apply weights to input
def example_train():
    logModel = LogarithmicRegression()
    targetModel = LogarithmicRegression(2, 4)
    logModel.train([x for x in range(1, 20)], [targetModel.predict(x) for x in range(1, 20)])
    for x in [1, 5, 10, 20]:
        print(x, ":", logModel.predict(x), "->", targetModel.predict(x))
    print("a", logModel.a, "->", targetModel.a)
    print("b", logModel.b, "->", targetModel.b)
    print(logModel.loss([x for x in range(1, 20)], [targetModel.predict(x) for x in range(1, 20)]))
def example_train_long():
    logModel = LogarithmicRegression()
    targetModel = LogarithmicRegression(2, 0.04)
    logModel.train([x for x in range(1, 1000)], [targetModel.predict(x) for x in range(1, 1000)])
def time_example_train_long():
    time_function(example_train_long, repetitions = 10)
time_example_train_long()