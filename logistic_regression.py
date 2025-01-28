from base_model import Model # Import base Model class
from math import exp # Import exponential function
class LogisticRegression(Model):
    def __init__(self, m = None, a = None, b = None): # No initial weights by default
        self.m, self.a, self.b = m, a, b # Set initial weights if given
    def train(self, x, y, yLimit = None, iterationLimit = 1000): # Train model weights on given data with Newton-Raphson method
        # Input type checks
        if type(x) != list or type(y) != list:
            raise TypeError("Data must be a list") # Invalid data type
        if len(x) != len(y):
            raise ValueError("Data lists must be same length") # Data length mismatch
        # Set initial values
        self.m = max(y) if yLimit is None else yLimit # Initially assume that maximum value is limit if no limit given
        self.a = -1.1 # Initialise to arbitrary constants
        self.b = -1.1
        loss = sum([(y[i] - self.predict(x[i]))**2 for i in range(len(x))]) / len(x) # Calculate initial MSE loss
        minLoss = float("inf")
        minLossM, minLossA, minLossB = self.m, self.a, self.b # Used to revert changes that increase loss
        iteration = 0
        while loss > 0 and iteration < iterationLimit: # Continue until correct parameter values found or iteration limit reached
            if loss < minLoss:
                minLoss = loss
                minLossM, minLossA, minLossB = self.m, self.a, self.b # Used to revert changes that increase loss
            # Use Newton-Raphson method to iteratively improve parameters
            if yLimit is None:
                mGradient = (2 * sum([((1 / (1 + exp(self.a + self.b * x[i]))) * (self.predict(x[i]) - y[i])) for i in range(len(x))]))
                if mGradient != 0: # Prevent div0
                    self.m -= loss / mGradient
            aGradient = (-2 * self.m * exp(self.a) * sum([(exp(self.b * x[i]) * (1 + exp(self.a + self.b * x[i]))**-2 * (self.predict(x[i]) - y[i])) for i in range(len(x))]))
            if aGradient != 0:
                self.a -= loss / aGradient
            bGradient = (-2 * self.m * exp(self.a) * sum([x[i] * (exp(self.b * x[i]) * (1 + exp(self.a + self.b * x[i]))**-2 * (self.predict(x[i]) - y[i])) for i in range(len(x))]))
            if bGradient != 0:
                self.b -= loss / bGradient
            loss = sum([(y[i] - self.predict(x[i]))**2 for i in range(len(x))]) / len(x) # Calculate new MSE loss
            iteration += 1
        self.m, self.a, self.b = minLossM, minLossA, minLossB # Use best known parameters
    def predict(self, x): # Predict output for given input
        if self.m is None or self.a is None or self.b is None:
            return None # No learned weights
        return self.m / (1 + exp(self.a + self.b * x)) # y = m / (1 + e^(a + bx))
expModel = LogisticRegression()
targetModel = LogisticRegression(1, 5, -1) # Used to generate data to train model with
expModel.train([a for a in range(-30, 30)], [targetModel.predict(a) for a in range(-30, 30)], 1) # Train with Newton-Raphson method
print("-30", expModel.predict(-30), "->", targetModel.predict(-30)) # Display model results with comparison to target model
print("-15", expModel.predict(-15), "->", targetModel.predict(-15))
print("-1", expModel.predict(-1), "->", targetModel.predict(-1))
print("0", expModel.predict(0), "->", targetModel.predict(0))
print("1", expModel.predict(1), "->", targetModel.predict(1))
print("15", expModel.predict(15), "->", targetModel.predict(15))
print("30", expModel.predict(30), "->", targetModel.predict(30))
print("m", expModel.m, "->", targetModel.m)
print("a", expModel.a, "->", targetModel.a)
print("b", expModel.b, "->", targetModel.b)
print(expModel.loss([a for a in range(-30, 30)], [targetModel.predict(a) for a in range(-30, 30)]))