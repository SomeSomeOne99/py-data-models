from base_model import Model # Import base Model class
from math import exp, log # Import exponential and logarithm function
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
        abs_x = [abs(val) for val in x]
        self.a = log((1 / y[abs_x.index(min(abs_x))]) - 1) # Estimate parameters, assuming min(abs(x))=0, therefore y=1/(1+e^a)
        self.b = (log((1 / y[abs_x.index(max(abs_x))]) - 1) - self.a) / x[abs_x.index(max(abs_x))] # Use final point for more accurate estimation
        loss = self.loss(x, y) # Calculate initial MSE loss
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
            loss = self.loss(x, y) # Calculate new MSE loss
            iteration += 1
        self.m, self.a, self.b = minLossM, minLossA, minLossB # Use best known parameters
    def predict(self, x): # Predict output for given input
        if self.m is None or self.a is None or self.b is None:
            return None # No learned weights
        return self.m / (1 + exp(self.a + self.b * x)) # y = m / (1 + e^(a + bx))
def example_train():
    expModel = LogisticRegression()
    targetModel = LogisticRegression(1, 5, -1) # Used to generate data to train model with
    expModel.train([a for a in range(-30, 30)], [targetModel.predict(a) for a in range(-30, 30)], 1) # Train with Newton-Raphson method
    for x in [-30, -15, -1, 0, 1, 1, 15, 30]:
        print(x, ":", expModel.predict(x), "->", targetModel.predict(x)) # Display model results with comparison to target model
    print("m", expModel.m, "->", targetModel.m)
    print("a", expModel.a, "->", targetModel.a)
    print("b", expModel.b, "->", targetModel.b)
    print(expModel.loss([a for a in range(-30, 30)], [targetModel.predict(a) for a in range(-30, 30)]))
example_train()