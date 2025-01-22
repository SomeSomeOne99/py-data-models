from base_model import Model # Import base Model class
class ExponentialRegression(Model):
    def __init__(self, a = None, b = None): # No initial weights by default
        self.a, self.b = a, b # Set initial weights if given
    def train(self, x, y, iterationLimit = 10000): # Train model weights on given data with Newton-Raphson method
        # Input type checks
        if type(x) != list or type(y) != list:
            raise TypeError("Data must be a list") # Invalid data type
        if len(x) != len(y):
            raise ValueError("Data lists must be same length") # Data length mismatch
        self.a = x[0] # Set weight to first data point
        self.b = (y[0] / x[0]) ** (1 / x[0]) # Set initial weight value using first value
        loss = sum([(y[i] - self.predict(x[i]))**2 for i in range(len(x))]) / len(x) # Calculate initial MSE loss
        prevLoss = float("inf")
        prevB = self.b # Used to revert negative changes
        iteration = 0
        while loss > 0 and iteration < iterationLimit and prevLoss > loss: # Continue until correct parameter value found or iteration limit reached
            prevLoss = loss
            prevB = self.b
            self.b -= loss / (2 * self.a * sum([(x[i] * (self.b ** (x[i] - 1)) * (self.a * (self.b ** x[i]) - y[i])) for i in range(len(x))])) # Use Newton-Raphson method to iteratively improve parameter
            loss = sum([(y[i] - self.predict(x[i]))**2 for i in range(len(x))]) / len(x) # Calculate new MSE loss
        if loss > prevLoss:
            self.b = prevB
    def train_naive(self, x, y, initialPrecision = 10, finalPrecision = -15): # Alternative naive iteration algorithm
        # Input type checks
        if type(x) != list or type(y) != list:
            raise TypeError("Data must be a list") # Invalid data type
        if len(x) != len(y):
            raise ValueError("Data lists must be same length") # Data length mismatch
        self.a = x[0] # Set weight to first data point
        self.b = (y[0] / x[0]) ** (1 / x[0]) # Set initial weight value using first value
        change = 10 ** initialPrecision # Initial weight variation
        minChange = 10 ** finalPrecision # Minimum weight variation for given precision
        while change > minChange: # Continue until minimum precision achieved
            bChange = change
            loss = sum([(y[i] - self.predict(x[i]))**2 for i in range(len(x))]) / len(x) # Calculate MSE loss
            while True:
                self.b += bChange
                newLoss = sum([(y[i] - self.predict(x[i]))**2 for i in range(len(x))]) / len(x) # Calculate new MSE loss after change
                if newLoss >= loss:
                    self.b -= bChange # Reverse change
                    bChange *= -1 # Reverse change direction
                    if bChange > 0: # Change is positive after two switches
                        break # current precision achieved
                else:
                    loss = newLoss
            change *= 0.1 # Increase variation precision
    def predict(self, input): # Predict output for given input
        if self.a is None or self.b is None:
            return None # No learned weights
        return self.a * (self.b ** input) # y = a b^x
expModel = ExponentialRegression()
expModel.train([1, 3], [2.718, 20.086]) # Train with Newton-Raphson method
print(expModel.predict(10)) # expecting e^10 = ~22026.5
expModel.train_naive([1, 3], [2.718, 20.086]) # Train with naive iterative method (clears past learning)
print(expModel.predict(10))