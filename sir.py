from base_model import Model # Import base Model class
class SIR(Model):
    def __init__(self, infRate = None, recRate = None): # (infection rate, recovery rate) No initial weights by default
        self.infRate, self.recRate = infRate, recRate # Set initial weights if given
        self.savedResults = []
    def train(self, x, y, iterationLimit = 1000, initialInf = 0.1, initialRec = 0): # Train model weights on given data with gradient descent (correct to ~2sf)
        # Input type checks
        if type(x) != list or type(y) != list:
            raise TypeError("Data must be a list") # Invalid data type
        if len(x) != len(y):
            raise ValueError("Data lists must be same length") # Data length mismatch
        if len(x) < 2:
            raise ValueError("Data must have two or more items")
        # Train infRate with S and I data
        self.infRate = (y[1][0] - y[2][0]) / (y[1][1] * y[1][0]) # Estimate initial parameters
        self.recRate = (y[2][2] - y[1][2]) / y[1][1]
        for _ in range(iterationLimit): # Continue until iteration limit reached
            self.infRate -= sum([(y[i - 1][0] * y[i - 1][1] * ((y[i][0] - self.predict(x[i], initialInf, initialRec)[0]) + (y[i][1] - self.predict(x[i], initialInf, initialRec)[1]))) for i in range(1, len(x))]) # Use gradient descent to minimise loss with S and I data
            self.recRate -= sum([(y[i - 1][1] * y[i - 1][2] * ((y[i][1] - self.predict(x[i], initialInf, initialRec)[1]) + (y[i][2] - self.predict(x[i], initialInf, initialRec)[2]))) for i in range(1, len(x))]) # Use gradient descent to minimise loss with I and R data
            self.savedResults = [] # Reset saved values
    def train_(self, x, y, iterationLimit = 1000, initialInf = 0.1, initialRec = 0): # Gradient descent with use of best parameter tracking (may be less accurate than train())
        # Input type checks
        if type(x) != list or type(y) != list:
            raise TypeError("Data must be a list") # Invalid data type
        if len(x) != len(y):
            raise ValueError("Data lists must be same length") # Data length mismatch
        if len(x) < 2:
            raise ValueError("Data must have two or more items")
        # Train infRate with S and I data
        self.infRate = (y[1][0] - y[2][0]) / (y[1][1] * y[1][0]) # Estimate initial parameters
        self.recRate = (y[2][2] - y[1][2]) / y[1][1]
        minLoss = sum([sum([(y[i][j] - predY)**2 for j, predY in enumerate(self.predict(x[i], initialInf, initialRec)[:2])]) for i in range(len(x))]) # Initial loss for S and I data
        bestInfRate = self.infRate
        for _ in range(iterationLimit): # Continue until iteration limit reached
            loss = sum([sum([(y[i][j] - predY)**2 for j, predY in enumerate(self.predict(x[i], initialInf, initialRec)[:2])]) for i in range(len(x))]) # New MSE loss
            if loss == 0: # Perfect loss
                return
            self.infRate -= sum([(y[i - 1][0] * y[i - 1][1] * ((y[i][0] - self.predict(x[i], initialInf, initialRec)[0]) + (y[i][1] - self.predict(x[i], initialInf, initialRec)[1]))) for i in range(1, len(x))]) # Use gradient descent to minimise loss with S and I data
            self.savedResults = [] # Reset saved values
            if loss < minLoss:
                minLoss = loss
                bestInfRate = self.infRate # Preserve best infRate
        self.infRate = bestInfRate # Use best known infRate # MAY REDUCE ACCURACY
        # Train recRate with I and R data
        minLoss = sum([sum([(y[i][j] - predY)**2 for j, predY in enumerate(self.predict(x[i], initialInf, initialRec)[1:])]) for i in range(len(x))]) # Initial loss for I and R data
        bestRecRate = self.recRate
        for _ in range(iterationLimit): # Continue until iteration limit reached
            loss = sum([sum([(y[i][j] - predY)**2 for j, predY in enumerate(self.predict(x[i], initialInf, initialRec)[1:])]) for i in range(len(x))]) # New MSE loss
            if loss == 0: # Perfect loss
                return
            self.recRate -= sum([(y[i - 1][1] * y[i - 1][2] * ((y[i][1] - self.predict(x[i], initialInf, initialRec)[1]) + (y[i][2] - self.predict(x[i], initialInf, initialRec)[2]))) for i in range(1, len(x))]) # Use gradient descent to minimise loss with I and R data
            self.savedResults = [] # Reset saved values
            if loss < minLoss:
                minLoss = loss
                bestRecRate = self.recRate # Preserve best recRate
        self.recRate = bestRecRate # Use best known recRate # APPEARS TO REDUCE ACCURACY
    def train_naive(self, x, y, initialInf = 0.1, initialRec = 0, initialPrecision = -1, finalPrecision = -15): # Alternative naive iteration algorithm
        # Input type checks
        if type(x) != list or type(y) != list:
            raise TypeError("Data must be a list") # Invalid data type
        if len(x) != len(y):
            raise ValueError("Data lists must be same length") # Data length mismatch
        self.infRate = 0.1 # Arbitrary initialisation
        self.recRate = 0.1
        infChange = 10 ** initialPrecision # Initial weight increment
        minChange = 10 ** finalPrecision # Minimum weight increment for given precision
        while infChange > minChange: # Continue until minimum precision achieved
            loss = self.loss(x, y) # Calculate MSE loss (average deemed unnecessary, length division omitted)
            while True:
                self.infRate += infChange
                self.savedResults = [] # Reset saved values
                # Find optimal recRate for current infRate
                recChange = 10 ** initialPrecision # New increment for recRate
                while recChange > minChange: # Continue until minimum precision achieved for recRate
                    loss2 = self.loss(x, y) # MSE, loss2 to prevent interference with infRate
                    while True:
                        self.recRate += recChange
                        self.savedResults = [] # Reset saved values
                        newLoss2 = self.loss(x, y) # MSE, newLoss2 to prevent interference with infRate
                        if newLoss2 >= loss2:
                            self.recRate -= recChange # Reverse change
                            self.savedResults = [] # Reset saved values
                            recChange *= -1 # Reverse change direction
                            if recChange > 0: # Change is positive after two switches
                                break # Current precision achieved
                        else:
                            loss2 = newLoss2
                    recChange *= 0.1 # Increase increment precision
                newLoss = self.loss(x, y) # New MSE loss after changes
                if newLoss >= loss:
                    self.infRate -= infChange # Reverse change
                    self.savedResults = [] # Reset saved values
                    infChange *= -1 # Reverse change direction
                    if infChange > 0: # Change is positive after two switches
                        break # Current precision achieved
                else:
                    loss = newLoss
            infChange *= 0.1 # Increase increment precision
    def predict(self, x, initialInf = 0.1, initialRec = 0): # Predict output for given input (S, I, R)
        if self.infRate is None or self.recRate is None:
            return None # No learned weights
        if type(x) != int and type(x) != float:
            return TypeError("Input must be a numeric value")
        if x < 0:
            return ValueError("Input must be zero or positive")
        if len(self.savedResults) == 0: # No saved results, start from initial position
            self.savedResults = [[1 - initialInf - initialRec, initialInf, initialRec]] # Initial position
        if x >= len(self.savedResults): # Insufficient saved results:
            for i in range(len(self.savedResults), x + 1): # Generate new results as needed
                self.savedResults.append([self.savedResults[i - 1][0] - (self.infRate * self.savedResults[i - 1][1] * self.savedResults[i - 1][0]),
                    self.savedResults[i - 1][1] + (self.infRate * self.savedResults[i - 1][1] * self.savedResults[i - 1][0]) - (self.recRate * self.savedResults[i - 1][1]),
                    self.savedResults[i - 1][2] + (self.recRate * self.savedResults[i - 1][1])])
        return self.savedResults[x] # Return requested value
    def loss(self, x, y):
        return sum([sum([(y[i][j] - pred)**2 for j, pred in enumerate(self.predict(x[i]))]) for i in range(len(x))]) / len(x * 3)
def example_train():
    sirModel = SIR()
    targetModel = SIR(0.75, 0.5)
    sirModel.train([x for x in range(25)], [targetModel.predict(x) for x in range(25)]) # Train with Newton-Raphson method
    for x in [0, 10, 100, 1000]:
        print(x, ":", sirModel.predict(x), "->", targetModel.predict(x)) # Predict for given inputs
    print("infRate", sirModel.infRate, "->", targetModel.infRate)
    print("recRate", sirModel.recRate, "->", targetModel.recRate)
    print("loss", sirModel.loss([x for x in range(25)], [targetModel.predict(x) for x in range(25)]))
def example_train_naive():
    sirModel = SIR()
    targetModel = SIR(0.75, 0.5)
    sirModel.train_naive([x for x in range(100)], [targetModel.predict(x) for x in range(100)]) # Train with naive iterative method (clears past learning)
    for x in [0, 10, 100, 1000]:
        print(x, ":", sirModel.predict(x), "->", targetModel.predict(x)) # Predict for given inputs
    print("infRate", sirModel.infRate, "->", targetModel.infRate)
    print("recRate", sirModel.recRate, "->", targetModel.recRate)
    print("loss", sirModel.loss([x for x in range(100)], [targetModel.predict(x) for x in range(100)]))