from base_model import Model # Import base Model class
class SIR(Model):
    def __init__(self, infRate = None, recRate = None): # (infection rate, recovery rate) No initial weights by default
        self.infRate, self.recRate = infRate, recRate # Set initial weights if given
    def train(self, x, y, iterationLimit = 10000): # Train model weights on given data with gradient descent
        # Input type checks
        if type(x) != list or type(y) != list:
            raise TypeError("Data must be a list") # Invalid data type
        if len(x) != len(y):
            raise ValueError("Data lists must be same length") # Data length mismatch
        if len(x) < 2:
            raise ValueError("Data must have two or more items")
        # Train infRate with S and I data
        self.infRate = 0.25 # Arbitrary initialisation
        self.recRate = 0.3 # Arbitrary initialisation
        minLoss = sum([sum([(y[i][j] - predY)**2 for j, predY in enumerate(self.predict(x[i])[:2])]) for i in range(len(x))]) # Initial loss for S and I data
        bestInfRate = self.infRate
        for _ in range(iterationLimit): # Continue until iteration limit reached
            loss = sum([sum([(y[i][j] - predY)**2 for j, predY in enumerate(self.predict(x[i])[:2])]) for i in range(len(x))]) # New MSE loss
            if loss == 0: # Perfect loss
                return
            self.infRate -= sum([(y[i - 1][0] * y[i - 1][1] * (y[i][0] - self.predict(x[i])[0]) * (y[i][1] - self.predict(x[i])[1])) for i in range(1, len(x))]) # Use gradient descent to minimise loss with S and I data
            if loss < minLoss:
                minLoss = loss
                bestInfRate = self.infRate # Preserve best infRate
        self.infRate = bestInfRate # Use best known infRate
        # Train recRate with I and R data
        minLoss = sum([sum([(y[i][j] - predY)**2 for j, predY in enumerate(self.predict(x[i])[1:])]) for i in range(len(x))]) # Initial loss for I and R data
        bestRecRate = self.recRate
        for _ in range(iterationLimit): # Continue until iteration limit reached
            loss = sum([sum([(y[i][j] - predY)**2 for j, predY in enumerate(self.predict(x[i])[1:])]) for i in range(len(x))]) # New MSE loss
            if loss == 0: # Perfect loss
                return
            self.recRate -= sum([(y[i - 1][1] * y[i - 1][2] * (y[i][1] - self.predict(x[i])[1]) * (y[i][2] - self.predict(x[i])[2])) for i in range(1, len(x))]) # Use gradient descent to minimise loss with I and R data
            if loss < minLoss:
                minLoss = loss
                bestRecRate = self.recRate # Preserve best recRate
        self.recRate = bestRecRate # Use best known recRate
    def train_naive(self, x, y, initialPrecision = -1, finalPrecision = -15): # Alternative naive iteration algorithm
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
            loss = sum([sum([(y[i][j] - predY)**2 for j, predY in enumerate(self.predict(x[i]))]) for i in range(len(x))]) # Calculate MSE loss (average deemed unnecessary, length division omitted)
            while True:
                self.infRate += infChange
                # Find optimal recRate for current infRate
                recChange = 10 ** initialPrecision # New increment for recRate
                while recChange > minChange: # Continue until minimum precision achieved for recRate
                    loss2 = sum([sum([(y[i][j] - predY)**2 for j, predY in enumerate(self.predict(x[i]))]) for i in range(len(x))]) # MSE, loss2 to prevent interference with infRate
                    while True:
                        self.recRate += recChange
                        newLoss2 = sum([sum([(y[i][j] - predY)**2 for j, predY in enumerate(self.predict(x[i]))]) for i in range(len(x))]) # MSE, newLoss2 to prevent interference with infRate
                        if newLoss2 >= loss2:
                            self.recRate -= recChange # Reverse change
                            recChange *= -1 # Reverse change direction
                            if recChange > 0: # Change is positive after two switches
                                break # Current precision achieved
                        else:
                            loss2 = newLoss2
                    recChange *= 0.1 # Increase increment precision
                newLoss = sum([sum([(y[i][j] - predY)**2 for j, predY in enumerate(self.predict(x[i]))]) for i in range(len(x))]) # New MSE loss after changes
                if newLoss >= loss:
                    self.infRate -= infChange # Reverse change
                    infChange *= -1 # Reverse change direction
                    if infChange > 0: # Change is positive after two switches
                        break # Current precision achieved
                else:
                    loss = newLoss
            infChange *= 0.1 # Increase increment precision
    def predict(self, x, step = 1): # Predict output for given input (S, I, R)
        if self.infRate is None or self.recRate is None:
            return None # No learned weights
        if type(x) != int and type(x) != float:
            return TypeError("Input must be a numeric value")
        if x < 0:
            return ValueError("Input must be zero or positive")
        currentSIR = [0.9, 0.1, 0] # Initial position
        for _ in range(1, x, step):
            currentSIR = [currentSIR[0] - ((self.infRate ** step) * currentSIR[1] * currentSIR[0] / sum(currentSIR)),
                currentSIR[1] + ((self.infRate ** step) * currentSIR[1] * currentSIR[0] / sum(currentSIR)) - ((self.recRate ** step) * currentSIR[1]),
                currentSIR[2] + ((self.recRate ** step) * currentSIR[1])]
        return currentSIR
sirModel = SIR()
targetModel = SIR(0.5, 0.25)
print("train")
sirModel.train([x for x in range(25)], [targetModel.predict(x) for x in range(25)]) # Train with Newton-Raphson method
for testX in [0, 10, 100, 1000]:
    print(sirModel.predict(testX), "->", targetModel.predict(testX)) # Predict for given inputs
print(sirModel.infRate, "->", targetModel.infRate)
print(sirModel.recRate, "->", targetModel.recRate)
print("train_naive")
sirModel.train_naive([x for x in range(100)], [targetModel.predict(x) for x in range(100)]) # Train with naive iterative method (clears past learning)
for testX in [0, 10, 100, 1000]:
    print(sirModel.predict(testX), "->", targetModel.predict(testX)) # Predict for given inputs
print(sirModel.infRate, "->", targetModel.infRate)
print(sirModel.recRate, "->", targetModel.recRate)