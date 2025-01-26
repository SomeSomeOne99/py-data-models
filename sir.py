from base_model import Model # Import base Model class
class SIR(Model):
    def __init__(self, infRate = None, recRate = None): # (infection rate, recovery rate) No initial weights by default
        self.infRate, self.recRate = infRate, recRate # Set initial weights if given
    def train(self, x, y, iterationLimit = 10000): # Train model weights on given data with Newton-Raphson method
        # Input type checks
        if type(x) != list or type(y) != list:
            raise TypeError("Data must be a list") # Invalid data type
        if len(x) != len(y):
            raise ValueError("Data lists must be same length") # Data length mismatch
        if len(x) < 2:
            raise ValueError("Data must have two or more items")
        # Train infection rate
        self.infRate = (1 - y[1][0] / y[0][0]) ** (1 / (x[1] - x[0])) # Use first pair of points for initial estimation
        self.recRate = 0.1 #y[2][2] / y[1][2] - 1
        loss = sum([[(y[i][j] - x_)**2 for x_ in self.predict(x[i])] for i in range(len(x))]) / len(x) # Initial MSE loss
        minLoss = float("inf")
        bestInfRate = self.infRate
        iteration = 0
        while loss > 0 and iteration < iterationLimit: # Continue until correct parameter value found or iteration limit reached
            if minLoss > loss:
                minLoss = loss
                bestInfRate = self.infRate # Preserve best known value
            self.infRate -= loss / (2 * sum([((-self.infRate * x[i - 1][0] * x[i - 1][1]) * (self.predict(x[i])[0] - y[i][0])) for i in range(len(1, x))])) # Train infection rate using Newton-Raphson method with S and I data
            loss = sum([(y[i] - self.predict(x[i]))**2 for i in range(len(x))]) / len(x) # New MSE loss
            iteration += 1
        if loss > minLoss:
            self.infRate = bestInfRate
        loss = sum([(y[i] - self.predict(x[i]))**2 for i in range(len(x))]) / len(x) # Calculate initial MSE loss
        minLoss = float("inf")
        bestRecRate = self.recRate
        iteration = 0
        while loss > 0 and iteration < iterationLimit: # Continue until correct parameter value found or iteration limit reached
            if minLoss > loss:
                bestRecRate = self.recRate # Preserve best known value
            self.recRate -= loss / (2 * () * sum([((self.recRate * x[i - 1][1] * x[i - 1][2]) * (self.predict(x[i]) - y[i])) for i in range(len(x))])) # Train infection rate using Newton-Raphson method with I and R data
            loss = sum([(y[i] - self.predict(x[i]))**2 for i in range(len(x))]) / len(x) # Calculate new MSE loss
            iteration += 1
        if loss > minLoss:
            self.recRate = bestRecRate
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
sirModel.train([x for x in range(100)], [targetModel.predict(x) for x in range(100)]) # Train with Newton-Raphson method
print(sirModel.predict(10))
print("train_naive")
sirModel.train_naive([x for x in range(100)], [targetModel.predict(x) for x in range(100)]) # Train with naive iterative method (clears past learning)
for testX in [0, 10, 100, 1000]:
    print(sirModel.predict(testX), "->", targetModel.predict(testX)) # Predict for given inputs
print(sirModel.infRate, "->", targetModel.infRate)
print(sirModel.recRate, "->", targetModel.recRate)