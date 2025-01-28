class Model(): # General model class
    def __init__(self):
        pass
    def train(self, data): # Train parameters of model to given data
        pass
    def predict(self, input): # Return model prediction for given input
        pass
    def loss(self, x, y):
        return sum([(y[i] - self.predict(x[i]))**2 for i in range(len(x))]) / len(x)