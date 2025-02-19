from timeit import Timer # Import Timer class
class Model(): # General model class
    def __init__(self):
        pass
    def train(self, x, y): # Train parameters of model to given data
        pass
    def predict(self, x): # Return model prediction for given input
        pass
    def loss(self, x, y):
        return sum([(y[i] - self.predict(x[i]))**2 for i in range(len(x))]) / len(x)
def time_function(func, repetitions = 1000):
    timer = Timer(func) # Create timer object
    time = timer.timeit(repetitions) # Measure time for function
    print(time, "sec total") # Display average time per function call
    print(time / repetitions, "sec/run") # Display average time per function call
def plot_model(model: Model, inputs, targets, *hyperparameters):
    import matplotlib.pyplot as plt
    model.train(inputs, targets, *hyperparameters)
    predictions = [model.predict(input_) for input_ in inputs]
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(targets)), targets, label='Targets')
    plt.plot(range(len(predictions)), predictions, label='Predictions', linestyle='--')
    plt.legend()
    plt.title('Model Predictions')
    plt.ylabel('Value')
    plt.show()