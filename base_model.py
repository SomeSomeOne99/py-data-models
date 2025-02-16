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
def grid_search(model: Model, inputs, targets = None, *hyperparams):
    def generate_is(is_num, is_limits):
        increment_list = []
        nums = [0 for x in range(is_num)]
        complete = False
        while not complete:
            for i in range(is_num - 1, -2, -1):
                if i == -1:
                    complete = True
                    break
                if nums[i] < is_limits[i]:
                    nums[i] += 1
                    break
                nums[i] = 0
            if not complete:
                increment_list.append(nums.copy())
        return increment_list
    minLoss = float("inf")
    minLossHyperparams = [hyperparam[0] for hyperparam in hyperparams]
    for hyperparams_is in generate_is(len(hyperparams), [len(hyperparam) - 1 for hyperparam in hyperparams]):
        selected_hyperparams = [hyperparams[i][hyperparams_is[i]] for i in range(len(hyperparams))]
        if targets is None:
            model.train(inputs, *selected_hyperparams)
            loss = model.loss(inputs)
        else:
            model.train(inputs, targets, *selected_hyperparams)
            loss = model.loss(inputs, targets)
        if loss < minLoss:
            minLoss = loss
            minLossHyperparams = selected_hyperparams
    return minLoss, minLossHyperparams