from base_model import Model
from multiprocessing import Process, Queue
from time import sleep
def grid_seach_worker(model: Model, inputs, targets, hyperparameterQueue: Queue, resultsQueue: Queue):
    while not hyperparameterQueue.empty():
        selected_hyperparams = hyperparameterQueue.get(block = False)
        if targets is None:
            model.train(inputs, *selected_hyperparams)
            loss = model.loss(inputs)
        else:
            model.train(inputs, targets, *selected_hyperparams)
            loss = model.loss(inputs, targets)
        resultsQueue.put((loss, selected_hyperparams))
def grid_search(model: Model, inputs, targets = None, worker_processes = 10, *hyperparams):
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
    if __name__ == "grid_search":
        hyperparameterQueue = Queue()
        for hyperparams_is in generate_is(len(hyperparams), [len(hyperparam) - 1 for hyperparam in hyperparams]):
            hyperparameterQueue.put([hyperparams[i][hyperparams_is[i]] for i in range(len(hyperparams))])
        resultsCount = hyperparameterQueue.qsize()
        resultsQueue = Queue()
        processes = [Process(target = grid_seach_worker, args = (model, inputs, targets, hyperparameterQueue, resultsQueue), name = "grid_search_worker") for _ in range(worker_processes)] # Initialise processes
        for process in processes: # Start all processes
            process.start()
        minLoss = float("inf")
        minLossHyperparams = [hyperparam[0] for hyperparam in hyperparams]
        while resultsCount > 0:
            loss, hyperparams = resultsQueue.get()
            if loss < minLoss:
                minLoss = loss
                minLossHyperparams = hyperparams.copy()
            resultsCount -= 1
        for process in processes:
            process.kill()
        return minLoss, minLossHyperparams