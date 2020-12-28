from batch_dataset import dataset
import matplotlib.pyplot as plt
import numpy as np


def cleaning_input(data):
    for i in range(len(data[0])):
        everage_data = np.sum(data[:, i]) / len(data)
        min_data = np.min(data[:, i])
        data[:, i] = (data[:, i] - everage_data) / min_data
    
    length = len(data)
    one = np.ones([length, 1])
    data = np.append(one, data, 1)
    
    return data


def hypotheses(x, theta):
    return np.sum(theta.transpose() * x)


def cost_function(theta, dataset):
    summary = 0
    for example in dataset:
        hyp = hypotheses(example[:-1], theta)
        summary += (hyp - example[-1]) ** 2
    '''
    return sum([(hypotheses(example[:-1], theta) - example[-1]) ** 2 for example in dataset])\
            / (2 * len(dataset))'''
    return summary / (2 * len(dataset))


def learning_algorithm(dataset, alpha=0.1, amount_of_iterations=100):
    dataset = np.array(dataset, dtype=np.int64)
    dataset = cleaning_input(dataset)
    theta = np.array([1 for i in range(len(dataset[0]) - 1)])
    error_monitoring = []

    for _ in range(amount_of_iterations):
        error_monitoring.append(cost_function(theta, dataset))

        new = []
        for i in range(len(theta)):
            summary = 0
            for example in dataset:
                hyp = hypotheses(example[:-1], theta)
                summary += (hyp - example[-1]) * example[i]
            
            new.append(theta[i] - (summary / (len(dataset))) * alpha)

        theta = np.array(new)
    
    return dataset, theta, error_monitoring


def main(dataset):
    dataset, theta, error_monitoring = learning_algorithm(dataset, alpha=0.001, amount_of_iterations=1000)

    while True:
        command = input('>> ')
        if command == 'theta':
            print(theta)
        elif command == 'e_all':
            print(error_monitoring)
        elif command == 'error':
            print(error_monitoring[0], error_monitoring[-1])
            plt.plot(range(len(error_monitoring)), error_monitoring)
            plt.show()
        elif command == 'result':  # possible only with 2 features!!
            if len(theta) > 2:
                print('impossible, # features > 2')
                continue
            plt.plot(dataset[:, 1], dataset[:, 2], 'o')
            data_range = [min(dataset[:, 1]), max(dataset[:, 2])]
            plt.plot(data_range,
                    [theta[0] + theta[1] * data_range[0],
                    theta[0] + theta[1] * data_range[1]])
            plt.show()
        elif command == 'prediction':
            data = input('your data(number)')
            if data.isdigit():
                print(sum([int(data) * float(i) for i in theta[1:]]) + theta[0])
            else:
                print('not a number')
        elif command == 'help':
            print('theta - print theta values')
            print('e_all - show error values')
            print('error - show error graph')
            print('result - show outcome function graph with the dataset')
            print('prediction {number} - make prediction, based on learned values')
            print('finish - finish program')
        elif command == 'finish':
            break


if __name__ == '__main__':
    main(dataset)
# by the way, this one is more useful: θ=(XT *  X)−1 * XT * y1