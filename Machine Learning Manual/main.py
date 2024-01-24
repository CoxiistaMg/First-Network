from tools import data, backpropagation
from random import random

data.DATA_FILE = 'tools/data/'
iris = data.DataSet('iris.csv', 1)

n1 = backpropagation.Network((iris.len_inputs, 5, 5, iris.len_results), ['tangh', 'tangh'], True)
#n1.trainer_weights_mini_batch(iris.trainer, iris.trainer_results,2, 1000, 0.1)
#n1.trainer_weights_stochastic(iris.trainer, iris.trainer_results, 1000, 0.1)

print(n1.feedforward(iris.trainer[0], iris.trainer_results[0]))

'''n2 = backpropagation.Network((1, 1), [])
n2.trainer_weights_mini_batch(data=[
        backpropagation.Matrix([[0], [0]]),
        backpropagation.Matrix([[1], [0]]),
        backpropagation.Matrix([[0], [1]]),
        backpropagation.Matrix([[1], [1]])
                                        ], responses=[0, 1, 1, 0], len_mini_batch=1, eppochs=70, lr=1)'''

n3 = backpropagation.Network((1, 1), [], False)

lst = []
resp = []
for each in range(150):
    rad = random()
    lst.append(backpropagation.Matrix([[rad]]))
    resp.append(rad*1.8 + 32)

n3.trainer_weights_mini_batch(lst, resp, 1, 1000, 0.1)
