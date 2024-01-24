from .base.matrix import Matrix, Array
from .base.act_func import dict_act_func

from math import e, log
from random import random



class Network:
    def __init__(self, sizes: tuple, activation_funcs_hidden_layer: tuple,  classification=True) -> None:
        self.cost = self.softmax
        self.__hits = 0
        if not classification:
            self.cost = self.linear_regression
            sizes = list(sizes)
            sizes[-1] = 1
            sizes = tuple(sizes)

        self.weights = [ Matrix([[random() for _ in range(weights)] for _ in range(network)])
                for network, weights in zip(sizes[1:], sizes[:-1])]
        self.biases = [Matrix([[random()] for _ in range(network)]) for network in sizes[1:]]
        self.activation_funcs = list(activation_funcs_hidden_layer)
        
        if classification:
            self.cost = self.softmax
        
        self.classification = classification
        
    def feedforward(self, inputs, real_value):
        hist = {
            'modules': [inputs.t()],
            'derivates': [],
            'cost': 0
        }
        layer_w = self.weights[:]
        layer_b = self.biases[:]
        layer_act = list(self.activation_funcs[:])
        last_layer_w = layer_w.pop()
        last_layer_b = layer_b.pop()

        for weights, biases in zip(layer_w, layer_b):
            func = layer_act.pop(0)
            layer = dict_act_func[func](((weights * inputs) + biases).matrix)

            inputs = Matrix(layer['func'])
            hist['modules'].append(inputs)
            hist['derivates'].append(Matrix(layer['derivate']))
        
        preview = ((last_layer_w * inputs) + last_layer_b).matrix 
        preview = self.cost(preview, real_value)

        hist['modules'].append(preview['preview'])
        hist['derivates'].append(preview['derivate_cost'])
        hist['cost'] += preview['cost']
        if self.classification:
            self.__hits += preview['hit']
   
        return hist

    def backpropagation(self, inputs, real_value):
        nabla_w = []
        nabla_b = []
        feedforward = self.feedforward(inputs, real_value)
        modules = feedforward['modules']
        modules.pop()
        derivates = feedforward['derivates']
        cost = feedforward['cost']
        delta = derivates[-1]
      
      
        for pos in range(len(modules))[::-1]:
            nabla_w.append(delta * modules[pos])
            nabla_b.append(delta)
            delta = (delta.t() * self.weights[pos]).matrix[0]
            dv = derivates[pos-1].t().matrix[0]
           
            delta = Matrix([(delta * dv).list]).t()
     
        return nabla_w[::-1], nabla_b[::-1], cost
    
    def trainer_weights_stochastic(self, data, responses, eppochs=10, lr=0.1):
        st_nabla_w = [layer * 0 for layer in self.weights]
        st_nabla_b = [layer * 0 for layer in self.biases]

        for eppoch in range(eppochs):
            self.__hits = 0
            nabla_w = st_nabla_w[:]
            nabla_b = st_nabla_b[:]
            cost = 0
            for pos in range(len(data)):
                delta_w, delta_b, delta_cost = self.backpropagation(data[pos], responses[pos])
                nabla_w = [nw + dw for nw, dw  in zip(nabla_w, delta_w)]
                nabla_b = [nw + dw for nw, dw  in zip(nabla_b, delta_b)]
                cost += delta_cost 
             
                         
            self.weights = [ w - (nw*(lr/len(data))) for w, nw in zip(self.weights, nabla_w)]
            self.biases = [ b - (nb*(lr/len(data))) for b, nb in zip(self.biases, nabla_b)]

            print('=============================================')
            print(f'eppoch:{eppoch+1} \n cost:{cost/len(data)}')
            if self.classification:
                print(f' acc:{self.__hits/len(data)*100}%')
                
 

    def trainer_weights_mini_batch(self, data, responses, len_mini_batch=5,  eppochs=10, lr=0.1):
        st_nabla_w = [layer * 0 for layer in self.weights]
        st_nabla_b = [layer * 0 for layer in self.biases]

        batch_size = len(data) // len_mini_batch
        
        for eppoch in range(eppochs):
            self.__hits = 0
            nabla_w = st_nabla_w[:]
            nabla_b = st_nabla_b[:]
            cost = 0
            for batch in range(batch_size):
                for pos in range(len_mini_batch*batch, len_mini_batch*(batch+1)):
                    delta_w, delta_b, delta_cost = self.backpropagation(data[pos], responses[pos])
                    nabla_w = [nw + dw for nw, dw  in zip(nabla_w, delta_w)]
                    nabla_b = [nw + dw for nw, dw  in zip(nabla_b, delta_b)]
                    cost += delta_cost 
                
                
                self.weights = [ w - (nw*(lr/len(data))) for w, nw in zip(self.weights, nabla_w)]
                self.biases = [ b - (nb*(lr/len(data))) for b, nb in zip(self.biases, nabla_b)]

            print('=============================================')
            print(f'eppoch:{eppoch+1} \n cost:{cost/len(data)}')
            if self.classification:
                print(f' acc:{self.__hits/len(data)*100}%')
 

    @staticmethod
    def softmax(r, real):
        hit = 1
        lst = [e ** r[pos][0] for pos in range(len(r))]
        sum_ = sum(lst)

        lst =  [(e ** r[pos][0]) / sum_ for pos in range(len(r))]
        dv =   [[(e ** r[pos][0]) / sum_] for pos in range(len(r))]
        if dv[real][0] <= 1/len(dv):
            hit = 0
        cost = -log(dv[real][0])
        dv[real][0] -= 1

        dict_ = {
            'preview': Matrix([lst]),
            'derivate_cost': Matrix(dv),
            'cost': cost,
            'hit': hit
        }
        return dict_
    
    @staticmethod
    def linear_regression(r, real):
        preview = r[0].list
        response = real
        cost = (preview[0] - response)
        dict_ = {
            'preview': Matrix([preview]),
            'derivate_cost': Matrix([[(preview[0] - response)]]),
            'cost': cost ** 2
        }
        return dict_

