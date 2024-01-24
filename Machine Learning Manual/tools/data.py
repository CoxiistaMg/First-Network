from csv import reader
from .base.matrix import Matrix, Array
from random import shuffle as sh
from collections import Counter

DATA_FILE = 'data/'
def get_last_element(element):
    return element[-1]


class DataSet:
    def __init__(self, name, trainer_rate=0.5, classification=True, shuffle=True) -> None:
        self.name = name
        with open(DATA_FILE+name, 'r') as file:
            self.data = list(reader(file))[1:]
            self.data.sort(key=get_last_element)
            self.responses = [response[-1] for response in self.data] 

            counter = Counter(self.responses)
            value_min = counter[min(counter)]

            self.types_responses = tuple(counter.keys())
            
            for each in self.data:
                each[-1] = self.types_responses.index(each[-1])
            
            self.data = [[float(element) for element in lst] for lst in self.data]
            self.len_each_value = int((value_min * len(self.types_responses) * trainer_rate) // len(self.types_responses))

            for each in counter:
                counter[each] = self.responses.index(each)

            self.trainer = []
            for pos in range(self.len_each_value):
                for response in self.types_responses:
                    index = counter[response] + pos
                    self.trainer.append(self.data[index][:])

            self.trainer_results = [int(element.pop()) for element in self.trainer]
            self.results = tuple([int(element.pop())]for element in self.data)
            self.data = tuple(Matrix([element[:-1]]).t() for element in self.data)
            self.trainer =  [Matrix([element]).t() for element in self.trainer]
            self.len_results = len(self.types_responses)
            self.len_inputs = self.trainer[0].i
    
