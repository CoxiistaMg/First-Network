class Array:
    cursor = 0
    def __init__(self, list: list) -> None:
        self.list = list

    def __str__(self):
        return f'Array: {self.list}'
    
    def __mul__(self, other):
        if type(self) == type(other):
            return Array([each * other for each, other in zip(self.list, other.list)])
        
        return Array([element * other for element in self.list])

    def __add__(self, other):
        if type(self) == type(other):
            return Array([each + other for each, other in zip(self.list, other.list)])
            
        return Array([element + other for element in self.list])

    def __sub__(self, other):
        if type(self) == type(other):
            return Array([each - other for each, other in zip(self.list, other.list)])
            
        return Array([element - other for element in self.list])

    def __iter__(self):
        return iter(self.list)

    def __next__(self):
         if self.cursor > len(self.list)-1: 

            raise StopIteration 
         
         self.cursor += 1
         return self.list[self.cursor - 1]

    def __getitem__(self, item):
         return self.list[item]
    
    def __repr__(self):
        return (f'\nArray({self.list})')
    
    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)


class Matrix:
    def __init__(self, matrix: list) -> None:
        self.matrix = [Array(line) for line in matrix]
        self._i = range(len(matrix))
        self._j = range(len(matrix[0]))
        self.i = len(self._i)
        self.j = len(self._j)
    
    def __mul__(self, other):
        matrix = []
        if type(self) == type(other):
            other = other.t()

            for line1 in self.matrix:
                matrix.append([])
                for line2 in other.matrix:
                    matrix[-1].append(sum(line1 * line2))

        else:
            for line in self.matrix:
                value = line * other
                matrix.append(value.list)

        return Matrix(matrix)

    def __add__(self, other):
        matrix = []
        if type(self) == type(other):
            for line1, line2 in zip(self.matrix, other.matrix):
                matrix.append((line1 + line2).list)
            return Matrix(matrix)

        else:
            for line in self.matrix:
                value = line + other
                matrix.append(value.list)
        
        return Matrix(matrix)
    
    def __sub__(self, other):
        matrix = []
        if type(self) == type(other):
            for line1, line2 in zip(self.matrix, other.matrix):
                matrix.append((line1 - line2).list)
            return Matrix(matrix)

        else:
            for line in self.matrix:
                value = line - other
                matrix.append(value.list)

    def __str__(self) -> str:
        return f'Matrix {self.i}x{self.j}: {self.matrix}'

    def __repr__(self) -> str:
        return f'\nMatrix {self.i}x{self.j}: {self.matrix}\n'
    
    def t(self):
        matrix = [
            [ self.matrix[i][j] for i in self._i] for j in self._j
        ]

        return Matrix(matrix)

    def apply_function(self, function):
        if self.j != 1:
            raise 'Deve ser aplicada apenas em matrizes que possuem uma coluna 1'
        self.matrix = [[function(each[0])]for each in self.matrix]
        

if __name__ == '__main__':
    mt = Matrix([[4, 2, 6], [5, 3, 7]])
    mt1 = Matrix([[4, 2, 6], [5, 3, 7], [3, 2, 1]])