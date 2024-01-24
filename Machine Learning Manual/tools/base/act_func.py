from math import e
# bases
#===================================================================
def base_act_func(derivate):
    def activate(function):
        def value(r):
            if type(r) == float or type(r) == int:
                func_response = function(r)
                deri_response = derivate(func_response)
            
            else:
                func_response = [[]]
                deri_response = []
                for line in r:
                    func = []
                    deri = []

                    for col in line:
                        response = function(col)
                        func.append(response)
                        deri.append(derivate(response))

                    func_response[0].append(func[0])
                    deri_response.append(deri)
                        
            return {'func': func_response, 'derivate': deri_response}

        return value
    return activate
#===================================================================

# derivates
#===================================================================
def sigmoid_d(r):
    return r * (1 - r)

def tangh_d(r):
    return (1 - r ** 2)

def relu_d(r):
    if r > 0:
        return 1
    return 0
#===================================================================

# activates functions
#===================================================================
@base_act_func(sigmoid_d)
def sigmoid(r:float  or list):
    return 1 / (1 + e ** -r)

@base_act_func(tangh_d)
def tangh(r:float  or list):
    return (e ** r - e ** -r) / (e ** r + e ** -r)

@base_act_func(relu_d)
def relu(r:float or list):
    if r > 0:
        return r
    return 0


#===================================================================

dict_act_func = {
    'sigmoid': sigmoid,
    'tangh': tangh,
    'relu': relu,

}
