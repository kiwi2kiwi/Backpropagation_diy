from Neuron import *


input = [1.5,1,2,5,0.2]
target = [0.5,1,4,0.1]
input_layer = [input_Neuron(0),input_Neuron(1),input_Neuron(2)]
neuron_number = 3

weights = []

layer_layout = [[3],[4]]

layers = []
active_neurons = []
for idx, l in enumerate(layer_layout):
    layer = []
    for neur in l:
        n = Neuron(neuron_number)
        neuron_number += 1

        if idx == 0:
            for i in input_layer:
                n.parent_connections[i] = [i, 100, []]
        else:
            for i in layers[idx-1]:
                n.parent_connections[i] = [i, 100, []]
        layer.append(n)
    layers.append(layer)

for layer in layers:
    for n in layer:
        n.wire()



def error_function(pre,tar):
    return round((pre - tar)**2,3)

def deriv_error_function(pre,tar):
    return 2*(pre - tar)

def compute_error():
    for idx, inp in enumerate(input_layer):
        inp.set_input(input[idx])

    for idx, n in enumerate(layers[-1]):
        pred = n.activation()
        print("error: ", error_function(pred, target[idx]))

def backprop():
    compute_error()
    learning_rate = 0.1
    for idx, n in enumerate(layers[-1]):
        error_through_a_zero = deriv_error_function(n.activation(), target[idx])
        n.gradient_descent(error_through_a_zero, learning_rate)






backprop()
backprop()
backprop()
backprop()
backprop()
backprop()

print("end")