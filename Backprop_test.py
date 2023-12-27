from Neuron import *
import numpy as np

from sklearn import datasets
iris = datasets.load_iris()
from sklearn.utils import shuffle


input = [1.5]
target = [0.5]
input_layer = [input_Neuron(0)]
neuron_number = 4

weights = []

layer_layout = [[1]]



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
        active_neurons.append(n)
    layers.append(layer)

for layer in layers:
    for n in layer:
        n.wire()



def error_function(pre,tar):
    return round((pre - tar)**2,3)

def deriv_error_function(pre,tar):
    return 2*(pre - tar)

def compute_error(target):
    for idx, n in enumerate(layers[-1]):
        pred = n.activation()
        print("error: ", error_function(pred, target[idx]))

def predict(slice_of_data):
    for idx, inp in enumerate(input_layer):
        inp.set_input(slice_of_data[idx])
    prediction = []
    for i in layers[-1]:
        prediction.append(i.activation())
    return prediction


def backprop(target):
    compute_error(target)
    learning_rate = 0.01
    for idx, n in enumerate(layers[-1]):
        error_through_a_zero = deriv_error_function(n.activation(), target[idx])
        n.gradient_descent(error_through_a_zero, learning_rate)


def train(data):

    for ds in data:
        predict(ds[:-1])
        backprop([ds[-1]])

def run():
    input_layer[0].set_input(input[0])


    compute_error(target)
    learning_rate = 0.1
    for idx, n in enumerate(layers[-1]):
        error_through_a_zero = deriv_error_function(n.activation(), target[idx])
        n.gradient_descent(error_through_a_zero, learning_rate)
    for a in active_neurons:
        a.reset_neuron()

run()
run()
run()
run()
run()
run()
run()
run()
run()





print("end")