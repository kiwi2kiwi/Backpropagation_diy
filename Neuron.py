import numpy as np
import random

class Neuron():
    def __init__(self, name):
        super(Neuron, self).__init__()
        self.parent_connections = {} # closer to input
        self.children_connections = {} # closer to output
        self.output = 0
        self.activated = False
        self.name = name
        print("Hey, im a neuron!")

    def reset_neuron(self):
        self.output = 0
        self.activated = False
        for p in self.parent_connections.keys():
            self.parent_connections[p][2] = []
        for c in self.children_connections.keys():
            self.children_connections[c][2] = []


    def wire(self):
        weight = round(random.uniform(0, 1), 2)
        #weight = 0.8
        for p in self.parent_connections.keys():
            parent_connection = self.parent_connections[p]
            parent = parent_connection[0]
            parent.children_connections[self.__hash__()] = [self, weight, []]
            self.parent_connections[p] = [parent, weight, []]


    def change_weight(self):
        for p in self.parent_connections.keys():
            parent_connection = self.parent_connections[p]
            parent = parent_connection[0]
            new_weight_to_parent = parent_connection[2]
            #new_weight = parent_connection[1] + sum(new_weight_to_parent)
            parent.children_connections[self.__hash__()] = [self, new_weight_to_parent[0], new_weight_to_parent]
            self.parent_connections[p] = [parent, new_weight_to_parent[0], new_weight_to_parent]


    def get_weight(self, parent):
        return self.parent_connections[parent][1]

    def gradient_descent(self, bis_hier, learning_rate):

        ab_hier = self.a_null_a_eins() * bis_hier
        for p in self.parent_connections.keys():
            parent_connection = self.parent_connections[p]
            self.parent_connections[p][0].gradient_descent(ab_hier, learning_rate)

            error_durch_w = self.a_null_w_parent(parent_connection[0]) * bis_hier

            self.parent_connections[p][2].append(self.get_weight(p) - (learning_rate * error_durch_w))

            print("weight: ", round(parent_connection[1],2), " adjust: ", round(parent_connection[2][0] - self.get_weight(p),2))
        self.change_weight()
        self.reset_neuron()


    def activation_function(self, z):
        return z

    #        return 1. / (1 + np.exp(-z))

    def deri_activation_function(self, z):
        return z

    def activation(self):
        if self.activated:
            return self.output

        summation = 0
        for p in self.parent_connections.keys():
            parent_connection = self.parent_connections[p]
            summation += parent_connection[0].activation() * parent_connection[1]
        self.output = self.activation_function(summation)
        self.activated = True
        return self.output

    def deri_activation(self):
        summation = 0
        for p in self.parent_connections.keys():
            summation += self.parent_connections[p].deri_activation()
        return self.deri_activation_function(summation)

    def a_null_w_parent(self, parent):
        return self.deri_activation_function(parent.activation())

    def a_null_a_eins(self):
        summation = 0
        for p in self.parent_connections.keys():
            parent_connection = self.parent_connections[p]
            summation += parent_connection[1]
        return self.deri_activation_function(summation)

    def __hash__(self):
        return self.name


class input_Neuron():
    def __init__(self, name):
        super(input_Neuron, self).__init__()
        self.children_connections = {} # closer to output
        self.output = 0
        self.activated = False
        self.name = name

    def gradient_descent(self, a, b):
        pass

    def set_input(self, input):
        self.activated = True
        self.output = input

    def activation_function(z):
        return 1. / (1 + np.exp(-z))

    def activation(self):
        return self.output

    def deri_activation(self):
        return self.output

    def reset_neuron(self):
        self.output = 0
        self.activated = False

    def __hash__(self):
        return self.name