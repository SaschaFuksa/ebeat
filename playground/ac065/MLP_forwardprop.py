# video6 von valerio velardo, implementieren eines neuralen netzwerks
import numpy as np


class MLP:

    def __init__(self, num_inputs=3, num_hidden=[3, 5], num_outputs=2):
        # konstruktur mit 3 verschiedenen attributen,[3,5] bedeutet das wir ein neurales netz mit zwei hidden layers haben
        # das erste hidden layer hat 3 neuronen und das zweite hidden layer 5, bei outputs bedeutet die 2 = 2 neuronen.
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]
        # initate random weights
        self.weights = []
        # kreieren einer Matrix mit der methode np.random., die methode kreiert random arrays
        # diese arrays können verschiedene dimensionen haben, diese zwei dimensionen werden mit den werten (layers[i] und layers[i+1]
        # 2 d array ( Matrix) = number of rows (layers[i]
        # number of columns layers[i+1] also nummer von neuronen im subsequent layer
        for i in range(len(layers) - 1):
            w = np.random.rand (layers[i], layers[i + 1])
            self.weights.append(w)

    def forward_propagate(self, inputs):
        # erst kommt net input und dann die aktivation, das wird hier klargestellt
        activations = inputs

        for w in self.weights:
            # calculate the net inputs
            net_inputs = np.dot(activations, w)
            # calculate the activations
            activations = self._sigmoid(net_inputs)  # die net inputs werden in die sigmoid avtivation funktion gegeben

        return activations

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    # create an MLP
    mlp = MLP()
    # create some inputs
    inputs = np.random.rand(mlp.num_inputs)
    # inputs sollte vektor haben, der die same number of items hat as the number of neurons in the input layer
    # perform forward prop
    outputs = mlp.forward_propagate(inputs)
    # print the results
    print("The Network Input is: {}".format(inputs))
    print("The Network Output is: {}".format(outputs))
