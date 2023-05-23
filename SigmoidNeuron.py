from Neuron import Neuron
import math


# The sigmoid function is used for mapping some probabilities
class SigmoidNeuron(Neuron):
    def __init__(self, weights):
        super().__init__(weights)
        self.activation = 0

    def calculate_activation(self):
        if self.weights:
            self.activation = 1.0 / (1.0 + math.exp(-sum(self.weights)))


def main():
    weights = [0.5, 0.3, -0.1]
    neuron = SigmoidNeuron(weights)
    neuron.calculate_activation()
    print(neuron.activation)


# if __name__ == "__main__":
#     main()