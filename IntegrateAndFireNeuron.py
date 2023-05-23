from Neuron import Neuron


class IntegrateAndFireNeuron(Neuron):
    # By default, the Threshold is 1.0
    def __init__(self, weights, threshold=1.0):
        super().__init__(weights)
        self.threshold = threshold
        self.activation = 0

    def calculate_activation(self):
        if self.weights:
            if sum(self.weights) >= self.threshold:
                self.activation = 1


# def main():
#     weights = [0.5, 0.4, -0.1]
#     neuron = IntegrateAndFireNeuron(weights)
#     neuron.calculate_activation()
#     print(neuron.activation)


# if __name__ == "__main__":
#     main()
