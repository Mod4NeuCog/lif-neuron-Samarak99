from Neuron import Neuron


class IntegrateAndFireNeuron(Neuron):
    def __init__(self, weights, threshold=1.0):
        super().__init__(weights)
        self.threshold = threshold

    def calculate_output(self):
        if sum(self.weights) >= self.threshold:
            return 1.0
        else:
            return 0.0


def main():
    weights = [0.5, 0.4, -0.1]
    neuron = IntegrateAndFireNeuron(weights)
    output = neuron.calculate_output()
    print(output)


if __name__ == "__main__":
    main()
