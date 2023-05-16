from SigmoidNeuron import SigmoidNeuron


class SigmoidNeuronLayer:
    def __init__(self, num_neurons, weights):
        self.neurons = []
        for i in range(num_neurons):
            neuron = SigmoidNeuron(weights[i])
            self.neurons.append(neuron)
