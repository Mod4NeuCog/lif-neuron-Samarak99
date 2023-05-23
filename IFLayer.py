# IntegrateAndFireNeuronLayer
from IntegrateAndFireNeuron import IntegrateAndFireNeuron


class IFLayer:
    def __init__(self, num_neurons, weights, thresholds):
        self.neurons = []
        self.activations_per_layer = []

        for i in range(num_neurons):
            #I added this if statement here because when we are creating the first layer we wil; not give it weights
            if weights:
                neuron = IntegrateAndFireNeuron(weights[i], thresholds[i])
            else:
                neuron = IntegrateAndFireNeuron(False, thresholds[i])

            neuron.calculate_activation()
            self.neurons.append(neuron)
            self.activations_per_layer.append(neuron.activation)

