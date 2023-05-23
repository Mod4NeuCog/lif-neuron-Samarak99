from SigmoidNeuron import SigmoidNeuron


class SigmoidNeuronLayer:
    def __init__(self, num_neurons, weights):
        self.neurons = []
        self.activations_per_layer = []

        for i in range(num_neurons):

            # I added this if statement here because when we are creating the first layer we wil; not give it weights
            if weights:
                neuron = SigmoidNeuron(weights[i])
            else:
                neuron = SigmoidNeuron(False)

            # Calculate the activations per neuron
            neuron.calculate_activation()
            self.neurons.append(neuron) # append the neuron
            self.activations_per_layer.append(neuron.activation) # append the activation
