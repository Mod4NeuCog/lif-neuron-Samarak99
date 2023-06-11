# IntegrateAndFireNeuronLayer
from IntegrateAndFireNeuron import IntegrateAndFireNeuron
import time


class IFLayer:
    def __init__(self, num_neurons, weights, thresholds , previously_fired_neurons , network_start_time):
        self.neurons = []
        self.activations_per_layer = []
        self.fired_neurons_count = previously_fired_neurons
        self.activity = []  # we will save the number of neurons saved at each time
        self.network_start_time = network_start_time


        for i in range(num_neurons):
            #I added this if statement here because when we are creating the first layer we wil; not give it weights
            if weights:
                neuron = IntegrateAndFireNeuron(weights[i], thresholds[i])
            else:
                neuron = IntegrateAndFireNeuron(False, thresholds[i])

            neuron.calculate_activation()
            if neuron.activation == 1:
                self.fired_neurons_count = self.fired_neurons_count + 1
                self.track_activity()

            self.neurons.append(neuron)
            self.activations_per_layer.append(neuron.activation)

    def track_activity(self):
        duration = time.time() - self.network_start_time
        self.activity.append((duration, self.fired_neurons_count))

