from SigmoidNeuron import SigmoidNeuron


class Network:
    def __init__(self, neurons_per_layer, weights, activation_input):
        self.layers = []
        self.num_layers = len(neurons_per_layer)
        # initially a matrix
        self.weights = weights
        self.activations = [activation_input]
        # initially a vector
        self.neurons_per_layer = neurons_per_layer

    # this function get the output of the neurons in the last array, and the one with higher output wins
    def winner_take_all(self):
        # I start in the second layer because the inputs of  the first layers are already initialize
        for i in range(1, self.num_layers):
            layer_activations = []
            # Here in the second layer j will be 0,1,2 because we have 3 neurons
            for j in range(self.neurons_per_layer[i]):
                # for each neuron in this layer need to get the weight through
                # first get index of previous layer and the number of neuron in previous one
                previous_layer_neuron_number = self.neurons_per_layer[i-1]

                current_neuron_weights = []
                # now for this neuron get the weights through looping into the weights matrix
                for x in range(previous_layer_neuron_number):
                    indexed_weight = self.weights[i-1][x][j]
                    current_neuron_weights.append(indexed_weight)

                # now that we have indexed weight for this neuron, we multiply it by the previous activation inputs
                neuron_inputs = [inputs * weight for inputs, weight in zip(self.activations[i - 1], current_neuron_weights)]
                sigmoid_neuron = SigmoidNeuron(neuron_inputs)
                sigmoid_neuron.calculate_activation()
                layer_activations.append(sigmoid_neuron.activation)

            self.activations.append(layer_activations)

        print('//////////////////////////////////////')
        print('The Activation Matrix is the following')
        print(self.activations)
        print('//////////////////////////////////////')

        last_activation_layer = self.activations[self.num_layers - 1]
        max_value_index = last_activation_layer.index(max(last_activation_layer))
        return max_value_index


def main():
    # how many layers and how many neuron in each layer
    neurons_per_layer = [2, 3, 4, 5]

    # there are the weights, and it is a matrix with connections, so I have in each row the origin and the columns are
    # the targets : w_11, w_12, w_13, w_21, w_22, w_23
    # rows are origin , number of previous neurons in layer, and columns are number of target number of neuron in layer
    weights = []
    weights_layer12 = [
        [1, 3, 4],
        [2, 6, 7]
    ]
    weights.append(weights_layer12)

    weights_layer23 = [
        [2, 2, 3, 5],
        [3, 2, 1, 1],
        [4, 3, 2, 1]
    ]
    weights.append(weights_layer23)

    weights_layer34 = [
        [2, 2, 3, 5, 6],
        [3, 2, 1, 1, 7],
        [4, 3, 2, 1, 8],
        [4, 3, 2, 1, 8]
    ]
    weights.append(weights_layer34)

    # Inputs here: I initialize the first input which will be for me here vector (y1, y2) and can be considered as
    # the pixels at init, so value between 0-1
    activation_input = [0.4, 0.7]

    network = Network(neurons_per_layer, weights , activation_input)
    winner_neuron = network.winner_take_all()
    print("Neuron " + str(winner_neuron+1) + " in the last layer ("+str(len(neurons_per_layer))+") is the Winner")


if __name__ == "__main__":
    main()
