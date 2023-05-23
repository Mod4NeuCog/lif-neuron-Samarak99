from SigmoidNeuron import SigmoidNeuron
from SigmoidNeuronLayer import SigmoidNeuronLayer
from IFLayer import IFLayer


class Network:
    def __init__(self, neurons_per_layer, weights, activation_input, threshold=None):
        if threshold is None:
            threshold = []
        self.layers = []
        self.num_layers = len(neurons_per_layer)
        # initially a matrix
        self.weights = weights
        self.activations = [activation_input]
        # initially a vector
        self.neurons_per_layer = neurons_per_layer
        self.layers = []
        self.threshold = threshold

    # this function get the output of the neurons in the last array, and the one with higher output wins
    def winner_take_all(self):
        for i in range(0 , self.num_layers):
            #Create the first layer but without appending the activations because this is intiliaze at the beginning
            if i == 0:
                ## if there is a threshold it means I want to create an integrate and fire Layer not sigmoid
                if len(self.threshold) > 0:
                    layer = IFLayer(self.neurons_per_layer[i], False, self.threshold[i])
                else:
                    layer = SigmoidNeuronLayer(self.neurons_per_layer[i], False)
            else:
                weights_per_layer = []
                # Step 1 : if not first layer I need to get the updated matrix of weights to Pass to layer
                # Here in the second layer j will be 0,1,2 because we have 3 neurons
                for j in range(self.neurons_per_layer[i]):
                    # for each neuron in this layer need to get the weight through
                    # first get index of previous layer and the number of neuron in previous one
                    previous_layer_neuron_number = self.neurons_per_layer[i - 1]

                    current_neuron_weights = []
                    # now for this neuron get the weights through looping into the weights matrix
                    for x in range(previous_layer_neuron_number):
                        indexed_weight = self.weights[i - 1][x][j]
                        current_neuron_weights.append(indexed_weight)

                    # now that we have indexed weight for this neuron, we multiply it by the previous activation inputs
                    neuron_inputs = [inputs * weight for inputs, weight in
                                     zip(self.activations[i - 1], current_neuron_weights)]

                    # for each neuron we add the weight and pass it to the layers
                    weights_per_layer.append(neuron_inputs)

                # Step 2 : Create the layer and pass the updated weights
                ## if there is a threshold it means I want to create an integrate and fire Layer not sigmoid
                if len(self.threshold) >0:
                    layer = IFLayer(self.neurons_per_layer[i], weights_per_layer, self.threshold[i])
                else:
                    layer = SigmoidNeuronLayer(self.neurons_per_layer[i], weights_per_layer)

                # append the activations in the layer in the matrix of the network :
                self.activations.append(layer.activations_per_layer)

            # Append each layer to the layers vector - This is just to store everything
            self.layers.append(layer)

        ## printing activation MAtrix
        print('//////////////////////')
        print(self.activations)
        print('//////////////////////')

        # get last layer index and get max value index inside it
        last_activation_layer = self.activations[self.num_layers - 1]
        max_value_index = last_activation_layer.index(max(last_activation_layer))
        return max_value_index


def main():
    # how many layers and how many neuron in each layer
    neurons_per_layer = [2, 3, 4]

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

    # Inputs here: I initialize the first input which will be for me here vector (y1, y2) and can be considered as
    # the pixels at init, so value between 0-1
    # activation_input = [0.4, 0.7] # this is for a Sigmoid Network
    activation_input = [0, 1] # this is for an IF Network

    ## IF Network
    #If I want to create an Integrate and Fire instead of a sigmoid Function I need to add the threshold matrix
    # according to the neurons_per_layer = [2, 3, 4]
    threshold =[
        [1, 1.2],
        [1.2, 1.2, 1.0],
        [10, 1.5, 1.6, 1.2]]


    network = Network(neurons_per_layer, weights , activation_input , threshold)
    winner_neuron = network.winner_take_all()
    print("Neuron " + str(winner_neuron+1) + " in the last layer ("+str(len(neurons_per_layer))+") is the Winner")


if __name__ == "__main__":
    main()
