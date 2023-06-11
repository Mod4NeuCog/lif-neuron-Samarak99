import time
import matplotlib.pyplot as plt

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
        self.start_time = time.time() # track how many time it took for the whole network
        self.end_time = None
        self.duration = None
        self.fired_neurons_count = 0
        self.activation_input = activation_input
        self.initially_active_neuron = sum(neuron == 1 for neuron in self.activation_input)
        self.activity = [] # we will save the number of neurons saved at each time

    # this function get the output of the neurons in the last array, and the one with higher output wins
    def winner_take_all(self):
        print('total')
        print(self.activity)

        for i in range(0 , self.num_layers):
            #Create the first layer but without appending the activations because this is intiliaze at the beginning
            if i == 0:
                num_fired_neurons = self.initially_active_neuron
                ## if there is a threshold it means I want to create an integrate and fire Layer not sigmoid
                if len(self.threshold) > 0:
                    layer = IFLayer(self.neurons_per_layer[i], False, self.threshold[i] , self.initially_active_neuron , self.start_time)
                else:
                    layer = SigmoidNeuronLayer(self.neurons_per_layer[i], False)
                self.track_activity(num_fired_neurons)
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
                    layer = IFLayer(self.neurons_per_layer[i], weights_per_layer, self.threshold[i] , self.fired_neurons_count, self.start_time)
                else:
                    layer = SigmoidNeuronLayer(self.neurons_per_layer[i], weights_per_layer)

                num_fired_neurons = layer.fired_neurons_count

                # append the activations in the layer in the matrix of the network :
                self.activations.append(layer.activations_per_layer)

                # append the activty of the layer to the network
                self.activity = self.activity + layer.activity


            # for each layer add how many neurons fired
            self.fired_neurons_count = num_fired_neurons

            # Append each layer to the layers vector - This is just to store everything
            self.layers.append(layer)

        ## printing activation MAtrix
        print('This Is our activation Matrix')
        print(self.activations)
        print('//////////////////////')

        # Now the  network is done we save last ending time
        self.end_time = time.time()

        # get last layer index and get max value index inside it
        last_activation_layer = self.activations[self.num_layers - 1]
        max_value_index = last_activation_layer.index(max(last_activation_layer))
        return max_value_index

    def get_duration(self):
        if self.end_time is None:
            return None

        return self.end_time - self.start_time

    def track_activity(self , fired_nb):
        duration = time.time() - self.start_time
        self.activity.append((duration, fired_nb))

    def plot_activity(self , index):
        durations = [t[0] for t in self.activity]
        spikes = [t[1] for t in self.activity]

        plt.plot(durations, spikes)
        plt.xlabel("Time (s)")
        plt.ylabel("Total Activity")
        title ="Total Activity of the Network over Time , Net: "+ str(index)
        plt.title(title)
        plt.show()


def main():

    # My parameters are the neurons per layers, how many layers, the weights , the threshold, the activation input.
    # So in order to test my model activity and duration wrt to parameters, I am gonna create other values. and then compare it.
    # For simplicity, going create a second network only with 2 neurons in 2 layers
    neurons_per_layer_array = [[2, 3, 4], [2, 2]]
    weights_array = []
    activation_input_array = [[0, 1], [1, 1]]
    threshold_array = []


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
    weights_array.append(weights)

    weights_net2 = []
    weights_layer12 = [
        [0.5, 6],
        [2, 3]
    ]
    weights_net2.append(weights_layer12)
    weights_array.append(weights_net2)

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
    threshold_array.append(threshold)

    threshold_net2 = [
        [1, 1],
        [2, 10]]
    threshold_array.append(threshold_net2)

    ## Now to test the network with different parameters, we loop according to our combination of parameters
    for i in range(len(neurons_per_layer_array)):
        network = Network(neurons_per_layer_array[i], weights_array[i], activation_input_array[i], threshold_array[i])
        winner_neuron = network.winner_take_all()
        print('Netowrk : ', i+1)
        print("Neuron " + str(winner_neuron + 1) + " in the last layer (" + str(
            len(neurons_per_layer)) + ") is the Winner")

        print('//////////////////////')
        print('This Is How many neuron fired')
        print(network.fired_neurons_count)

        print('//////////////////////')
        print('This Is the activty of the network : (time, fired_neuron_count)')
        network.plot_activity(i+1)



if __name__ == "__main__":
    main()


## Fulfilling all layers

# Layer 0 :
# sending a vector of inputs. I expect as output same a vector (collecting O1... Os get 1 or 0 vectors)
# my variables a vector of boolean vectors

# Layer 1 & 2:
# we have the notion of time : at everytime step I have an input and output. Different notion of time. one is logical it takes every layer
# when we want to compare to brain we don't have a series a computation. So to map the series of computation I would expect that the output is not exactly at the first time.
# model is the raw data, the output of this feed forward. it is a vector
# behavior is the relation one input I can have 2 different outputs. Function should be that now I know that I have 2 diff states, there should be a difference according to the initial state

# Layer 3 : State Transition
# One State you get a vector of variable . you define a state as a vector of values of these variables
# What is the variable of each neuron. for each neuron we have a membrane potential as variables.
# On a network level we get an output or not

# Layer 4 : Network Level
# connectivity : which neurons are connected .How they are connected
# each of a neuron is a system, and how they are connected can be according to the index/name.
# So 2 things : components and how they are connected.

# EF: Experimental Framework
# mouse : active duration : is it matching the activity duration of the network
# what to want to observe as an output of the network and EF:
# If the action time fits the duration of activty of the network. So we want to the output to be yes or no (acceptor result).
# and now the EF : has many components, and is connected to each other. we have a generator and can be embedded to the acceptor or no. we have the raw input.
# Input, translate transuse, then the acceptor is comparing.
# agreement with you experiment frame to your model - so at some point we can have multiple classes, models ...

# input output system we call it a component

