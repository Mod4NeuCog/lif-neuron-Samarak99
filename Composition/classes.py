class NeuronClass:

    def __init__(self):
        self.synapses = []
        self.axons = 1
        self.dendrites = []
        self.nucleus = 1

        # composition
        for t in range(1, 6):
            n = Synapse(t).electrical_signal
            self.synapses.append(n)

        for t in range(1, 6):
            d = Dendrite()
            self.dendrites.append(d)

    def compute_transition(self):
        print("there's a propagation")
            #print("there's a propagation")



class Synapse:

    def __init__(self, index):
        self.electrical_signal = index


class Dendrite:

    def __init__(self):
        self.electrical_axons_signal = False
