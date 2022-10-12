from NeuronalCell import NeuronalCell


class PyramidalCell(NeuronalCell):
    dendritic_spines = int

    def __init__(self):
        super(NeuronalCell, self).__init__()
        self.dendritic_spines = 1

    def transform_synaptic(self):
        print("Transforming Synaptic : " + str(self.dendritic_spines))


pc = PyramidalCell()
pc.transform_synaptic()
