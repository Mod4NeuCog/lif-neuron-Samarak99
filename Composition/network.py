# 1) Create a mother Netwrok. with neurons 1000. do a function inside loop to update membre potential

class Neuron:

    #membrane_potential: int

    def __init__(self):
        self.membrane_potential = 1

    def update_membrane_pot(self):
        self.membrane_potential += 1


class NeuronNetwork:

    def __init__(self):
        self.neurons = []

        for t in range(0,100):
            n = Neuron()
            self.neurons.append(n)

    def big_update(self):
        for i in range(0, 100):
            self.neurons[i].update_membrane_pot()
            print(self.neurons[i].membrane_potential)


def main():

    main_network = NeuronNetwork()
    main_network.big_update()


main()


