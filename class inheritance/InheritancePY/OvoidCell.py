from NeuronalCell import NeuronalCell


class OvoidCell(NeuronalCell):
    egg_shaped_form = bool

    def __init__(self):
        super(NeuronalCell, self).__init__()
        self.egg_shaped_form = 1

    def comprise_nadph(self):
        print("comprising 13% of NADPH" + str(self.comprise_nadph()))
