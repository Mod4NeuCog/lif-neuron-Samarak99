class NeuronalCell:
    shape = ''
    length = float
    number_of_connections = int
    excitability = float

    def __init__(self):
        self.shape = 'oval'
        self.length = 0.0
        self.number_of_connections = 0
        self.excitability = 2.3

    def compute_potential(self):
        newVarTest = self.length + self.excitability
        print("Generated val : " + str(newVarTest))
