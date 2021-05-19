class ResidueReturn:
    def __init__(self, norm_forces=0., norm_constraints=0., mechanical_power=0.):
        self.norm_forces = norm_forces
        self.norm_constraints = norm_constraints
        self.mechanical_power = mechanical_power

    def __add__(self, other):
        norm_forces = self.norm_forces + other.norm_forces
        norm_constraints = self.norm_constraints + other.norm_constraints
        mechanical_power = self.mechanical_power + other.mechanical_power
        return ResidueReturn(norm_forces, norm_constraints, mechanical_power)
