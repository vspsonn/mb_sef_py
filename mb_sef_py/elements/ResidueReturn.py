class ResidueReturn:
    def __init__(self, norm_forces=0., norm_constraints=0.):
        self.norm_forces = norm_forces
        self.norm_constraints = norm_constraints

    def __add__(self, other):
        norm_forces = self.norm_forces + other.norm_forces
        norm_constraints = self.norm_constraints + other.norm_constraints
        return ResidueReturn(norm_forces, norm_constraints)
