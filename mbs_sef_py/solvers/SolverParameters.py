class SolverParameters:
    def __init__(self):
        self.h = 1.e-2
        self.T = 1.
        self.nit_max = 10
        self.tol_res_forces = 1.e-6
        self.tol_res_constraints = 1.e-6


class TimeIntegrationParameters(SolverParameters):
    def __init__(self):
        SolverParameters.__init__(self)
        self.rho = 1.
