import numpy as np

from .TypeOfVariables import TypeOfVariables
from .Node import Node


class NodeLagrangeMultipliers(Node):
    def __init__(self, number_of_multipliers, scaling, name=None):
        Node.__init__(self, name)
        self.number_of_multipliers = number_of_multipliers
        self.lambd = [np.zeros((number_of_multipliers, ))]*2
        self.scaling = scaling

    @staticmethod
    def get_field():
        return TypeOfVariables.LAGRANGE_MULTIPLIER

    def get_motion_coordinates(self, configuration):
        return self.lambd[configuration]

    def get_number_of_dofs(self):
        return self.number_of_multipliers

    def initialize(self, model):
        Node.initialize(self, model)
        self.lambd[0] *= 0.
        self.lambd[1] *= 0.

    def kinematic_update(self, inc, previous_index, current_index):
        self.lambd[current_index] = self.lambd[previous_index] + self.scaling * inc
