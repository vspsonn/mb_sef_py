import numpy as np

from .Node import Node
from .TypeOfVariables import TypeOfVariables


class NodeRn(Node):
    def __init__(self, x0, name):
        Node.__init__(self, name)
        self.ndof = self.get_number_of_dofs()
        if x0 is None:
            self.x0 = np.zeros((self.ndof,))
        else:
            self.x0 = x0
        self.x = None

    @staticmethod
    def get_field():
        return TypeOfVariables.MOTION

    def get_motion_coordinates(self, configuration):
        return self.x[configuration]

    def initialize(self, model):
        Node.initialize(self, model)
        self.x = [self.x0[:], self.x0[:]]

    def kinematic_update(self, inc, previous_index, current_index):
        self.x[current_index] = self.x[previous_index] + inc


class NodeR3(NodeRn):
    def __init__(self, x0=Node, name=None):
        NodeRn.__init__(self, x0, name)

    def get_number_of_dofs(self):
        return 3
