import numpy as np

from .Node import Node
from .TypeOfVariables import TypeOfVariables
from ..math import UnitQuaternion


class NodeSO3(Node):
    def __init__(self, R_0=UnitQuaternion(), name=None):
        Node.__init__(self, name)
        self.R_0 = R_0
        self.R = None

    @staticmethod
    def get_field():
        return TypeOfVariables.MOTION

    def get_number_of_motion_coordinates(self):
        return 4

    def get_motion_coordinates(self, configuration):
        R = self.R[configuration]
        return np.block([R.q.e0, R.q.e])

    def get_number_of_dofs(self):
        return 3

    def initialize(self, model):
        Node.initialize(self, model)
        self.R = [UnitQuaternion(ref_unit_quaternion=self.R_0),
                  UnitQuaternion(ref_unit_quaternion=self.R_0)]

    def kinematic_update(self, inc, previous_index, current_index):
        self.R[current_index] = self.R[previous_index] * UnitQuaternion.get_unitquat_from_parameters(inc)
