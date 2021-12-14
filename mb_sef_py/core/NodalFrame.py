import numpy as np

from .TypeOfVariables import TypeOfVariables
from .Node import Node
from ..math.SE3 import Frame


class NodalFrame(Node):
    def __init__(self, frame_ref=Frame(), name=None):
        Node.__init__(self, name)
        self.frame_ref = Frame(ref_frame=frame_ref)
        self.frame_0 = Frame(ref_frame=frame_ref)
        self.frame = None

    @staticmethod
    def get_field():
        return TypeOfVariables.MOTION

    def get_number_of_motion_coordinates(self):
        return 7

    def get_motion_coordinates(self, configuration):
        frame = self.frame[configuration]
        return np.block([frame.x, frame.q.e0, frame.q.e])

    def get_number_of_dofs(self):
        return 6

    def initialize(self, model):
        Node.initialize(self, model)
        self.frame = [Frame(ref_frame=self.frame_0), Frame(ref_frame=self.frame_0)]

    def set_frame_0(self, frame_0):
        self.frame_0 = Frame(ref_frame=frame_0)
        self.frame = [Frame(ref_frame=self.frame_0), Frame(ref_frame=self.frame_0)]

    def kinematic_update(self, inc, previous_index, current_index):
        self.frame[current_index] = self.frame[previous_index] * Frame.get_frame_from_parameters(inc)


class NodalRelativeFrame(NodalFrame):
    def __init__(self, A):
        NodalFrame.__init__(self)
        self.A = A

    @staticmethod
    def get_field():
        return TypeOfVariables.RELATIVE_MOTION

    def get_number_of_dofs(self):
        return self.A.shape[1]

    def kinematic_update(self, inc, previous_index, current_index):
        self.frame[current_index] = self.frame[previous_index] * Frame.get_frame_from_parameters(np.matmul(self.A, inc))
