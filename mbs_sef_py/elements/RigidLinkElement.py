import numpy as np

from .Element import ElementWithConstraints
from .ElementProperties import ElementWithConstraintsProperties
from ..core.TypeOfVariables import TypeOfVariables
from ..math.SE3 import Frame


class RigidLinkProperties(ElementWithConstraintsProperties):
    def __init__(self):
        ElementWithConstraintsProperties.__init__(self)

    @staticmethod
    def get_element_type():
        return RigidLinkElement


class RigidLinkElement(ElementWithConstraints):
    def __init__(self, props, node_1, node_2):
        ElementWithConstraints.__init__(self, props)
        self.add_node(node_1)
        self.add_node(node_2)

        self.frame_0 = None
        self.inverse_adjoint_frame_0 = None

    def get_number_of_dofs(self):
        return 12

    @staticmethod
    def get_number_of_constraints():
        return 6

    def initialize(self, model):
        ElementWithConstraints.initialize(self, model)
        frame_A = self.list_nodes[TypeOfVariables.MOTION][0].frame_0
        frame_B = self.list_nodes[TypeOfVariables.MOTION][1].frame_0
        self.frame_0 = frame_A.get_inverse() * frame_B
        self.bt = np.block([self.frame_0.get_inverse_adjoint(), -np.eye(6)])

    def assemble_constraint_and_bt(self, model):
        frame_A = self.list_nodes[TypeOfVariables.MOTION][0].frame[model.current_configuration]
        frame_B = self.list_nodes[TypeOfVariables.MOTION][1].frame[model.current_configuration]
        self.constraint = Frame.get_parameters_from_frame(frame_B.get_inverse() * frame_A * self.frame_0)
