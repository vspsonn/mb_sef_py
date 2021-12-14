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
    def __init__(self, props, node_A, node_B):
        ElementWithConstraints.__init__(self, props)
        self.add_node(node_A)
        self.add_node(node_B)

        self.frame_ref = None
        self.inverse_adjoint_frame_ref = None

    def get_number_of_dofs(self):
        return 12

    @staticmethod
    def get_number_of_constraints():
        return 6

    def initialize(self, model):
        ElementWithConstraints.initialize(self, model)
        frame_A = self.list_nodes[TypeOfVariables.MOTION][0].frame_ref
        frame_B = self.list_nodes[TypeOfVariables.MOTION][1].frame_ref
        self.frame_ref = frame_A.get_inverse() * frame_B
        self.bt = np.block([self.frame_ref.get_inverse_adjoint(), -np.eye(6)])

    def assemble_constraint_and_bt(self, model):
        frame_A = self.list_nodes[TypeOfVariables.MOTION][0].frame[model.current_configuration]
        frame_B = self.list_nodes[TypeOfVariables.MOTION][1].frame[model.current_configuration]
        self.constraint = Frame.get_parameters_from_frame(frame_B.get_inverse() * frame_A * self.frame_ref)
