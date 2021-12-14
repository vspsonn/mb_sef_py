import numpy as np

from .Element import ElementWithConstraints
from .ElementProperties import ElementWithConstraintsProperties
from ..core.TypeOfVariables import TypeOfVariables
from ..math.SE3 import Frame, breve6


class ClampedFrameProperties(ElementWithConstraintsProperties):
    def __init__(self):
        ElementWithConstraintsProperties.__init__(self)
        pass

    @staticmethod
    def get_element_type():
        return ClampedFrame


class ClampedFrame(ElementWithConstraints):
    def __init__(self, props, node):
        ElementWithConstraints.__init__(self, props)
        self.add_node(node)
        self.bt = np.eye(6)
        self.inverse_frame_ref = None

    def get_number_of_dofs(self):
        return 6

    @staticmethod
    def get_number_of_constraints():
        return 6

    def initialize(self, model):
        ElementWithConstraints.initialize(self, model)
        frame_ref = self.list_nodes[TypeOfVariables.MOTION][0].frame_ref
        self.inverse_frame_ref = frame_ref.get_inverse()

    def assemble_constraint_and_bt(self, model):
        frame_A = self.list_nodes[TypeOfVariables.MOTION][0].frame[model.current_configuration]
        self.constraint = Frame.get_parameters_from_frame(self.inverse_frame_ref * frame_A)

    def assemble_kt_impl(self, model):
        lambd = self.list_nodes[TypeOfVariables.LAGRANGE_MULTIPLIER][0].lambd[model.current_configuration]
        self.at = breve6(0.5 * lambd)
        return True
