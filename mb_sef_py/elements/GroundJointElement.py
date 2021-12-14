import numpy as np

from .Element import ElementWithConstraints
from ..core.TypeOfVariables import TypeOfVariables
from ..core.NodalFrame import NodalRelativeFrame
from ..math.SE3 import Frame, breve6


class GroundJointElement(ElementWithConstraints):
    def __init__(self, props, node):
        ElementWithConstraints.__init__(self, props)
        self.add_node(node)

    def get_number_of_dofs(self):
        return 6 + self.elem_props.get_number_of_relative_dof()

    @staticmethod
    def get_number_of_constraints():
        return 6

    def mesh(self, model):
        ElementWithConstraints.mesh(self, model)
        node_rel_dof = model.add_node(NodalRelativeFrame, self.elem_props.A)
        self.add_node(node_rel_dof)

    def initialize(self, model):
        ElementWithConstraints.initialize(self, model)
        node_rel_dof = self.list_nodes[TypeOfVariables.RELATIVE_MOTION][0]
        node_rel_dof.set_frame_0(self.list_nodes[TypeOfVariables.MOTION][0].frame_ref)

        self.bt = np.block([-np.eye(6), self.elem_props.A])

        ATA = np.matmul(np.transpose(self.elem_props.A), self.elem_props.A)
        node_rel_dof.v0 = np.linalg.solve(ATA, np.matmul(np.transpose(self.elem_props.A),
                                                         model.v[self.loc_dof[:6]]))
        model.v[self.loc_dof[6:self.get_number_of_dofs()]] = node_rel_dof.v0

    def assemble_constraint_and_bt(self, model):
        frame_A = self.list_nodes[TypeOfVariables.MOTION][0].frame[model.current_configuration]
        frame_I = self.list_nodes[TypeOfVariables.RELATIVE_MOTION][0].frame[model.current_configuration]
        self.constraint = Frame.get_parameters_from_frame(frame_A.get_inverse() * frame_I)

    def assemble_kt_impl(self, model):
        lambd = self.list_nodes[TypeOfVariables.LAGRANGE_MULTIPLIER][0].lambd[model.current_configuration]
        lambd_breve6 = breve6(0.5 * lambd)
        self.at = np.matmul(np.transpose(self.bt), np.matmul(lambd_breve6, self.bt))
        return True
