import numpy as np

from .Element import ElementWithConstraints
from ..core.TypeOfVariables import TypeOfVariables
from ..core.NodalFrame import NodalRelativeFrame
from ..math.SE3 import Frame, breve6


class KinematicJointElement(ElementWithConstraints):
    def __init__(self, props, node_A, node_B):
        ElementWithConstraints.__init__(self, props)
        self.add_node(node_A)
        self.add_node(node_B)

    def get_number_of_dofs(self):
        return 12 + self.elem_props.get_number_of_relative_dof()

    @staticmethod
    def get_number_of_constraints():
        return 6

    def mesh(self, model):
        ElementWithConstraints.mesh(self, model)
        node_rel_dof = model.add_node(NodalRelativeFrame, self.elem_props.A)
        self.add_node(node_rel_dof)

    def initialize(self, model):
        ElementWithConstraints.initialize(self, model)
        frame_ref = self.list_nodes[TypeOfVariables.MOTION][0].frame_ref.get_inverse() * self.list_nodes[TypeOfVariables.MOTION][1].frame_ref
        node_rel_dof = self.list_nodes[TypeOfVariables.RELATIVE_MOTION][0]
        node_rel_dof.set_frame_0(frame_ref)

        n_rel_dof = self.elem_props.get_number_of_relative_dof()
        self.bt = np.zeros((6, 12+n_rel_dof))
        self.bt[:, 6:] = np.block([-np.eye(6), self.elem_props.A])
        self.bt[:, :6] = frame_ref.get_inverse_adjoint()

        ATA = np.matmul(np.transpose(self.elem_props.A), self.elem_props.A)
        node_rel_dof.v0 = - np.linalg.solve(ATA, np.matmul(np.transpose(self.elem_props.A),
                                                           np.matmul(self.bt[:, :12], model.v[self.loc_dof[:12]])))
        model.v[self.loc_dof[12:self.get_number_of_dofs()]] = node_rel_dof.v0

    def assemble_constraint_and_bt(self, model):
        frame_A = self.list_nodes[TypeOfVariables.MOTION][0].frame[model.current_configuration]
        frame_B = self.list_nodes[TypeOfVariables.MOTION][1].frame[model.current_configuration]
        frame_I = self.list_nodes[TypeOfVariables.RELATIVE_MOTION][0].frame[model.current_configuration]

        self.bt[:, :6] = frame_I.get_inverse_adjoint()
        self.constraint = Frame.get_parameters_from_frame(frame_B.get_inverse() * frame_A * frame_I)

    def assemble_bt_impl(self, model):
        frame_I = self.list_nodes[TypeOfVariables.RELATIVE_MOTION][0].frame[model.current_configuration]
        self.bt[:, :6] = frame_I.get_inverse_adjoint()
        return True

    def assemble_kt_impl(self, model):
        lambd = self.list_nodes[TypeOfVariables.LAGRANGE_MULTIPLIER][0].lambd[model.current_configuration]
        lambd_breve6 = breve6(0.5 * lambd)
        self.at = np.matmul(np.transpose(self.bt), np.matmul(lambd_breve6, self.bt))
        return True
