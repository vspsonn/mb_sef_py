import numpy as np

from .Element import ElementWithConstraints
from .ElementProperties import ElementWithConstraintsProperties
from ..core.TypeOfVariables import TypeOfVariables
from ..math.SE3 import Frame


class ServoConstraintProperties(ElementWithConstraintsProperties):
    def __init__(self, imposed_displacement):
        ElementWithConstraintsProperties.__init__(self)
        self.imposed_displacement = imposed_displacement

    @staticmethod
    def get_element_type():
        return ServoConstraintElement


class ServoConstraintElement(ElementWithConstraints):
    def __init__(self, props, element_number):
        ElementWithConstraints.__init__(self, props)
        self.element_number = element_number

        self.previous_time = None
        self.previous_frame = None
        self.current_frame = None

    def get_number_of_dofs(self):
        return 1

    @staticmethod
    def get_number_of_constraints():
        return 1

    def mesh(self, model):
        ElementWithConstraints.mesh(self, model)
        element = model.list_elements[self.element_number]
        node_rel_dof = element.list_nodes[TypeOfVariables.RELATIVE_MOTION][0]
        self.add_node(node_rel_dof)

    def initialize(self, model):
        ElementWithConstraints.initialize(self, model)
        node_rel_dof = self.list_nodes[TypeOfVariables.RELATIVE_MOTION][0]
        self.bt = np.matmul(np.transpose(node_rel_dof.A), node_rel_dof.A)

        self.previous_frame = node_rel_dof.frame_0
        self.current_frame = self.previous_frame
        self.previous_time = model.time

    def assemble_constraint_and_bt(self, model):
        node_rel_dof = self.list_nodes[TypeOfVariables.RELATIVE_MOTION][0]

        if model.time > self.previous_time:
            self.previous_frame = self.current_frame

            imposed_displacement = self.elem_props.imposed_displacement(model.time)
            if model.time > 0.:
                imposed_displacement -= self.elem_props.imposed_displacement(self.previous_time)
            imposed_frame = Frame.get_frame_from_parameters(node_rel_dof.A[:, 0] * imposed_displacement)
            self.current_frame = self.previous_frame * imposed_frame

            self.previous_time = model.time

        relative_frame = self.current_frame.get_inverse() * node_rel_dof.frame[model.current_configuration]
        self.constraint = np.matmul(np.transpose(node_rel_dof.A), Frame.get_parameters_from_frame(relative_frame))
