import numpy as np

from .Element import Element
from .ElementProperties import ElementProperties
from ..core.TypeOfVariables import TypeOfVariables
from ..math.SO3 import tilde


class ExternalForceProperties(ElementProperties):
    def __init__(self):
        ElementProperties.__init__(self)
        self.forces = np.zeros((6,))
        self.time_dependent_force = None
        self.follower = False

    @staticmethod
    def get_element_type():
        return ExternalForce


class ExternalForce(Element):
    def __init__(self, props, node):
        Element.__init__(self, props)
        self.add_node(node)

    def get_number_of_dofs(self):
        return 6

    def assemble_res_impl(self, model, solver_params):
        if self.elem_props.time_dependent_force is not None:
            self.elem_props.forces = self.elem_props.time_dependent_force(model.time)

        if self.elem_props.follower:
            self.res = - self.elem_props.forces[:]
        else:
            nodal_frame = self.list_nodes[TypeOfVariables.MOTION][0].frame[model.current_configuration]
            RT = nodal_frame.q.get_inverse().get_rotation_matrix()
            self.res[:3] = - np.matmul(RT, self.elem_props.forces[:3])
            self.res[3:] = - np.matmul(RT, self.elem_props.forces[3:])

    def assemble_kt_impl(self, model):
        if self.elem_props.follower:
            return False
        else:
            self.at = np.block([[np.zeros((3, 3)), tilde(self.res[:3])],
                                [np.zeros((3, 3)), tilde(self.res[3:])]])
            return True
