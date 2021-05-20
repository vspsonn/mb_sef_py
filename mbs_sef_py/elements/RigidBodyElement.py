import numpy as np

from .Element import Element
from .ElementProperties import ElementProperties
from ..core.Model import TypeOfAnalysis
from ..math.SO3 import tilde


class RigidBodyProperties(ElementProperties):
    def __init__(self, m, J):
        ElementProperties.__init__(self)
        self.m = m
        self.J = J

    @staticmethod
    def get_element_type():
        return RigidBodyElement


class RigidBodyElement(Element):
    def __init__(self, props, node):
        Element.__init__(self, props)
        self.add_node(node)
        self.mt = np.block([[self.elem_props.m * np.eye(3), np.zeros((3, 3))], [np.zeros((3, 3)), self.elem_props.J]])

    def get_number_of_dofs(self):
        return 6

    def assemble_res_impl(self, model, solver_params):
        if model.analysis_type != TypeOfAnalysis.DYNAMIC:
            self.res = np.zeros((6,))
            return

        v_dot = model.v_dot[self.loc_dof]
        u_dot, omega_dot = v_dot[:3], v_dot[3:]
        v = model.v[self.loc_dof]
        u, omega = v[:3], v[3:]

        self.res[:3] = self.elem_props.m * (u_dot + np.matmul(tilde(omega), u))
        self.res[3:] = np.matmul(self.elem_props.J, omega_dot) + \
                       np.matmul(tilde(omega), np.matmul(self.elem_props.J, omega))

    def assemble_ct_impl(self, model):
        v = model.v[self.loc_dof]
        u, omega = v[:3], v[3:]

        self.at = np.block([[tilde(self.elem_props.m * omega),
                             tilde(- self.elem_props.m * u)],
                            [np.zeros((3, 3)),
                             np.matmul(tilde(omega), self.elem_props.J) - tilde(np.matmul(self.elem_props.J, omega))]])
        return True

    def assemble_mt_impl(self, model):
        self.at = self.mt
        return True
