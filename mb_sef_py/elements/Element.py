import abc
import numpy as np

from .ResidueReturn import ResidueReturn
from ..core.TypeOfVariables import TypeOfVariables
from ..core.NodeLagrangeMultipliers import NodeLagrangeMultipliers


class Element(abc.ABC):
    def __init__(self, props):
        self.elem_props = props

        self.list_nodes = [[] for _ in range(TypeOfVariables.Count)]
        self.loc_dof = []
        self.res = None
        self.at = None
        self.st = None

    @abc.abstractmethod
    def get_number_of_dofs(self):
        pass

    def get_size_res(self):
        return self.get_number_of_dofs()

    def add_node(self, node):
        self.list_nodes[node.get_field()].append(node)

    def mesh(self, model):
        pass

    def build_loc_dof(self, model):
        self.loc_dof = []
        for field in range(TypeOfVariables.Count):
            for node in self.list_nodes[field]:
                i0 = model.dof_offsets[field] + node.get_first_index_dof()
                for i in range(i0, i0+node.get_number_of_dofs()):
                    self.loc_dof.append(i)

    def initialize(self, model):
        self.build_loc_dof(model)
        size_res = self.get_size_res()
        self.res = np.zeros((size_res,))
        self.st = np.zeros((size_res, size_res))

    def assemble_res(self, model, solver_params):
        self.build_loc_dof(model)
        self.assemble_res_impl(model, solver_params)
        return ResidueReturn(norm_forces=np.linalg.norm(self.res))

    def assemble_res_impl(self, model, solver_params):
        pass

    def assemble_st(self, model,  coefs):
        n_dof = self.get_number_of_dofs()
        if coefs['coef_k'] != 0. and self.assemble_kt_impl(model):
            self.st[np.ix_(range(n_dof), range(n_dof))] = coefs['coef_k'] * self.at
        else:
            self.st[np.ix_(range(n_dof), range(n_dof))] *= 0.
        if coefs['coef_c'] != 0. and self.assemble_ct_impl(model):
            self.st[np.ix_(range(n_dof), range(n_dof))] += coefs['coef_c'] * self.at
        if coefs['coef_m'] != 0. and self.assemble_mt_impl(model):
            self.st[np.ix_(range(n_dof), range(n_dof))] += coefs['coef_m'] * self.at

    def assemble_kt_impl(self, model):
        return False

    def assemble_ct_impl(self, model):
        return False

    def assemble_mt_impl(self, model):
        return False

    def get_mechanical_power(self, model):
        return np.dot(model.v[self.loc_dof], self.res)


class ElementWithConstraints(Element):
    def __init__(self, props):
        Element.__init__(self, props)
        self.bt = None
        self.constraint = None

    @staticmethod
    @abc.abstractmethod
    def get_number_of_constraints():
        pass

    def mesh(self, model):
        Element.mesh(self, model)
        node_lm = model.add_node(NodeLagrangeMultipliers, self.get_number_of_constraints(),
                                 self.elem_props.constraint_scaling)
        self.list_nodes[TypeOfVariables.LAGRANGE_MULTIPLIER].append(node_lm)

    def get_size_res(self):
        return self.get_number_of_dofs() + self.get_number_of_constraints()

    def assemble_res(self, model, solver_params):
        self.build_loc_dof(model)

        self.assemble_constraint_and_bt(model)

        lambd = self.list_nodes[TypeOfVariables.LAGRANGE_MULTIPLIER][0].lambd[model.current_configuration]
        self.res[:self.get_number_of_dofs()] = np.matmul(np.transpose(self.bt), lambd)
        self.res[-self.get_number_of_constraints():] = self.constraint

        ref = ResidueReturn(norm_forces=np.linalg.norm(self.res[:self.get_number_of_dofs()]),
                            norm_constraints=np.linalg.norm(self.res[-self.get_number_of_constraints():]))
        self.res[-self.get_number_of_constraints():] *= self.elem_props.constraint_scaling
        return ref

    def assemble_st(self, model,  coefs):
        Element.assemble_st(self, model, coefs)
        n_dof, n_cons = self.get_number_of_dofs(), self.get_number_of_constraints()
        if coefs['coef_b'] != 0. and self.assemble_bt_impl(model):
            sca_coef_b = self.elem_props.constraint_scaling * coefs['coef_b']
            self.st[np.ix_(range(n_dof), range(n_dof, n_dof+n_cons))] = sca_coef_b * np.transpose(self.bt)
            self.st[np.ix_(range(n_dof, n_dof+n_cons), range(n_dof))] = sca_coef_b * self.bt
        else:
            self.st[np.ix_(range(n_dof), range(n_dof, n_dof+n_cons))] *= 0.
            self.st[np.ix_(range(n_dof, n_dof+n_cons), range(n_dof))] *= 0.

    @abc.abstractmethod
    def assemble_constraint_and_bt(self, model):
        pass

    def assemble_bt_impl(self, model):
        return True

    def get_mechanical_power(self, model):
        return 0.
