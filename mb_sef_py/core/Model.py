import numpy as np
from scipy.sparse import csc_matrix

from .TypeOfVariables import TypeOfVariables
from .TypeOfAnalysis import TypeOfAnalysis
from ..elements.ResidueReturn import ResidueReturn


def dict_of_assembly_coefs():
    return {
        'coef_k': 0.,
        'coef_c': 0.,
        'coef_m': 0.,
        'coef_b': 0.
    }


class TripletsSparseRepresentation:
    def __init__(self, matrix=None, loc_dof=None):
        self.row = []
        self.col = []
        if matrix is None:
            self.data = []
        else:
            self.data = list(matrix.flatten())
            self.row = [i for i in loc_dof for _ in loc_dof]
            self.col = [j for _ in loc_dof for j in loc_dof]

    def add_with_matrix_and_loc_dof(self, matrix, loc_dof):
        self.data += list(matrix.flatten())
        self.row += [i for i in loc_dof for _ in loc_dof]
        self.col += [j for _ in loc_dof for j in loc_dof]


class Model:
    def __init__(self):
        self.number_of_dofs = None
        self.dof_offsets = None

        self.list_nodes = [[] for _ in range(TypeOfVariables.Count)]
        self.list_elements = []

        self.is_meshed = False
        self.is_core_initialized = False

        self.time = 0.
        self.previous_configuration = 0
        self.current_configuration = 0

        self.size_res = 0
        self.res = None
        self.st_triplets = None
        self.analysis_type = TypeOfAnalysis.NONE
        self.inc = None
        self.v = None
        self.v_dot = None

        self.mechanical_power = 0.

    def add_node(self, node_type, *args):
        new_node = node_type(*args)
        field = node_type.get_field()
        number_of_nodes = len(self.list_nodes[field])
        new_node.set_node_number(number_of_nodes)

        self.list_nodes[field].append(new_node)
        return new_node

    def get_node(self, field: int, node_number: int):
        return self.list_nodes[field][node_number]

    def add_element(self, props, *args, **kwargs):
        if props.get_element_type():
            element = props.get_element_type()(props, *args)
        else:
            element_type = kwargs['element_type']
            element = element_type(props, *args)
        self.list_elements.append(element)
        return len(self.list_elements) - 1

    def mesh(self):
        if self.is_meshed:
            return

        for element in self.list_elements:
            element.mesh(self)

        self.number_of_dofs = [0] * TypeOfVariables.Count
        for field in range(TypeOfVariables.Count):
            for node in self.list_nodes[field]:
                self.number_of_dofs[field] += node.mesh(self.number_of_dofs[field])

        self.dof_offsets = []
        self.dof_offsets.append(0)
        for field in range(1, TypeOfVariables.Count):
            self.dof_offsets.append(self.dof_offsets[-1] + self.number_of_dofs[field-1])

        self.is_meshed = True

    def get_number_of_motion_dofs(self):
        return self.number_of_dofs[TypeOfVariables.MOTION] + self.number_of_dofs[TypeOfVariables.RELATIVE_MOTION]

    def get_number_of_lagrange_multipliers(self):
        return self.number_of_dofs[TypeOfVariables.LAGRANGE_MULTIPLIER]

    def initialize(self, analysis_type: int):
        self.analysis_type = analysis_type
        self.mesh()

        self.previous_configuration = 0
        self.current_configuration = 1

        if self.analysis_type != TypeOfAnalysis.STATIC:
            n_fm = self.number_of_dofs[TypeOfVariables.MOTION] + self.number_of_dofs[TypeOfVariables.RELATIVE_MOTION]
            self.v = np.zeros((n_fm,))
            self.v_dot = np.zeros((n_fm,))
        else:
            self.v = None
            self.v_dot = None

        if not self.is_core_initialized:
            self.time = 0.
            self.mechanical_power = 0.

            for field in range(TypeOfVariables.Count):
                for node in self.list_nodes[field]:
                    node.initialize(self)

            for element in self.list_elements:
                element.initialize(self)

            self.is_core_initialized = True

            n_motion = self.get_number_of_motion_dofs()
            n_lm = self.get_number_of_lagrange_multipliers()
            self.size_res = n_motion + n_lm
            self.res = np.zeros((n_motion + n_lm,))
            self.st_triplets = TripletsSparseRepresentation()

        self.inc = np.zeros((self.size_res,))

    def assemble_res_st(self, coefs, solver_param):
        ref = ResidueReturn()
        self.res *= 0.
        self.st_triplets = TripletsSparseRepresentation()
        self.mechanical_power = 0.
        for element in self.list_elements:
            ref = ref + element.assemble_res(self, solver_param)
            self.res[element.loc_dof] += element.res[:]
            element.assemble_st(self, coefs)
            self.st_triplets.add_with_matrix_and_loc_dof(element.st, element.loc_dof)
            self.mechanical_power += element.get_mechanical_power(self)
        return ref

    def build_iteration_matrix_from_sparse_representation(self):
        return csc_matrix((self.st_triplets.data, (self.st_triplets.row, self.st_triplets.col)),
                          shape=(self.size_res, self.size_res))

    def kinematic_update(self, fields=range(TypeOfVariables.Count)):
        for field in fields:
            for node in self.list_nodes[field]:
                i0 = self.dof_offsets[field] + node.get_first_index_dof()
                i1 = i0 + node.get_number_of_dofs()
                node.kinematic_update(self.inc[i0:i1], self.previous_configuration, self.current_configuration)

    def advance_time_step(self, step_size):
        self.time += step_size
        tmp = self.previous_configuration
        self.previous_configuration = self.current_configuration
        self.current_configuration = tmp
