import abc


class Node(abc.ABC):
    def __init__(self, name):
        self.name = name

        self.node_number = None
        self.first_index_dof = None

        self.v0 = None

    @staticmethod
    @abc.abstractmethod
    def get_field():
        pass

    @abc.abstractmethod
    def get_number_of_dofs(self):
        pass

    @abc.abstractmethod
    def get_motion_coordinates(self, configuration):
        pass

    def get_number_of_motion_coordinates(self):
        return self.get_number_of_dofs()

    def set_node_number(self, node_number):
        self.node_number = node_number

    def get_node_number(self):
        return self.node_number

    def get_first_index_dof(self):
        return self.first_index_dof

    def mesh(self, number_of_dofs):
        self.first_index_dof = number_of_dofs
        return self.get_number_of_dofs()

    def set_initial_velocity(self, v0):
        self.v0 = v0

    def initialize(self, model):
        if self.v0 is not None:
            i0 = model.dof_offsets[self.get_field()] + self.get_first_index_dof()
            i1 = i0 + self.get_number_of_dofs()
            model.v[i0:i1] = self.v0[:]

    def kinematic_update(self, inc, previous_index, current_index):
        pass
