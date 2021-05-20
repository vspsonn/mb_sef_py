import abc


class Node(abc.ABC):
    def __init__(self, name):
        self.name = name

        self.number_of_nodal_dofs = None

        self.node_number = None
        self.first_index_dof = None

    def set_node_number(self, node_number):
        self.node_number = node_number

    def get_node_number(self):
        return self.node_number

    def get_first_index_dof(self):
        return self.first_index_dof

    @staticmethod
    @abc.abstractmethod
    def get_field():
        pass

    @abc.abstractmethod
    def get_number_of_dofs(self):
        pass

    def mesh(self, number_of_dofs):
        self.first_index_dof = number_of_dofs
        return self.get_number_of_dofs()

    def initialize(self):
        pass

    def kinematic_update(self, inc, previous_index, current_index):
        pass
