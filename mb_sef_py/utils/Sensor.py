import abc
import numpy as np
from ..core.TypeOfVariables import TypeOfVariables


class LogNodalFields:
    MOTION, VELOCITY, ACCELERATION = range(3)


class LogElementFields:
    LM = range(1)


class Sensor(abc.ABC):
    def __init__(self):
        self.dataset_id = None

    @abc.abstractmethod
    def get_group_name(self):
        pass

    @abc.abstractmethod
    def get_dataset_name(self):
        pass

    @abc.abstractmethod
    def get_dataset_number_of_rows(self):
        pass

    @abc.abstractmethod
    def log_step(self, model, step):
        pass


class SensorNode(Sensor):
    def __init__(self, node, log_field):
        Sensor.__init__(self)
        self.node = node
        self.log_field = log_field

    def get_group_name(self):
        if self.node.name is not None:
            return self.node.name
        else:
            return 'node_' + str(self.node.node_number)

    def get_dataset_name(self):
        if self.log_field == LogNodalFields.MOTION:
            return 'MOTION'
        elif self.log_field == LogNodalFields.VELOCITY:
            return 'VELOCITY'
        elif self.log_field == LogNodalFields.ACCELERATION:
            return 'ACCELERATION'
        else:
            return ''

    def get_dataset_number_of_rows(self):
        if self.log_field == LogNodalFields.MOTION:
            return 7
        elif self.log_field in [LogNodalFields.VELOCITY, LogNodalFields.ACCELERATION]:
            return 6
        else:
            return 0

    def log_step(self, model, step):
        self.dataset_id.resize(step+1, axis=1)
        if self.log_field == LogNodalFields.MOTION:
            frame = self.node.frame[model.current_configuration]
            self.dataset_id[:, step] = np.block([frame.x, frame.q.e0, frame.q.e])
        elif self.log_field == LogNodalFields.VELOCITY:
            first_index_dof = model.dof_offsets[TypeOfVariables.MOTION] + self.node.get_first_index_dof()
            self.dataset_id[:, step] = model.v[first_index_dof:first_index_dof+6]
        elif self.log_field == LogNodalFields.ACCELERATION:
            first_index_dof = model.dof_offsets[TypeOfVariables.MOTION] + self.node.get_first_index_dof()
            self.dataset_id[:, step] = model.v_dot[first_index_dof:first_index_dof+6]
