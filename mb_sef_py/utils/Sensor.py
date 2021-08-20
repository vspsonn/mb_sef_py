import abc


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
            return self.node.get_number_of_motion_coordinates()
        elif self.log_field in [LogNodalFields.VELOCITY, LogNodalFields.ACCELERATION]:
            return self.node.get_number_of_dofs()
        else:
            return 0

    def log_step(self, model, step):
        self.dataset_id.resize(step+1, axis=1)
        if self.log_field == LogNodalFields.MOTION:
            self.dataset_id[:, step] = self.node.get_motion_coordinates(model.current_configuration)
        elif self.log_field == LogNodalFields.VELOCITY:
            first_index_dof = model.dof_offsets[self.node.get_field()] + self.node.get_first_index_dof()
            self.dataset_id[:, step] = model.v[first_index_dof:first_index_dof+self.node.get_number_of_dofs()]
        elif self.log_field == LogNodalFields.ACCELERATION:
            first_index_dof = model.dof_offsets[self.node.get_field()] + self.node.get_first_index_dof()
            self.dataset_id[:, step] = model.v_dot[first_index_dof:first_index_dof+self.node.get_number_of_dofs()]
