import h5py


class Logger:
    def __init__(self, file_name, periodicity=1):
        self.file_name = file_name
        self.periodicity = periodicity
        self.number_of_steps_logged = 0

        self.model = None
        self.solver = None

        self.file_id = None
        self.time_dataset_id = None
        self.mechanical_power_dataset_id = None
        self.nit_dataset_id = None

        self.list_sensors = []

    def add_sensor(self, sensor):
        self.list_sensors.append(sensor)

    def initialize(self, model, solver):
        self.model = model
        self.solver = solver

        self.file_id = h5py.File(self.file_name + '.h5', mode='w')
        self.number_of_steps_logged = 0

        self.time_dataset_id = self.file_id.create_dataset('time', (1, ), maxshape=(None, ))
        self.mechanical_power_dataset_id = self.file_id.create_dataset('mechanical_power', (1,), maxshape=(None,))
        self.nit_dataset_id = self.file_id.create_dataset('number_of_iterations', (1,), maxshape=(None,))

        for sensor in self.list_sensors:
            group_name = sensor.get_group_name()
            name, size = sensor.get_dataset_name(), sensor.get_dataset_number_of_rows()
            sensor.dataset_id = self.file_id.create_dataset(group_name + '/' + name, (size, 0), maxshape=(size, None))

    def log_step(self, step_number):
        if step_number % self.periodicity == 0:
            self.time_dataset_id.resize((self.number_of_steps_logged+1, ))
            self.time_dataset_id[self.number_of_steps_logged] = self.model.time

            self.mechanical_power_dataset_id.resize((self.number_of_steps_logged+1, ))
            self.mechanical_power_dataset_id[self.number_of_steps_logged] = self.model.mechanical_power

            self.nit_dataset_id.resize((self.number_of_steps_logged + 1,))
            self.nit_dataset_id[self.number_of_steps_logged] = self.solver.number_of_iterations

            for sensor in self.list_sensors:
                sensor.log_step(self.model, self.number_of_steps_logged)

            self.number_of_steps_logged += 1

    def finalize(self):
        self.file_id.close()
