import abc
import numpy as np

from .ElementProperties import ElementWithConstraintsProperties


class KinematicJointProperties(ElementWithConstraintsProperties):
    def __init__(self):
        ElementWithConstraintsProperties.__init__(self)
        self.A = np.zeros((6, self.get_number_of_relative_dof()))

    @staticmethod
    @abc.abstractmethod
    def get_number_of_relative_dof():
        pass


class HingeJointProperties(KinematicJointProperties):
    def __init__(self, axis):
        KinematicJointProperties.__init__(self)
        self.axis = axis.reshape((3, 1))
        self.A[3:, :] = self.axis

    @staticmethod
    def get_number_of_relative_dof():
        return 1


class PrismaticJointProperties(KinematicJointProperties):
    def __init__(self, axis):
        KinematicJointProperties.__init__(self)
        self.axis = axis.reshape((3, 1))
        self.A[:3, :] = self.axis

    @staticmethod
    def get_number_of_relative_dof():
        return 1


class CylindricalJointProperties(KinematicJointProperties):
    def __init__(self, axis):
        KinematicJointProperties.__init__(self)
        self.axis = axis.reshape((3, 1))
        self.A[:3, 0] = self.A[3:, 1] = self.axis[:, 0]

    @staticmethod
    def get_number_of_relative_dof():
        return 2


class SphericalJointProperties(KinematicJointProperties):
    def __init__(self):
        KinematicJointProperties.__init__(self)
        self.A[3:, :] = np.eye(3)

    @staticmethod
    def get_number_of_relative_dof():
        return 3
