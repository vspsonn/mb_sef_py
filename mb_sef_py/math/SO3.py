import numpy as np


def tilde(x):
    x_tilde = np.zeros((3, 3))
    x_tilde[0, 1] = -x[2]
    x_tilde[0, 2] = x[1]
    x_tilde[1, 0] = x[2]
    x_tilde[1, 2] = -x[0]
    x_tilde[2, 0] = -x[1]
    x_tilde[2, 1] = x[0]
    return x_tilde


def tilde_x_tilde(x):
    x_tt = np.zeros((3, 3))
    x0_2, x1_2, x2_2 = x[0] ** 2, x[1] ** 2, x[2] ** 2
    x_tt[0, 0] = -x2_2 - x1_2
    x_tt[0, 1] = x_tt[1, 0] = x[1] * x[0]
    x_tt[0, 2] = x_tt[2, 0] = x[2] * x[0]
    x_tt[1, 1] = -x2_2 - x0_2
    x_tt[1, 2] = x_tt[2, 1] = x[2] * x[1]
    x_tt[2, 2] = -x1_2 - x0_2
    return x_tt


class UnitQuaternion:
    def __init__(self, e0=1., e=None, ref_unit_quaternion=None):
        if ref_unit_quaternion is None:
            self.e0 = e0
            if e is None:
                self.e = np.zeros((3,))
            else:
                self.e = e[:]
        else:
            self.e0 = ref_unit_quaternion.e0
            self.e = ref_unit_quaternion.e[:]

    def __mul__(self, other):
        e0 = self.e0 * other.e0 - np.dot(self.e[:], other.e[:])
        e = self.e0 * other.e[:] + other.e0 * self.e[:] + np.matmul(tilde(self.e[:]), other.e[:])
        return UnitQuaternion(e0=e0, e=e)

    def rotate_vector(self, vec):
        rotated_vec = np.copy(vec)
        rotated_vec += np.matmul(tilde_x_tilde(self.e), 2. * vec)
        rotated_vec[0] += 2.*self.e0*(self.e[1]*vec[2] - vec[1]*self.e[2])
        rotated_vec[1] += 2.*self.e0*(self.e[2]*vec[0] - vec[2]*self.e[0])
        rotated_vec[2] += 2.*self.e0*(self.e[0]*vec[1] - vec[0]*self.e[1])
        return rotated_vec

    def set_from_unitquaternion(self, q):
        self.e0 = q.e0
        self.e = q.e[:]

    def set_from_triad(self, v1, v2, v3=None):
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        if v3 is None:
            v3 = np.matmul(tilde(v1), v2)
        else:
            v3 = v3/np.linalg.norm(v3)

        ind = 0
        s_max = v1[0] + v2[1] + v3[2]
        s = v1[0] - v2[1] - v3[2]
        if s > s_max:
            s_max = s
            ind = 1
        s = -v1[0] + v2[1] - v3[2]
        if s > s_max:
            s_max = s
            ind = 2
        s = -v1[0] - v2[1] + v3[2]
        if s > s_max:
            s_max = s
            ind = 3

        s = 0.5 * np.sqrt(s_max + 1.)
        if ind == 0:
            self.e0 = s
            self.e[0] = 1./(4. * s) * (v2[2] - v3[1])
            self.e[1] = 1./(4. * s) * (v3[0] - v1[2])
            self.e[2] = 1./(4. * s) * (v1[1] - v2[0])
        elif ind == 1:
            self.e0 = 1./(4. * s) * (v2[2] - v3[1])
            self.e[0] = s
            self.e[1] = 1./(4. * s) * (v2[0] + v1[1])
            self.e[2] = 1./(4. * s) * (v3[0] + v1[2])
        elif ind == 2:
            self.e0 = 1./(4. * s) * (v3[0] - v1[2])
            self.e[0] = 1./(4. * s) * (v1[1] + v2[0])
            self.e[1] = s
            self.e[2] = 1./(4. * s) * (v3[1] + v2[2])
        else:
            self.e0 = 1. / (4. * s) * (v1[1] - v2[0])
            self.e[0] = 1. / (4. * s) * (v3[0] + v1[2])
            self.e[1] = 1. / (4. * s) * (v3[1] + v2[2])
            self.e[2] = s

    def get_inverse(self):
        return UnitQuaternion(e0=self.e0, e=-self.e[:])

    def get_rotation_matrix(self):
        e_tilde = tilde(self.e[:])
        return np.eye(3) + 2. * (self.e0 * e_tilde + np.matmul(e_tilde, e_tilde))

    @staticmethod
    def get_unitquat_from_parameters(parameters):
        p0 = np.sqrt(1. - 0.25 * np.dot(parameters, parameters))
        return UnitQuaternion(e0=p0, e=0.5 * parameters)

    @staticmethod
    def get_parameters_from_unitquat(unitquat):
        return 2. * unitquat.e[:]
