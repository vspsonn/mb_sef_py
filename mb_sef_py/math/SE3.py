import numpy as np
from .SO3 import UnitQuaternion, tilde


def tilde6(x):
    x_tilde6 = np.zeros((6, 6))
    x_tilde6[:3, :3] = x_tilde6[3:, 3:] = tilde(x[3:])
    x_tilde6[:3, 3:] = tilde(x[:3])
    return x_tilde6


def breve6(x):
    x_breve6 = np.zeros((6, 6))
    x_breve6[:3, 3:] = x_breve6[3:, :3] = tilde(x[:3])
    x_breve6[3:, 3:] = tilde(x[3:])
    return x_breve6


class Frame:
    def __init__(self, x=np.zeros((3,)), q=UnitQuaternion(), ref_frame=None):
        if ref_frame is None:
            self.x = x
            self.q = q
        else:
            self.x = ref_frame.x[:]
            self.q = UnitQuaternion(ref_unit_quaternion=ref_frame.q)

    def __mul__(self, other):
        x = self.x[:] + self.q.rotate_vector(other.x)
        q = self.q * other.q
        return Frame(x=x, q=q)

    def get_inverse(self):
        q_inv = self.q.get_inverse()
        x = q_inv.rotate_vector(-self.x)
        return Frame(x=x, q=q_inv)

    def get_adjoint(self):
        R = self.q.get_rotation_matrix()
        return np.block([[R, np.matmul(R, tilde(self.x[:]))], [np.zeros((3, 3)), R]])

    def get_inverse_adjoint(self):
        RT = np.transpose(self.q.get_rotation_matrix())
        return np.block([[RT, np.matmul(tilde(-self.x[:]), RT)], [np.zeros((3, 3)), RT]])

    @staticmethod
    def get_frame_from_parameters(parameters):
        p0 = np.sqrt(1. - 0.25 * np.dot(parameters[3:], parameters[3:]))
        q = UnitQuaternion(e0=p0, e=0.5 * parameters[3:])

        p_tilde_over_2 = tilde(0.5 * parameters[3:])
        TT = 1. / p0 * (np.eye(3) + np.matmul(p_tilde_over_2, p_tilde_over_2)) + p_tilde_over_2
        x = np.matmul(TT, parameters[:3])
        return Frame(x=x, q=q)

    @staticmethod
    def get_parameters_from_frame(frame):
        p = np.zeros((6,))
        p[3:] = 2. * frame.q.e[:]
        Tm1T = frame.q.e0 * np.eye(3) - tilde(frame.q.e[:])
        p[:3] = np.matmul(Tm1T, frame.x)
        return p

    @staticmethod
    def get_tangent_operator(parameters):
        T = np.zeros((6, 6))
        p0 = np.sqrt(1. - 0.25 * np.dot(parameters[3:], parameters[3:]))
        p_tilde = tilde(0.5 * parameters[3:])
        tmp = np.eye(3) + np.matmul(p_tilde, p_tilde)
        T[:3, :3] = T[3:, 3:] = 1. / p0 * tmp - p_tilde
        rho = np.dot(parameters[:3], parameters[3:])
        p_u_tilde = tilde(0.5 * parameters[:3])
        T[:3, 3:] = rho / (4. * p0 ** 3) * tmp - p_u_tilde + \
                    1. / (4. * p0) * (np.matmul(p_tilde, p_u_tilde) + np.matmul(p_u_tilde, p_tilde))
        return T

    @staticmethod
    def get_inverse_tangent_operator(parameters):
        Tm1 = np.zeros((6, 6))
        p0 = np.sqrt(1. - 0.25 * np.dot(parameters[3:], parameters[3:]))
        Tm1[:3, :3] = Tm1[3:, 3:] = p0 * np.eye(3) + tilde(0.5 * parameters[3:])
        rho = np.dot(parameters[:3], parameters[3:])
        Tm1[:3, 3:] = -rho / (4. * p0) * np.eye(3) + tilde(0.5 * parameters[:3])
        return Tm1

    @staticmethod
    def get_derivative_inverse_transposed_tangent_operator(parameters, direction):
        pu, pw = parameters[:3].reshape((3, 1)), parameters[3:].reshape((3, 1))
        du, dw = direction[:3].reshape((3, 1)), direction[3:].reshape((3, 1))

        p0, rho = np.sqrt(1. - 0.25 * np.dot(parameters[3:], parameters[3:])), np.dot(parameters[3:], parameters[:3])

        DTinvT0u = tilde(0.5 * du) - np.matmul((0.25/p0) * du, np.transpose(pw))
        DTinvT0r = tilde(0.5 * dw) - np.matmul((0.25/p0) * dw, np.transpose(pw))
        DTinvT2 = np.matmul(-0.25 * rho * 0.25/(p0*p0*p0) * du, np.transpose(pw)) - np.matmul(0.25/p0 * du, np.transpose(pu))

        return np.block([[np.zeros((3, 3)), DTinvT0u], [DTinvT0u, DTinvT0r+DTinvT2]])
