import abc
import numpy as np

from .Element import Element
from .ElementProperties import ElementProperties
from ..core import NodalFrame
from ..core.TypeOfAnalysis import TypeOfAnalysis
from ..core.TypeOfVariables import TypeOfVariables
from ..math import Frame
from ..math.GaussPoints import GaussPoints_2
from ..math.SE3 import tilde6, breve6
from ..math.SO3 import tilde


class BeamProperties(ElementProperties):
    def __init__(self):
        ElementProperties.__init__(self)
        self.K, self.M = self.get_beam_matrices()
        self.distributed_load = None
        self.distributed_follower_load = None
        self.gravity = None

    @staticmethod
    def get_element_type():
        return BeamElement

    @abc.abstractmethod
    def get_beam_matrices(self):
        pass


class BeamProperties_Enu(BeamProperties):
    def __init__(self, E, nu, A, A_1, A_2, J, I_1, I_2, rho):
        self.E = E
        self.nu = nu
        self.A = A
        self.A_1 = A_1
        self.A_2 = A_2
        self.J = J
        self.I_1 = I_1
        self.I_2 = I_2
        self.rho = rho
        BeamProperties.__init__(self)

    def get_beam_matrices(self):
        G = self.E / (2. * (1. + self.nu))
        K = np.diag([self.E * self.A, G * self.A_1, G * self.A_2,
                     G * self.J, self.E * self.I_1, self.E * self.I_2])
        M = self.rho * np.diag([self.A, self.A, self.A, self.J, self.I_1, self.I_2])
        return K, M


class BeamProperties_EIGJ(BeamProperties):
    def __init__(self, EA, GA_1, GA_2, GJ, EI_1, EI_2, m, m_11, m_22, m_33):
        self.EA = EA
        self.GA_1 = GA_1
        self.GA_2 = GA_2
        self.GJ = GJ
        self.EI_1 = EI_1
        self.EI_2 = EI_2
        self.m = m
        self.m_11 = m_11
        self.m_22 = m_22
        self.m_33 = m_33
        BeamProperties.__init__(self)

    def get_beam_matrices(self):
        K = np.diag([self.EA, self.GA_1, self.GA_2, self.GJ, self.EI_1, self.EI_2])
        M = np.diag([self.m, self.m, self.m, self.m_11, self.m_22, self.m_33])
        return K, M


def discretize_beam(model, node_start, node_end, number_of_element, props):
    frame_start = node_start.frame_ref
    frame_end = node_end.frame_ref

    parameters_of_relative_frame = Frame.get_parameters_from_frame(frame_start.get_inverse() * frame_end)

    n0, n1 = node_start, None
    for i in range(1, number_of_element+1):
        if i == number_of_element:
            n1 = node_end
        else:
            relative_frame_i = Frame.get_frame_from_parameters((1.*i)/number_of_element * parameters_of_relative_frame)
            frame = Frame(ref_frame=frame_start * relative_frame_i)
            n1 = model.add_node(NodalFrame, frame)

        model.add_element(props, n0, n1)
        n0 = n1


class BeamElement(Element):
    def __init__(self, props, node_A, node_B):
        Element.__init__(self, props)
        self.add_node(node_A)
        self.add_node(node_B)

        self.d0 = None
        self.L = None
        self.kt = None
        self.ct = None
        self.mt = None

        self.gps = GaussPoints_2()

    def get_number_of_dofs(self):
        return 12

    def initialize(self, model):
        Element.initialize(self, model)
        HA = self.list_nodes[TypeOfVariables.MOTION][0].frame_ref
        HB = self.list_nodes[TypeOfVariables.MOTION][1].frame_ref
        self.d0 = Frame.get_parameters_from_frame(HA.get_inverse() * HB)
        self.L = np.linalg.norm(self.d0[:3])

    def assemble_res_impl(self, model, solver_params):
        HA = self.list_nodes[TypeOfVariables.MOTION][0].frame[model.current_configuration]
        HB = self.list_nodes[TypeOfVariables.MOTION][1].frame[model.current_configuration]

        d = Frame.get_parameters_from_frame(HA.get_inverse() * HB)
        P = np.zeros((6, 12))
        P[:, :6], P[:, 6:] = -Frame.get_inverse_tangent_operator(-d), Frame.get_inverse_tangent_operator(d)
        PTK = np.matmul(np.transpose(P), self.elem_props.K / self.L)

        self.res = np.matmul(PTK, d - self.d0)
        self.kt = np.matmul(PTK, P)

        self.ct, self.mt = np.zeros((12, 12)), np.zeros((12, 12))
        Q = np.zeros((6, 12))
        for x, w in zip(*self.gps.xw):
            s = 0.5 * (x + 1.)
            T_star = s * np.matmul(Frame.get_tangent_operator(s * d), P[:, 6:])
            Q[:, :6], Q[:, 6:] = np.eye(6) - T_star, T_star

            distributed_load, distributed_load_flag = np.zeros((6,)), False
            if self.elem_props.distributed_load is not None:
                distributed_load_flag = True
                distributed_load += self.elem_props.distributed_load(s, model.time)
            if self.elem_props.gravity is not None:
                distributed_load_flag = True
                distributed_load[:3] += self.elem_props.M[0, 0] * self.elem_props.gravity

            if distributed_load_flag:
                H = HA * Frame.get_frame_from_parameters(s * d)
                RT = np.transpose(H.q.get_rotation_matrix())
                distributed_load[:3] = np.matmul(RT, distributed_load[:3])
                distributed_load[3:] = np.matmul(RT, distributed_load[3:])
                dloadQ = np.block([[np.matmul(tilde((0.5 * w * self.L) * distributed_load[:3]), Q[3:, :])],
                                   [np.matmul(tilde((0.5 * w * self.L) * distributed_load[3:]), Q[3:, :])]])
                self.kt -= np.matmul(np.transpose(Q), dloadQ)

            if self.elem_props.distributed_follower_load is not None:
                distributed_load_flag = True
                distributed_load += self.elem_props.distributed_follower_load(s, model.time)

            if distributed_load_flag:
                self.res -= np.matmul(np.transpose(Q), (0.5 * w * self.L) * distributed_load)

            if model.analysis_type == TypeOfAnalysis.DYNAMIC:
                m_gp = self.elem_props.M * (0.5 * w * self.L)
                v_gp = np.matmul(Q, model.v[self.loc_dof])
                mv_gp = np.matmul(m_gp, v_gp)

                vgp_tilde6T = np.transpose(tilde6(v_gp))
                self.res -= np.matmul(np.transpose(Q), np.matmul(vgp_tilde6T, mv_gp))
                self.ct -= np.matmul(np.transpose(Q), np.matmul(np.matmul(vgp_tilde6T, m_gp) + breve6(mv_gp), Q))

                QTM = np.matmul(np.transpose(Q), m_gp)
                self.res += np.matmul(QTM, np.matmul(Q, model.v_dot[self.loc_dof]))
                self.mt += np.matmul(QTM, Q)

    def assemble_kt_impl(self, model):
        HA = self.list_nodes[TypeOfVariables.MOTION][0].frame[model.current_configuration]
        HB = self.list_nodes[TypeOfVariables.MOTION][1].frame[model.current_configuration]

        d = Frame.get_parameters_from_frame(HA.get_inverse() * HB)
        P = np.zeros((6, 12))
        P[:, :6], P[:, 6:] = -Frame.get_inverse_tangent_operator(-d), Frame.get_inverse_tangent_operator(d)
        F = np.matmul(self.elem_props.K, d-self.d0)
        self.kt[:6, :] += np.matmul(Frame.get_derivative_inverse_transposed_tangent_operator(-d, F), P)
        self.kt[6:, :] += np.matmul(Frame.get_derivative_inverse_transposed_tangent_operator(d, F), P)

        self.at = self.kt
        return True

    def assemble_ct_impl(self, model):
        self.at = self.ct
        return True

    def assemble_mt_impl(self, model):
        self.at = self.mt
        return True
