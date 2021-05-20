import numpy as np
from scipy.integrate import quad as scipy_quad

from mb_sef_py.core import Model, NodalFrame
from mb_sef_py.elements import BeamProperties_EIGJ, discretize_beam, HingeJointProperties, GroundJointElement, \
    ServoConstraintProperties, RigidBodyProperties, RigidLinkProperties, \
    CylindricalJointProperties, ExternalForceProperties
from mb_sef_py.math import Frame
from mb_sef_py.solvers import TimeIntegrationParameters, GeneralizedAlpha
from mb_sef_py.utils import Logger, SensorNode, LogNodalFields


def imposed_rotational_velocity(t):
    A1, A2, T1, T2, T3, omega = 0.8, 1.2, 0.5, 1., 1.25, 60.
    if t < T1:
        return 0.5 * A1 * omega * (1. - np.cos(np.pi * t/T1))
    elif t < T2:
        return A1 * omega
    elif t < T3:
        return A1 * omega + 0.5 * (A2 - A1) * omega * (1. - np.cos(np.pi * (t - T2)/(T3 - T2)))
    else:
        return A2 * omega


def imposed_rotation(t):
    return scipy_quad(imposed_rotational_velocity, 0., t)[0]


model = Model()

beam_props = BeamProperties_EIGJ(EA=313.4e6, GA_1=60.5e6, GA_2=60.5e6,
                                 GJ=272.7e3, EI_1=354.5e3, EI_2=354.5e3,
                                 m=11.64, m_11=2 * 13.17e-3, m_22=13.17e-3, m_33=13.17e-3)
beam_props.gravity = np.array([0., 0., -9.81])

p_left = np.array([0., 0., 0.])
p_mid = np.array([3., 0., 0.])
p_mid_d = np.array([3., 0., 0.05])
p_right = np.array([6., 0., 0.])

node_left = model.add_node(NodalFrame, Frame(x=p_left))
node_mid = model.add_node(NodalFrame, Frame(x=p_mid), 'node_mid')
node_mid_d = model.add_node(NodalFrame, Frame(x=p_mid_d))
node_right = model.add_node(NodalFrame, Frame(x=p_right))

hj_props = HingeJointProperties(axis=np.array([1., 0., 0.]))
hinge = model.add_element(hj_props, node_left, element_type=GroundJointElement)
sc_props = ServoConstraintProperties(imposed_rotation)
model.add_element(sc_props, hinge)

number_of_elements = 8
discretize_beam(model, node_left, node_mid, number_of_elements, beam_props)
discretize_beam(model, node_mid, node_right, number_of_elements, beam_props)

rb_props = RigidBodyProperties(m=70.573, J=np.diag([2.0325, 1.0163, 1.0163]))
model.add_element(rb_props, node_mid_d)

ef_props = ExternalForceProperties()
ef_props.forces[2] = -9.81 * 70.573
model.add_element(ef_props, node_mid_d)

rl_props = RigidLinkProperties()
model.add_element(rl_props, node_mid, node_mid_d)

cj_props = CylindricalJointProperties(axis=np.array([1., 0., 0.]))
model.add_element(cj_props, node_right, element_type=GroundJointElement)

logger = Logger('beam_rotatingshaft', periodicity=2)
logger.add_sensor(SensorNode(node_mid, LogNodalFields.MOTION))
logger.add_sensor(SensorNode(node_mid, LogNodalFields.VELOCITY))

time_integration_parameters = TimeIntegrationParameters()
time_integration_parameters.rho = .0
time_integration_parameters.T = 2.5e-0
time_integration_parameters.h = 1.e-4
time_integration_parameters.tol_res_forces = 1.e-6
integrator = GeneralizedAlpha(model, time_integration_parameters, logger)

integrator.solve()
