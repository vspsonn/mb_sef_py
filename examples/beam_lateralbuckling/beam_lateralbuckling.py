import numpy as np

from mb_sef_py.core import Model, NodalFrame
from mb_sef_py.elements import BeamProperties_EIGJ, ClampedFrameProperties, discretize_beam, RigidLinkProperties, \
    KinematicJointElement, GroundJointElement, ServoConstraintProperties
from mb_sef_py.elements.KinematicJointProperties import SphericalJointProperties, HingeJointProperties
from mb_sef_py.math import UnitQuaternion, Frame
from mb_sef_py.solvers import TimeIntegrationParameters, GeneralizedAlpha
from mb_sef_py.utils import Logger, SensorNode, LogNodalFields


def imposed_rotation(t):
    T = 0.4
    if t < T:
        return - 0.5 * np.pi * (1. - np.cos(np.pi * t / T))
    else:
        return - np.pi


model = Model()

beam_L_props = BeamProperties_EIGJ(EA=73e6, GA_1=5.025e6, GA_2=23.40e6,
                                   GJ=877.2, EI_1=60830, EI_2=608.3,
                                   m=2.68, m_11=2233e-6 + 22.33e-6, m_22=2233e-6, m_33=22.33e-6)

beam_l_props = BeamProperties_EIGJ(EA=33.02e6, GA_1=10.81e6, GA_2=10.81e6,
                                   GJ=914.5, EI_1=1189, EI_2=1189,
                                   m=1.212, m_11=2 * 43.65e-6, m_22=43.65e-6, m_33=43.65e-6)

beam_c_props = BeamProperties_EIGJ(EA=132.1e6, GA_1=43.22e6, GA_2=43.22e6,
                                   GJ=14630, EI_1=19020, EI_2=19020,
                                   m=4.85, m_11=2 * 698.3e-6, m_22=698.3e-6, m_33=698.3e-6)

L = 1
l = 0.25
c = 0.05
d = 1.e-4

p_BCL = np.array([0., 0., 0.])
p_MidL = np.array([L/2., 0., 0.])
p_TipL = np.array([L, 0., 0.])
p_TipL_l = np.array([L, d, 0.])
p_BotL_l = np.array([L, d, -l])
p_BC_c = np.array([L-c, d, -l])

q_l = UnitQuaternion()
q_l.set_from_triad(-np.array([0., 0., -1.]), np.array([-1., 0., 0.]))

node_0 = model.add_node(NodalFrame, Frame(x=p_BCL))
node_1 = model.add_node(NodalFrame, Frame(x=p_MidL))
node_2 = model.add_node(NodalFrame, Frame(x=p_TipL))
node_3 = model.add_node(NodalFrame, Frame(x=p_TipL))
node_4 = model.add_node(NodalFrame, Frame(x=p_TipL_l, q=q_l))
node_5 = model.add_node(NodalFrame, Frame(x=p_BotL_l, q=q_l))
node_6 = model.add_node(NodalFrame, Frame(x=p_BotL_l))
node_7 = model.add_node(NodalFrame, Frame(x=p_BC_c))

cl_props = ClampedFrameProperties()
model.add_element(cl_props, node_0)

number_of_element_L = 10
discretize_beam(model, node_0, node_1, number_of_element_L, beam_L_props)
discretize_beam(model, node_1, node_2, number_of_element_L, beam_L_props)

rl_props = RigidLinkProperties()
model.add_element(rl_props, node_2, node_3)

sj_props = SphericalJointProperties()
model.add_element(sj_props, node_3, node_4, element_type=KinematicJointElement)

number_of_element_l = 10
discretize_beam(model, node_4, node_5, number_of_element_l, beam_l_props)

hj_props = HingeJointProperties(axis=np.array([0., 1., 0.]))
model.add_element(hj_props, node_5, node_6, element_type=KinematicJointElement)

number_of_element_c = 5
discretize_beam(model, node_7, node_6, number_of_element_c, beam_c_props)

hinge = model.add_element(hj_props, node_7, element_type=GroundJointElement)

sc_props = ServoConstraintProperties(imposed_rotation)
model.add_element(sc_props, hinge)

logger = Logger('beam_lateralbuckling', periodicity=2)
logger.add_sensor(SensorNode(node_1, LogNodalFields.MOTION))
logger.add_sensor(SensorNode(node_1, LogNodalFields.VELOCITY))

time_integration_parameters = TimeIntegrationParameters()
time_integration_parameters.rho = .9
time_integration_parameters.T = 0.5
time_integration_parameters.h = 1.e-3
integrator = GeneralizedAlpha(model, time_integration_parameters, logger)

integrator.solve()
