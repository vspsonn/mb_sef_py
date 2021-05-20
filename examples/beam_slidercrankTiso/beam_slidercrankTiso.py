import numpy as np

from mb_sef_py.core import NodalFrame, Model
from mb_sef_py.elements import BeamProperties_EIGJ, discretize_beam, HingeJointProperties, KinematicJointElement, \
    GroundJointElement, PrismaticJointProperties, ServoConstraintProperties
from mb_sef_py.math import Frame
from mb_sef_py.solvers import TimeIntegrationParameters, GeneralizedAlpha
from mb_sef_py.utils import Logger, SensorNode, LogNodalFields


def imposed_rotation(t):
    if t < 5.:
        return 3. * np.pi * t**2 / 50.
    else:
        return 3. * np.pi / 2.


model = Model()

L_c = 10.
L_r = 20.

x0 = np.array([0., 0., 0.])
xM = np.array([L_c/2, 0., 0.])
xB = np.array([L_c, 0., 0.])
xN = np.array([L_c + L_r/2, 0., 0.])
xS = np.array([L_c + L_r, 0., 0.])

node_0 = model.add_node(NodalFrame, Frame(x0), 'node_0')
node_M = model.add_node(NodalFrame, Frame(xM), 'node_M')
node_Bc = model.add_node(NodalFrame, Frame(xB))
node_Br = model.add_node(NodalFrame, Frame(xB), 'node_Br')
node_N = model.add_node(NodalFrame, Frame(xN), 'node_N')
node_S = model.add_node(NodalFrame, Frame(xS))
node_Sg = model.add_node(NodalFrame, Frame(xS))

beam_props = BeamProperties_EIGJ(EA=112.e8, GA_1=149.3e8, GA_2=149.3e8,
                                 GJ=149.3e4, EI_1=149.3e6, EI_2=149.3e6,
                                 m=432., m_11=11518.2e-3, m_22=5759.1e-3, m_33=5759.1e-3)

discretize_beam(model, node_0, node_M, 5, beam_props)
discretize_beam(model, node_M, node_Bc, 5, beam_props)

discretize_beam(model, node_Br, node_N, 10, beam_props)
discretize_beam(model, node_N, node_S, 10, beam_props)

hj_props = HingeJointProperties(axis=np.array([0., 0., 1.]))
gound_hinge = model.add_element(hj_props, node_0, element_type=GroundJointElement)
model.add_element(hj_props, node_Bc, node_Br, element_type=KinematicJointElement)
model.add_element(hj_props, node_S, node_Sg, element_type=KinematicJointElement)

pj_props = PrismaticJointProperties(axis=np.array([1., 0., 0.]))
model.add_element(pj_props, node_Sg, element_type=GroundJointElement)

sc_props = ServoConstraintProperties(imposed_rotation)
model.add_element(sc_props, gound_hinge)

logger = Logger('beam_slidercrankTiso', periodicity=2)
logger.add_sensor(SensorNode(node_0, LogNodalFields.MOTION))
logger.add_sensor(SensorNode(node_M, LogNodalFields.MOTION))
logger.add_sensor(SensorNode(node_Br, LogNodalFields.MOTION))
logger.add_sensor(SensorNode(node_N, LogNodalFields.MOTION))

time_integration_parameters = TimeIntegrationParameters()
time_integration_parameters.rho = .95
time_integration_parameters.T = 10.
time_integration_parameters.h = 1.e-3
integrator = GeneralizedAlpha(model, time_integration_parameters, logger)

integrator.solve()
