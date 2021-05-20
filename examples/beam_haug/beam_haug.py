import numpy as np

from mb_sef_py.core import Model, NodalFrame
from mb_sef_py.elements import BeamProperties_EIGJ, discretize_beam, \
    GroundJointElement, HingeJointProperties, ServoConstraintProperties
from mb_sef_py.math import Frame
from mb_sef_py.solvers import TimeIntegrationParameters, GeneralizedAlpha
from mb_sef_py.utils import Logger, SensorNode, LogNodalFields


def imposed_rotation(t):
    T, omega = 15., 4.
    tau = t/T
    if tau < 1.:
        phi = 0.5 * tau * tau + (np.cos(2. * np.pi * tau) - 1.) / (2. * np.pi)**2
    else:
        phi = tau - 0.5
    return omega * T * phi


model = Model()

frame_root = Frame()
node_root = model.add_node(NodalFrame, frame_root, 'node_root')
frame_tip = Frame(x=np.array([8., 0., 0.]))
node_tip = model.add_node(NodalFrame, frame_tip, 'node_tip')


beam_props = BeamProperties_EIGJ(EA=5.03e6, GA_1=1.94e6, GA_2=1.94e6,
                                 GJ=566, EI_1=566, EI_2=566,
                                 m=0.201, m_11=2 * 22.7e-6, m_22=22.7e-6, m_33=22.7e-6)
discretize_beam(model, node_root, node_tip, 10, beam_props)

hj_props = HingeJointProperties(axis=np.array([0., 0., 1.]))
hinge = model.add_element(hj_props, node_root, element_type=GroundJointElement)

sc_props = ServoConstraintProperties(imposed_rotation)
model.add_element(sc_props, hinge)


logger = Logger('beam_haug', periodicity=2)
logger.add_sensor(SensorNode(node_root, LogNodalFields.MOTION))
logger.add_sensor(SensorNode(node_tip, LogNodalFields.MOTION))


time_integration_parameters = TimeIntegrationParameters()
time_integration_parameters.rho = .0
time_integration_parameters.T = 20.
time_integration_parameters.h = 2.e-3
integrator = GeneralizedAlpha(model, time_integration_parameters, logger)

integrator.solve()
