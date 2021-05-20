import numpy as np

from mb_sef_py.core import Model, NodalFrame
from mb_sef_py.elements import BeamProperties_EIGJ, discretize_beam, ClampedFrameProperties, RigidLinkProperties, \
    ExternalForceProperties
from mb_sef_py.math import Frame, UnitQuaternion
from mb_sef_py.solvers import TimeIntegrationParameters, GeneralizedAlpha
from mb_sef_py.utils import Logger, SensorNode, LogNodalFields


def loading(t):
    loads = np.zeros((6, ))
    if t > 2.:
        loads[2] = 0.
    elif t < 1.:
        loads[2] = 50. * t
    else:
        loads[2] = 50. * (2. - t)
    return loads


model = Model()

p0 = np.array([0., 0., 0.])
p1 = np.array([10., 0., 0.])
p2 = np.array([10., 10., 0.])

q_2 = UnitQuaternion()
q_2.set_from_triad(np.array([0., 1., 0.]), np.array([-1., 0., 0.]))

node_0 = model.add_node(NodalFrame, Frame(x=p0))
node_1_1 = model.add_node(NodalFrame, Frame(x=p1), 'mid node')
node_1_2 = model.add_node(NodalFrame, Frame(x=p1, q=q_2))
node_2 = model.add_node(NodalFrame, Frame(x=p2, q=q_2), 'tip node')


cl_props = ClampedFrameProperties()
model.add_element(cl_props, node_0)

rl_props = RigidLinkProperties()
model.add_element(rl_props, node_1_1, node_1_2)

beam_props = BeamProperties_EIGJ(EA=1.e6, GA_1=1.e6, GA_2=1.e6,
                                 GJ=1.e3, EI_1=1.e3, EI_2=1.e3,
                                 m=1., m_11=20., m_22=10., m_33=10.)
number_of_element = 10
discretize_beam(model, node_0, node_1_1, number_of_element, beam_props)
discretize_beam(model, node_1_2, node_2, number_of_element, beam_props)

ef_props = ExternalForceProperties()
ef_props.time_dependent_force = loading
model.add_element(ef_props, node_1_1)

logger = Logger('beam_rightangle', periodicity=3)
logger.add_sensor(SensorNode(node_1_1, LogNodalFields.MOTION))
logger.add_sensor(SensorNode(node_2, LogNodalFields.MOTION))

time_integration_parameters = TimeIntegrationParameters()
time_integration_parameters.rho = .95
time_integration_parameters.T = 30.
time_integration_parameters.h = 1.e-2
integrator = GeneralizedAlpha(model, time_integration_parameters, logger)

integrator.solve()
