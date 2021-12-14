import numpy as np

from mb_sef_py.core import Model, NodalFrame
from mb_sef_py.elements import BeamProperties_EIGJ, discretize_beam, HingeJointProperties, GroundJointElement
from mb_sef_py.math import Frame, UnitQuaternion
from mb_sef_py.solvers import TimeIntegrationParameters, GeneralizedAlpha
from mb_sef_py.utils import Logger, SensorNode, LogNodalFields

model = Model()

E, nu = 5.e6, 0.5
G = E/(2 * (1. + nu))
rho = 1.1e3
r = 5.e-3
A, I = np.pi * r**2, 0.25 * np.pi * r**4

beam_props = BeamProperties_EIGJ(EA=E*A, GA_1=G*A, GA_2=G*A, GJ=G*2*I, EI_1=E*I, EI_2=E*I,
                                 m=rho*A, m_11=rho*2*I, m_22=rho*I, m_33=rho*I)
beam_props.gravity = np.array([0., -9.81, 0.])


p_root = np.array([0., 0., 0.])
p_tip = np.array([1., 0., 0.])

# f_0 = Frame(x=p_root)
# node_0 = model.add_node(NodalFrame, f_0)
# f_1 = Frame(x=p_tip)
# node_1 = model.add_node(NodalFrame, f_1)

theta = 1.e-18 * np.pi
n_0 = np.array([0., np.cos(theta), np.sin(theta)])
b_0 = np.array([0., -np.sin(theta), np.cos(theta)])
n_0 = np.array([0., 1., 0.])
b_0 = np.array([0., 0., 1.])
q_0 = UnitQuaternion()
q_0.set_from_triad(np.array([1., 0., 0.]), n_0, b_0)
q_1 = UnitQuaternion()
q_1.set_from_triad(np.array([1., 0., 0.]), n_0, b_0)
f_0 = Frame(x=p_root, q=q_0)
node_0 = model.add_node(NodalFrame, f_0)
f_1 = Frame(x=p_tip, q=q_1)
node_1 = model.add_node(NodalFrame, f_1)

discretize_beam(model, node_0, node_1, 10, beam_props)

hj_props = HingeJointProperties(axis=np.array([0., 0., 1.]))
model.add_element(hj_props, node_0, element_type=GroundJointElement)

logger = Logger('beam_flexible_pendulum', periodicity=2)
logger.add_sensor(SensorNode(node_1, LogNodalFields.MOTION))

time_integration_parameters = TimeIntegrationParameters()
time_integration_parameters.rho = .0
time_integration_parameters.T = 1.
time_integration_parameters.h = 1.e-3
time_integration_parameters.tol_res_forces = 1.e-6

integrator = GeneralizedAlpha(model, time_integration_parameters, logger)
integrator.solve()
