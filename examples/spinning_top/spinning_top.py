import numpy as np

from mb_sef_py.core import Model, NodalFrame
from mb_sef_py.elements import GroundJointElement, SphericalJointProperties, \
    RigidBodyProperties, ExternalForceProperties, RigidLinkProperties
from mb_sef_py.math import Frame
from mb_sef_py.solvers import TimeIntegrationParameters, GeneralizedAlpha
from mb_sef_py.utils import Logger, SensorNode, LogNodalFields

model = Model()

p_root = np.array([0., 0., 0.])
p_tip = np.array([0., 1., 0.])

node_0 = model.add_node(NodalFrame, Frame(x=p_root))
node_1 = model.add_node(NodalFrame, Frame(x=p_tip))

node_0.set_initial_velocity(np.array([0., 0., 0., 0., 150, -4.61538]))
rel_frame = node_0.frame_ref.get_inverse() * node_1.frame_ref
node_1.set_initial_velocity(np.matmul(rel_frame.get_inverse_adjoint(), node_0.v0))

sj = SphericalJointProperties()
model.add_element(sj, node_0, element_type=GroundJointElement)

rl_props = RigidLinkProperties()
model.add_element(rl_props, node_0, node_1)

rb_props = RigidBodyProperties(m=15., J=np.diag([0.234375, 0.46875, 0.234375]))
model.add_element(rb_props, node_1)

ef_props = ExternalForceProperties()
ef_props.forces[2] = -9.81 * 15.
model.add_element(ef_props, node_1)

logger = Logger('spinning_top', periodicity=1)
logger.add_sensor(SensorNode(node_1, LogNodalFields.MOTION))
logger.add_sensor(SensorNode(node_1, LogNodalFields.VELOCITY))

time_integration_parameters = TimeIntegrationParameters()
time_integration_parameters.rho = .85
time_integration_parameters.T = 2.
time_integration_parameters.h = 2.e-3
time_integration_parameters.nit_max = 10
integrator = GeneralizedAlpha(model, time_integration_parameters, logger)

integrator.solve()
