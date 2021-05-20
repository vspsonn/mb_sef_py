import numpy as np

from mb_sef_py.core import Model, NodalFrame
from mb_sef_py.elements import BeamProperties_EIGJ, discretize_beam, HingeJointProperties, GroundJointElement, \
    KinematicJointElement, ServoConstraintProperties
from mb_sef_py.math import UnitQuaternion, Frame
from mb_sef_py.solvers import TimeIntegrationParameters, GeneralizedAlpha
from mb_sef_py.utils import Logger, SensorNode, LogNodalFields


model = Model()

beam12_props = BeamProperties_EIGJ(EA=52.99e6, GA_1=16.88e6, GA_2=16.88e6,
                                   GJ=733.5, EI_1=1131, EI_2=1131,
                                   m=1.997, m_11=2 * 42.6e-6, m_22=42.6e-6, m_33=42.6e-6)
beam3_props = BeamProperties_EIGJ(EA=13.25e6, GA_1=4.22e6, GA_2=4.22e6,
                                  GJ=45.84, EI_1=70.66, EI_2=70.66,
                                  m=0.4992, m_11=2 * 2.662e-6, m_22=2.662e-6, m_33=2.662e-6)


l_13, l_2 = 0.12,  0.24
p_BL = np.array([0., 0., 0.])
p_TL = np.array([0., l_13, 0.])
p_TR = np.array([l_2, l_13, 0.])
p_BR = np.array([l_2, 0., 0.])

q_1 = UnitQuaternion()
q_1.set_from_triad(np.array([0., 1., 0.]), np.array([-1., 0., 0.]))
q_3 = UnitQuaternion()
q_3.set_from_triad(np.array([0., -1., 0.]), np.array([1., 0., 0.]))

f_0 = Frame(x=p_BL, q=q_1)
node_0 = model.add_node(NodalFrame, f_0)
f_1 = Frame(x=p_TL, q=q_1)
node_1 = model.add_node(NodalFrame, f_1)
f_2 = Frame(x=p_TL)
node_2 = model.add_node(NodalFrame, f_2)
f_3 = Frame(x=p_TR)
node_3 = model.add_node(NodalFrame, f_3)
f_4 = Frame(x=p_TR, q=q_3)
node_4 = model.add_node(NodalFrame, f_4)
f_5 = Frame(x=p_BR, q=q_3)
node_5 = model.add_node(NodalFrame, f_5)


number_of_elements = 8
discretize_beam(model, node_0, node_1, number_of_elements, beam12_props)
discretize_beam(model, node_2, node_3, number_of_elements, beam12_props)
discretize_beam(model, node_4, node_5, number_of_elements, beam3_props)

hj_props = HingeJointProperties(axis=np.array([0., 0., 1.]))
defect = 5. * np.pi/180.
crooked_hj_props = HingeJointProperties(axis=np.array([np.sin(defect), 0., np.cos(defect)]))

hinge = model.add_element(hj_props, node_0, element_type=GroundJointElement)
model.add_element(hj_props, node_1, node_2, element_type=KinematicJointElement)
model.add_element(crooked_hj_props, node_3, node_4, element_type=KinematicJointElement)
model.add_element(hj_props, node_5, element_type=GroundJointElement)

omega = -0.6
sc_props = ServoConstraintProperties(lambda t: omega * t)
model.add_element(sc_props, hinge)


logger = Logger('beam_fourbar', periodicity=2)
logger.add_sensor(SensorNode(node_3, LogNodalFields.MOTION))
logger.add_sensor(SensorNode(node_4, LogNodalFields.MOTION))


time_integration_parameters = TimeIntegrationParameters()
time_integration_parameters.rho = .0
time_integration_parameters.T = 12.
time_integration_parameters.h = 4.e-3
time_integration_parameters.tol_res_forces = 1.e-6
integrator = GeneralizedAlpha(model, time_integration_parameters, logger)

integrator.solve()
