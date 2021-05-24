import h5py
import numpy as np
import matplotlib.pyplot as plt


file_name = "spinning_top.h5"
file = h5py.File(file_name)

time = np.array(file["time"]).reshape(-1, 1)
energy = np.array(file["mechanical_power"]).reshape(-1, 1)
number_of_iterations = np.array(file["number_of_iterations"]).reshape(-1, 1)

_, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.plot(time, energy)
ax1.set_xlabel(r'Time [s]', rotation='horizontal')
ax1.set_ylabel(r'Mechanical power [J/s]', rotation='vertical')
ax1.grid()

ax2.plot(time, number_of_iterations)
ax2.set_xlabel(r'Time [s]', rotation='horizontal')
ax2.set_ylabel(r'number of iterations [-]', rotation='vertical')
ax2.grid()


motion_1 = file["node_1/MOTION"]
velocity_1 = file["node_1/VELOCITY"]
xyz = np.array(motion_1[:3, :])
v_xyz = np.array(velocity_1[:3, :])

_, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.plot(time, xyz[0, :] - xyz[0, 0], 'b-')
ax1.plot(time, xyz[1, :] - xyz[1, 0], 'g-')
ax1.set_xlabel(r'Time [s]', rotation='horizontal')
ax1.set_ylabel(r'xy-displacements [m]', rotation='vertical')
ax1.grid()

ax2.plot(time, xyz[2, :] - xyz[2, 0], 'r-')
ax2.set_xlabel(r'Time [s]', rotation='horizontal')
ax2.set_ylabel(r'z-displacement [m]', rotation='vertical')
ax2.grid()

_, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.plot(time, v_xyz[0, :], 'b-')
ax1.plot(time, v_xyz[1, :], 'g-')
ax1.set_xlabel(r'Time [s]', rotation='horizontal')
ax1.set_ylabel(r'xy-velocities [m/s]', rotation='vertical')
ax1.grid()

ax2.plot(time, v_xyz[2, :], 'r-')
ax2.set_xlabel(r'Time [s]', rotation='horizontal')
ax2.set_ylabel(r'z-velocity [m/s]', rotation='vertical')
ax2.grid()

plt.tight_layout()
plt.show()
