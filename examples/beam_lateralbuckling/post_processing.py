import h5py
import numpy as np
import matplotlib.pyplot as plt


file_name = "beam_lateralbuckling.h5"
file = h5py.File(file_name)

time = np.array(file["time"]).reshape(-1, 1)
motion_1 = np.array(file["node_1/MOTION"])
velocity_1 = np.array(file["node_1/VELOCITY"])

xyz = motion_1[:3, :]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
ax1.plot(time, xyz[0, :] - xyz[0, 0], 'b-')
ax1.plot(time, xyz[1, :] - xyz[1, 0], 'r-')
ax1.set_xlabel(r'Time [s]', rotation='horizontal')
ax1.set_ylabel(r'Mid point displacement [m]', rotation='vertical')
ax1.grid()

ax2.plot(time, velocity_1[3, :])
ax2.set_xlabel(r'Time [s]', rotation='horizontal')
ax2.set_ylabel(r'Mid point velocity [m/s]', rotation='vertical')
ax2.grid()

ax3.plot(time, 0.5*np.arccos(motion_1[3, :]) * 180 / np.pi)
ax3.set_xlabel(r'Time [s]', rotation='horizontal')
ax3.set_ylabel(r'Moment profile [-]', rotation='vertical')
ax3.grid()

plt.tight_layout()
plt.show()
