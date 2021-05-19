import h5py
import numpy as np
import matplotlib.pyplot as plt


file_name = "beam_rotatingshaft.h5"
file = h5py.File(file_name)

node_mid_motion = file["node_mid/MOTION"]
node_mid_velocity = file["node_mid/VELOCITY"]
xyz = np.array(node_mid_motion[:3, :])
v_xyz = np.array(node_mid_velocity[:3, :])


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
ax1.plot(xyz[1, :], xyz[2, :])
ax1.set_xlabel(r'y-displacement [m]', rotation='horizontal')
ax1.set_ylabel(r'z-displacement [m]', rotation='vertical')
ax1.grid()

ax2.plot(xyz[1, :], v_xyz[1, :])
ax2.set_xlabel(r'y-displacement [m]', rotation='horizontal')
ax2.set_ylabel(r'y-velocity [m/s]', rotation='vertical')
ax2.grid()

ax3.plot(xyz[2, :], v_xyz[2, :])
ax3.set_xlabel(r'z-displacement [m]', rotation='horizontal')
ax3.set_ylabel(r'z-velocity [m/s]', rotation='vertical')
ax3.grid()

plt.tight_layout()
plt.show()
