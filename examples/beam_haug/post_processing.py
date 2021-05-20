import h5py
import numpy as np
import matplotlib.pyplot as plt
from mb_sef_py.math.SE3 import tilde


file_name = "beam_haug.h5"
file = h5py.File(file_name)

time = np.array(file["time"]).reshape(-1, 1)
motion_0 = file["node_root/MOTION"]
motion_1 = file["node_tip/MOTION"]

xyz_0 = np.array(motion_0[:3, :])
xyz_1 = np.array(motion_1[:3, :])

e = motion_0[3:, 0]
R0 = np.eye(3) + 2*e[0]*tilde(e[1:]) + 2*np.matmul(tilde(e[1:]), tilde(e[1:]))

Defo = []
for n in range(len(time)):
    e = motion_0[3:, n]
    R = np.eye(3) + 2*e[0]*tilde(e[1:]) + 2*np.matmul(tilde(e[1:]), tilde(e[1:]))
    x_rb = xyz_0[:, n] + np.matmul(np.matmul(R, np.transpose(R0)), xyz_1[:, 0] - xyz_0[:, 0])
    defo = np.matmul(np.transpose(R), xyz_1[:, n] - x_rb)
    Defo.append(defo)
Defo = np.array(Defo)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.plot(time, Defo[:, 0], 'ro-')
ax1.set_xlabel(r'Time [s]', rotation='horizontal')
ax1.set_ylabel(r'Axial tip displacement [m]', rotation='vertical')
ax1.grid()

ax2.plot(time, Defo[:, 1], 'b-')
ax2.set_xlabel(r'Time [s]', rotation='horizontal')
ax2.set_ylabel(r'Transversal tip displacement [m]', rotation='vertical')
ax2.grid()

plt.tight_layout()
plt.show()
