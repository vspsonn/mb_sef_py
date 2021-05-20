import h5py
import numpy as np
import matplotlib.pyplot as plt

from mb_sef_py.math.SO3 import tilde

file_name = "beam_slidercrankTiso.h5"
file = h5py.File(file_name)

time = np.array(file["time"]).reshape(-1, 1)
motion_O = file["node_0/MOTION"]
motion_M = file["node_M/MOTION"]
motion_B = file["node_Br/MOTION"]
motion_N = file["node_N/MOTION"]

xyz_O = np.array(motion_O[:3, :])
xyz_M = np.array(motion_M[:3, :])
xyz_B = np.array(motion_B[:3, :])
xyz_N = np.array(motion_N[:3, :])

e = motion_O[3:, 0]
R0 = np.eye(3) + 2*e[0]*tilde(e[1:]) + 2*np.matmul(tilde(e[1:]), tilde(e[1:]))

Defo_M = []
Defo_N = []
for n in range(len(time)):
    e = motion_O[3:, n]
    R = np.eye(3) + 2*e[0]*tilde(e[1:]) + 2*np.matmul(tilde(e[1:]), tilde(e[1:]))
    x_rb = xyz_O[:, n] + np.matmul(np.matmul(R, np.transpose(R0)), xyz_M[:, 0] - xyz_O[:, 0])
    defo = np.matmul(np.transpose(R), xyz_M[:, n] - x_rb)
    Defo_M.append(defo)

    e = motion_B[3:, n]
    R = np.eye(3) + 2*e[0]*tilde(e[1:]) + 2*np.matmul(tilde(e[1:]), tilde(e[1:]))
    x_rb = xyz_B[:, n] + np.matmul(np.matmul(R, np.transpose(R0)), xyz_N[:, 0] - xyz_B[:, 0])
    defo = np.matmul(np.transpose(R), xyz_N[:, n] - x_rb)
    Defo_N.append(defo)

Defo_M = np.array(Defo_M)
Defo_N = np.array(Defo_N)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.plot(time, Defo_M[:, 1])
ax1.set_xlabel(r'Time [s]', rotation='horizontal')
ax1.set_ylabel(r'Deformation mid point link [m]', rotation='vertical')
ax1.grid()

ax2.plot(time, Defo_N[:, 1])
ax2.set_xlabel(r'Time [s]', rotation='horizontal')
ax2.set_ylabel(r'Deformation mid point cranck [m]', rotation='vertical')
ax2.grid()

plt.tight_layout()
plt.show()
