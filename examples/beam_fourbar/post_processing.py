import h5py
import numpy as np
import matplotlib.pyplot as plt


file_name = "beam_fourbar.h5"
file = h5py.File(file_name)

time = np.array(file["time"]).reshape(-1, 1)
motion_3 = file["node_3/MOTION"]

xyz = np.array(motion_3[:3, :])


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.plot(time, xyz[0, :] - xyz[0, 0], 'b-')
ax1.plot(time, xyz[1, :] - xyz[1, 0], 'g-')
ax1.set_xlabel(r'Time [s]', rotation='horizontal')
ax1.set_ylabel(r'In-plane displacement [m]', rotation='vertical')
ax1.grid()

ax2.plot(time, xyz[2, :], 'r-')
ax2.set_xlabel(r'Time [s]', rotation='horizontal')
ax2.set_ylabel(r'Out-of-plane displacement [m]', rotation='vertical')
ax2.grid()

plt.tight_layout()
plt.show()
