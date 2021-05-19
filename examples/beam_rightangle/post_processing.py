import h5py
import numpy as np
import matplotlib.pyplot as plt


file_name = "beam_rightangle.h5"
file = h5py.File(file_name)

time = np.array(file["time"]).reshape(-1, 1)
motion_1 = file["mid node/MOTION"]
motion_2 = file["tip node/MOTION"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.plot(time, motion_1[2, :], 'b-')
ax1.set_xlabel(r'Time [s]', rotation='horizontal')
ax1.set_ylabel(r'mid node out-of-place displacement [m]', rotation='vertical')
ax1.grid()

ax2.plot(time, motion_2[2, :], 'b-')
ax2.set_xlabel(r'Time [s]', rotation='horizontal')
ax2.set_ylabel(r'tip node out-of-plane displacement [m]', rotation='vertical')
ax2.grid()

plt.tight_layout()
plt.show()