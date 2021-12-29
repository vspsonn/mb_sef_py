import h5py
import numpy as np
import matplotlib.pyplot as plt
import csv

file_name = "beam_cclamped.h5"
file = h5py.File(file_name)

time = np.array(file["time"]).reshape(-1, 1)
motion_3 = file["node_16/MOTION"]
xyz = np.array(motion_3[:3, :])

with open('results.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    for i in range(len(time)):
        writer.writerow([time[i,0], xyz[2,i]])

plt.plot(time, xyz[2,:])
plt.xlabel('time (s)')
plt.ylabel('Z (m)')
plt.title('Beam Midpoint Displacement')
plt.grid(True)
plt.tight_layout()
plt.show()
