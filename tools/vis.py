from matplotlib import pyplot as plt
import numpy as np


def left2right(kp):
    kp[:, :, 2] = -1 * kp[:, :, 2]
    return kp


def preprocess(kpts):
    kpts[:, 0] = -1 * kpts[:, 0]
    kpts[:, 1] = -1 * kpts[:, 1]
    return kpts


def axis_trasform(kpts):
    kpts[:, :, 0] = -1 * kpts[:, :, 0]
    kpts[:, :, 1] = -1 * kpts[:, :, 1]
    return kpts


def root_algin_np(keypoints):
    root = np.expand_dims(keypoints[:, 0, :], 1)
    keypoints = keypoints - root
    return keypoints


kintree = np.array(
    [[7, 6], [6, 8], [6, 5], [15, 13], [13, 11], [11, 9], [9, 5], [5, 10], [10, 12], [12, 14], [14, 16], [5, 4], [4, 3],
     [3, 2], [2, 1], [17, 1],
     [1, 18], [17, 19], [18, 20], [19, 21], [20, 22], [21, 23], [22, 24]])
plt.ion()
fig = plt.figure()
ax = fig.gca(projection='3d')
points = np.load("./data1.npy")
points = points[:24]
x = points[:, 0]
y = points[:, 1]
z = points[:, 2]
for i in range(len(points)):
    xd = x[i]
    yd = y[i]
    zd = z[i]
    ax.scatter(xd, yd, zd)
    ax.text(xd - 0.01, yd + 0.01, zd + 0.02, '{}'.format(i))
plt.ioff()
plt.show()
