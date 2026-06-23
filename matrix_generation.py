import numpy as np

def generate_edm(pose):
    joint_order = list(pose.keys())

    points = np.array([pose[joint] for joint in joint_order], dtype=float)

    diffs = points[:, None, :] - points[None, :, :]

    edm = np.linalg.norm(diffs, axis=-1)

    return edm

