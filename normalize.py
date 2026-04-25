import numpy as np

def normalize_pose(pose):
    new_pose = normalize_position(pose)
    new_pose = normalize_scale(new_pose)
    new_pose = normalize_rotation(new_pose)
    return new_pose

def normalize_position(pose):
    pelvis = np.asarray(pose["pelvis"], dtype=float)

    centered = {k: np.asarray(v, dtype=float) - pelvis for k, v in pose.items()}

    return centered

def normalize_scale(pose, eps=1e-8, target_hip_width=1.0):
    s = np.linalg.norm(pose["right_hip"] - pose["left_hip"])
    if s < eps:
        raise ValueError("Scale too small; bad keypoints.")
    scale = target_hip_width / s
    scaled = {k: (p * scale).tolist() for k, p in pose.items()}

    return scaled

def get_unit_vector(v, eps=1e-8):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)

    if n < eps:
        raise ValueError("Cannot normalize near-zero vector")

    return v / n

def get_skeleton_front(pose):
    left_collar = np.asarray(pose["left_collar"], dtype=float)
    right_collar = np.asarray(pose["right_collar"], dtype=float)
    neck = np.asarray(pose["neck"], dtype=float)


    left_line = (left_collar - neck)
    right_line = (right_collar - neck)

    return get_unit_vector(np.cross(left_line, right_line))

def get_skeleton_up(pose):
    pelvis = np.asarray(pose["pelvis"], dtype=float)
    neck = np.asarray(pose["neck"], dtype=float)

    return get_unit_vector(neck-pelvis)

def get_skeleton_right(pose):
    left_shoulder = np.asarray(pose["left_shoulder"], dtype=float)
    right_shoulder = np.asarray(pose["right_shoulder"], dtype=float)

    return get_unit_vector(right_shoulder - left_shoulder)

def get_skeleton(pose):
    skeleton_front = get_skeleton_front(pose)
    skeleton_up = get_skeleton_up(pose)
    skeleton_right = get_skeleton_right(pose)

    skeleton_basis = np.column_stack([
    skeleton_right,
    skeleton_up,
    skeleton_front,
    ])

    return skeleton_basis


def normalize_rotation(pose):
    target_basis = np.eye(3)
    skeleton_basis = get_skeleton(pose)
    
    rotation = target_basis @ skeleton_basis.T

    origin = np.asarray(pose["pelvis"], dtype=float)

    rotated_pose = {}

    for joint_name, position in pose.items():
        p = np.asarray(position, dtype=float)

        rotated = rotation @ (p - origin)

        rotated = rotated + origin

        rotated_pose[joint_name] = rotated.tolist()

    return rotated_pose

    
