import matplotlib.pyplot as plt
import numpy as np
import os

points_advanced = [
    "pelvis", "spine1", "spine2", "spine3", "neck",
    "left_collar", "right_collar", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hand_thumb4", "right_hand_thumb4",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_foot", "right_foot",
    "head", "head_nose"
]

points = [
    "pelvis", "spine1", "spine2", "spine3", "neck",
    "left_collar", "right_collar", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_foot", "right_foot",
    "head"
]

connections_advanced = [
    ("pelvis", "spine1"), ("spine1", "spine2"), ("spine2", "spine3"), ("spine3", "neck"),
    ("neck", "head"),
    ("neck", "left_shoulder"), ("neck", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"), ("left_wrist", "left_hand_thumb4"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"), ("right_wrist", "right_hand_thumb4"),
    ("pelvis", "left_hip"), ("pelvis", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"), ("left_ankle", "left_foot"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"), ("right_ankle", "right_foot")
]

connections = [
    ("pelvis", "spine1"), ("spine1", "spine2"), ("spine2", "spine3"), ("spine3", "neck"),
    ("neck", "head"),
    ("neck", "left_shoulder"), ("neck", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("pelvis", "left_hip"), ("pelvis", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"), ("left_ankle", "left_foot"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"), ("right_ankle", "right_foot")
]

views = [
    (0, 0),      # front
    (0, 90),     # side
    (0, 180),    # back
    (0, 270),    # opposite side
    (45, 45),    # 3D diagonal view
    (90, 0),     # top-down
]

def save_transformed_dict_plotted(skeleton_dict, output_filename="plot1", L=10.0, dpi=150):
    output_dir = output_filename
    os.makedirs(output_dir, exist_ok=True)

    # Keep only joints that exist in the dictionary
    available_points = [p for p in points if p in skeleton_dict]

    if len(available_points) == 0:
        raise ValueError("No valid skeleton points found in skeleton_dict.")

    coords = np.array([skeleton_dict[p] for p in available_points], dtype=float)

    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    
    xlim = ylim = zlim = [-L, L]

    for elev, azim in views:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")

        # Orthographic projection is better for pose comparison
        ax.set_proj_type("ortho")

        # Points on one side of Y=0
        mask_front = y <= 0
        mask_back = y > 0

        ax.scatter(
            x[mask_front],
            y[mask_front],
            z[mask_front],
            c="r",
            s=12,
            label="Y <= 0"
        )

        ax.scatter(
            x[mask_back],
            y[mask_back],
            z[mask_back],
            c="gray",
            s=12,
            alpha=0.25,
            label="Y > 0"
        )

        # Skeleton connections
        for start, end in connections:
            if start not in skeleton_dict or end not in skeleton_dict:
                continue

            start_xyz = skeleton_dict[start]
            end_xyz = skeleton_dict[end]

            ax.plot(
                [start_xyz[0], end_xyz[0]],
                [start_xyz[1], end_xyz[1]],
                [start_xyz[2], end_xyz[2]],
                color="blue",
                linewidth=1.5
            )

        # Coordinate axes
        ax.plot([-L, L], [0, 0], [0, 0], color="k", linewidth=1)
        ax.plot([0, 0], [-L, L], [0, 0], color="k", linewidth=1)
        ax.plot([0, 0], [0, 0], [-L, L], color="k", linewidth=1)

        # XZ plane where Y = 0
        X, Z = np.meshgrid([-L, L], [-L, L])
        Y = np.zeros_like(X, dtype=float)

        ax.plot_surface(
            X,
            Y,
            Z,
            color="cyan",
            alpha=0.15,
            shade=False
        )

        # Axis labels near the positive directions
        ax.text(0.55 * L, 0, 0, "X", color="k", size=10)
        ax.text(0, 0.55 * L, 0, "Y", color="k", size=10)
        ax.text(0, 0, 0.55 * L, "Z", color="k", size=10)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)

        # Important: equal 3D aspect ratio
        ax.set_box_aspect([1, 1, 1])

        ax.view_init(elev=elev, azim=azim)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.grid(False)

        output_path = os.path.join(
            output_dir,
            f"view_{elev}_{azim}.png"
        )

        plt.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.05
        )

        plt.close(fig)