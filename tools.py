
def get_azure_kinect_skeleton():
skeleton = [
    "pelvis",          # 0
    "spine1",          # 1
    "spine2",          # 2
    "spine3",          # 3
    "neck",            # 4
    "head",            # 5
    "right_clavicle",  # 6
    "right_shoulder",  # 7
    "right_elbow",     # 8
    "right_wrist",     # 9
    "right_hand",      # 10
    "left_clavicle",   # 11
    "left_shoulder",   # 12
    "left_elbow",      # 13
    "left_wrist",      # 14
    "left_hand",       # 15
    "left_hand_tip",   # 16
    "right_hand_tip",  # 17
    "spine4",          # 18
    "right_hip",       # 19
    "right_knee",      # 20
    "right_ankle",     # 21
    "left_hip",        # 22
    "left_knee",       # 23
    "left_ankle",      # 24
    "left_foot",       # 25
    "right_foot",      # 26
    "jaw",             # 27
    "nose",            # 28
    "right_eye",       # 29
    "left_eye",        # 30
    "head_top"         # 31
]
return skeleton


"""
Select joints that we want from SMPL skeleton
"""
def get_smpl_skeleton():
     return np.array(
                                [
                                    [ 0, 1 ],
                                    [ 0, 2 ],
                                    [ 0, 3 ],
                                    [ 1, 4 ],
                                    [ 2, 5 ],
                                    [ 3, 6 ],
                                    [ 4, 7 ],
                                    [ 5, 8 ],
                                    [ 6, 9 ],
                                    [ 7, 10],
                                    [ 8, 11],
                                    [ 9, 12],
                                    [ 9, 13],
                                    [ 9, 14],
                                    [12, 15],
                                    [13, 16],
                                    [14, 17],
                                    [16, 18],
                                    [17, 19],
                                    [18, 20],
                                    [19, 21],

                                ]
                            )



def get_colors():
     colors = {
                                'pink': np.array([197, 27, 125]),  # L lower leg
                                'light_pink': np.array([233, 163, 201]),  # L upper leg
                                'light_green': np.array([161, 215, 106]),  # L lower arm
                                'green': np.array([77, 146, 33]),  # L upper arm
                                'red': np.array([215, 48, 39]),  # head
                                'light_red': np.array([252, 146, 114]),  # head
                                'light_orange': np.array([252, 141, 89]),  # chest
                                'purple': np.array([118, 42, 131]),  # R lower leg
                                'light_purple': np.array([175, 141, 195]),  # R upper
                                'light_blue': np.array([145, 191, 219]),  # R lower arm
                                'blue': np.array([69, 117, 180]),  # R upper arm
                                'gray': np.array([130, 130, 130]),  #
                                'white': np.array([255, 255, 255]),  #
                                'pinkish': np.array([204, 77, 77]),
                            }
     return colors



def saveCSVFileFromListOfDicts(filename,inputDicts):
    labels = list()
    #--------------------------
    for frame in inputDicts:
      for label in frame.keys():
        if not label in labels: 
           labels.append(label)
    #--------------------------   
    f = open(filename, 'w')
    #Write header..
    #------------------------------------------------------------------------  
    for column in range(len(labels)):
      if (column>0):
        f.write(',')
      f.write("%s_3DX,"%(labels[column]))
      f.write("%s_3DY,"%(labels[column]))
      f.write("%s_3DZ" %(labels[column]))
    f.write('\n')
    #------------------------------------------------------------------------  
    #Write body..
    #------------------------------------------------------------------------ 
    for frame in inputDicts: 
     for column in range(len(labels)):
      if (column>0):
        f.write(',')
      if (labels[column] in frame):
         f.write("%f,"%(frame[labels[column]][0]))
         f.write("%f,"%(frame[labels[column]][1]))
         f.write("%f" %(frame[labels[column]][2]))
      else:
         f.write("0")
     f.write('\n')
    #------------------------------------------------------------------------  
    f.close()


def saveCSVFileFromListOfDictsFollowingSkeletonOrder(filename, inputDicts, skeleton):
    """
    Saves a CSV file where joint columns follow the given skeleton order.
    Each element in inputDicts is a dictionary mapping joint names to [x, y, z] coordinates.
    """
    with open(filename, 'w') as f:
        # --- Write header ---
        for i, joint in enumerate(skeleton):
            if i > 0:
                f.write(',')
            f.write(f"{joint}_3DX,{joint}_3DY,{joint}_3DZ")
        f.write('\n')

        # --- Write frame data ---
        for frame in inputDicts:
            for i, joint in enumerate(skeleton):
                if i > 0:
                    f.write(',')
                if joint in frame:
                    coords = frame[joint]
                    f.write(f"{coords[0]:.6f},{coords[1]:.6f},{coords[2]:.6f}")
                else:
                    f.write("0,0,0")
            f.write('\n')




def smpl_to_32joints(hmr_output):
    """
    Converts SMPL (22x3) joints from hmr_output to a 32-joint skeleton format.

    Args:
        hmr_output (dict): Contains 'joints3d' tensor of shape (B, 22, 3)

    Returns:
        np.ndarray: Array of shape (32, 3)
    """
    # Extract 22 joints
    joints_22 = hmr_output['joints3d'][0, :22, :]
    if isinstance(joints_22, torch.Tensor):
        joints_22 = joints_22.cpu().numpy()

    # Initialize 32-joint array
    joints_32 = np.zeros((32, 3))

    # Basic mapping (fill directly from SMPL)
    mapping = {
        0: 0,  1: 3,  2: 6,  3: 9,  4: 12,
        5: 13, 6: 14, 7: 16, 8: 18, 9: 20,
        10: 17, 11: 19, 12: 21,
        15: 1, 16: 2, 17: 4, 18: 5,
        19: 7, 20: 8, 21: 10, 22: 11,
        23: 15
    }

    for target, source in mapping.items():
        joints_32[target] = joints_22[source]

    # Estimate head top (24) by extending the neck-head vector
    neck = joints_32[4]
    head = joints_32[23]
    joints_32[24] = head + 0.25 * (head - neck)

    # Duplicate hands (13, 14)
    joints_32[13] = joints_32[9]
    joints_32[14] = joints_32[12]

    # Fill remaining joints (25–31) as small variations or copies of nearby points
    for i in range(25, 32):
        joints_32[i] = joints_32[24]  # could be refined if you have dataset definitions

    return joints_32



