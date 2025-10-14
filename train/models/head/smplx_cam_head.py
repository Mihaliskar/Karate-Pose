import torch
import torch.nn as nn

from .smplx_local import SMPLX
from ...core import config


class SMPLXCamHeadBodyOnly(nn.Module):
    def __init__(self, img_res=224):

        super(SMPLXCamHead, self).__init__()
        self.smplx = SMPLX(config.SMPLX_MODEL_DIR, num_betas=11)
        self.add_module('smplx', self.smplx)
        self.img_res = img_res

    def forward(self, rotmat, shape, cam, cam_intrinsics,
                bbox_scale, bbox_center, img_w, img_h,
                normalize_joints2d=False, trans=False, trans2=False,
                learned_scale=None):

        smpl_output = self.smplx(
            betas=shape,
            body_pose=rotmat[:, 1:22].contiguous(),
            global_orient=rotmat[:, 0].unsqueeze(1).contiguous(),
            pose2rot=False,
        ) 

        output = {
            'vertices': smpl_output.vertices,
            'joints3d': smpl_output.joints,
        }

        joints3d = output['joints3d']
        batch_size = joints3d.shape[0]
        device = joints3d.device

        cam_t = convert_pare_to_full_img_cam(
            pare_cam=cam,
            bbox_height=bbox_scale * 200.,
            bbox_center=bbox_center,
            img_w=img_w,
            img_h=img_h,
            focal_length=cam_intrinsics[:, 0, 0],
            crop_res=self.img_res,
        )

        joints2d = perspective_projection(
            joints3d,
            rotation=torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1),
            translation=cam_t,
            cam_intrinsics=cam_intrinsics,
        )
        if normalize_joints2d:
            # Normalize keypoints to [-1,1]
            joints2d = joints2d / (self.img_res / 2.)
        
        output['joints2d'] = joints2d
        output['pred_cam_t'] = cam_t

        return output





class SMPLXCamHead(nn.Module):
    def __init__(self, img_res=224):
        super(SMPLXCamHead, self).__init__()
        self.smplx = SMPLX(config.SMPLX_MODEL_DIR, num_betas=11)
        self.add_module('smplx', self.smplx)
        self.img_res = img_res

    def forward(self, rotmat, shape, cam, cam_intrinsics,
                bbox_scale, bbox_center, img_w, img_h,
                normalize_joints2d=False):

        # --- Forward SMPL-X ---
        smpl_output = self.smplx(
            betas=shape,
            body_pose=rotmat[:, 1:22].contiguous(),
            global_orient=rotmat[:, 0].unsqueeze(1).contiguous(),
            pose2rot=False,
        )

        all_joints = smpl_output.joints  # shape (B, ~118, 3)
        vertices = smpl_output.vertices  # shape (B, N, 3)

        # --- Split into subsets ---
        # These index ranges are based on standard SMPL-X ordering
        joints_body = all_joints[:, :22, :]
        joints_left_hand = all_joints[:, 22:37, :]
        joints_right_hand = all_joints[:, 37:52, :]
        joints_head = all_joints[:, 52:, :]  # face/eyes/jaw region

        # --- Camera projection setup ---
        batch_size = all_joints.shape[0]
        device = all_joints.device

        cam_t = convert_pare_to_full_img_cam(
            pare_cam=cam,
            bbox_height=bbox_scale * 200.,
            bbox_center=bbox_center,
            img_w=img_w,
            img_h=img_h,
            focal_length=cam_intrinsics[:, 0, 0],
            crop_res=self.img_res,
        )

        # --- Project each group separately ---
        joints2d_body = perspective_projection(
            joints_body,
            rotation=torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1),
            translation=cam_t,
            cam_intrinsics=cam_intrinsics,
        )

        joints2d_left_hand = perspective_projection(
            joints_left_hand,
            rotation=torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1),
            translation=cam_t,
            cam_intrinsics=cam_intrinsics,
        )

        joints2d_right_hand = perspective_projection(
            joints_right_hand,
            rotation=torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1),
            translation=cam_t,
            cam_intrinsics=cam_intrinsics,
        )

        joints2d_head = perspective_projection(
            joints_head,
            rotation=torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1),
            translation=cam_t,
            cam_intrinsics=cam_intrinsics,
        )

        if normalize_joints2d:
            joints2d_body = joints2d_body / (self.img_res / 2.)
            joints2d_left_hand = joints2d_left_hand / (self.img_res / 2.)
            joints2d_right_hand = joints2d_right_hand / (self.img_res / 2.)
            joints2d_head = joints2d_head / (self.img_res / 2.)

        # --- Prepare output dict ---
        output = {
            # legacy outputs
            'vertices': vertices,
            'joints3d': joints_body,
            'joints2d': joints2d_body,
            'pred_cam_t': cam_t,

            # explicit hands & head
            'left_hand_3d': joints_left_hand,
            'right_hand_3d': joints_right_hand,
            'head_3d': joints_head,

            'left_hand_2d': joints2d_left_hand,
            'right_hand_2d': joints2d_right_hand,
            'head_2d': joints2d_head,
        }

        return output





def perspective_projection(points, rotation, translation, cam_intrinsics):

    K = cam_intrinsics
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)
    projected_points = points / points[:, :, -1].unsqueeze(-1)
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points.float())
    return projected_points[:, :, :-1]


def convert_pare_to_full_img_cam(
        pare_cam, bbox_height, bbox_center,
        img_w, img_h, focal_length, crop_res=224):

    s, tx, ty = pare_cam[:, 0], pare_cam[:, 1], pare_cam[:, 2]
    res = 224
    r = bbox_height / res
    tz = 2 * focal_length / (r * res * s)

    cx = 2 * (bbox_center[:, 0] - (img_w / 2.)) / (s * bbox_height)
    cy = 2 * (bbox_center[:, 1] - (img_h / 2.)) / (s * bbox_height)

    cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)

    return cam_t
