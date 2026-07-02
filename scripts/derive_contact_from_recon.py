#!/usr/bin/env python3
"""Derive a [T,53] contact.pt for RePHO from a VisTracker recon .pkl (in-the-wild path).

The demo ships a BEHAVE ground-truth contact.pt; for in-the-wild video we infer it from
hand-vertex ↔ object-surface proximity. Layout matches the demo exactly:
  col 0        = object-in-contact (any hand touching)
  cols 1:53    = per-body (52, MuJoCo/smpl_2_mujoco order); we set the LEFT-hand block
                 (cols 17..32) and RIGHT-hand block (36..51) as whole-hand blocks, matching
                 how the demo GT marks contact (all 16 hand-joint slots together).

A hand is "in contact" on a frame if the min distance from its SMPL-H hand vertices to the
posed object mesh surface is below --thresh (meters). RePHO treats contact as guidance and
refines it via RL, so a proximity seed is the intended kind of signal.
"""
import argparse
import os

import joblib
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation
import smplx

# MuJoCo-order hand slots (must match interact_to_tracking_format / env left_hand_ids/right_hand_ids)
LEFT_HAND_MUJ = list(range(17, 33))
RIGHT_HAND_MUJ = list(range(36, 52))

# SMPL-H body vertex→hand association: use MANO hand vertex sets via SMPL-H segmentation.
# Simize by using the SMPL-H right/left wrist+hand joints' skinned vertices is complex; instead we
# use hand JOINT positions (wrist + finger joints) as proxies for hand location and measure to obj.
# SMPL-H joint indices: left wrist=20, right wrist=21; left hand fingers 22..36, right 37..51 (SMPLH).
SMPLH_LEFT_HAND_JOINTS = [20] + list(range(22, 37))
SMPLH_RIGHT_HAND_JOINTS = [21] + list(range(37, 52))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon_file", required=True)
    ap.add_argument("--obj_mesh", required=True, help="object .obj (canonical, will be posed by obj_angles/trans)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_path", default="./body_models")
    ap.add_argument("--gender", default="male")
    ap.add_argument("--thresh", type=float, default=0.08, help="hand-to-object contact distance (m)")
    ap.add_argument("--obj_sample", type=int, default=2000)
    args = ap.parse_args()

    d = joblib.load(args.recon_file)
    poses = d["poses"].astype(np.float32)          # (T,156)
    trans = d["trans"].astype(np.float32)          # (T,3)
    betas = d["betas"].astype(np.float32)
    if betas.ndim == 2:
        betas = betas.mean(0)
    obj_angles = d["obj_angles"]                   # (T,3,3) rot matrices
    obj_trans = d["obj_trans"].astype(np.float32)  # (T,3)
    T = poses.shape[0]

    sm = smplx.create(args.model_path, model_type="smplh", gender=args.gender, use_pca=False, ext="pkl")
    out = sm(
        body_pose=torch.from_numpy(poses[:, 3:66]),
        global_orient=torch.from_numpy(poses[:, :3]),
        left_hand_pose=torch.from_numpy(poses[:, 66:111]),
        right_hand_pose=torch.from_numpy(poses[:, 111:156]),
        betas=torch.cat([torch.from_numpy(betas)[None].repeat(T, 1),
                         torch.zeros(T, 16 - betas.shape[0])], 1),
        transl=torch.from_numpy(trans),
    )
    joints = out.joints.detach().numpy()  # (T, J, 3)
    lh = joints[:, SMPLH_LEFT_HAND_JOINTS, :]   # (T, nL, 3)
    rh = joints[:, SMPLH_RIGHT_HAND_JOINTS, :]  # (T, nR, 3)

    mesh = trimesh.load(args.obj_mesh, force="mesh")
    ov = np.asarray(mesh.vertices, np.float32)
    ov = ov - ov.mean(0)  # center like RePHO samples it
    # sample surface points for distance
    pts, _ = trimesh.sample.sample_surface(mesh, args.obj_sample) if len(mesh.faces) else (ov, None)
    pts = np.asarray(pts, np.float32) - np.asarray(mesh.vertices, np.float32).mean(0)

    contact = np.zeros((T, 53), np.float32)
    lcount = rcount = 0
    for t in range(T):
        R = np.asarray(obj_angles[t], np.float64)
        op = (pts @ R.T) + obj_trans[t]  # posed object surface points (T,N,3)
        dl = np.linalg.norm(lh[t][:, None, :] - op[None, :, :], axis=-1).min()
        dr = np.linalg.norm(rh[t][:, None, :] - op[None, :, :], axis=-1).min()
        lc = dl < args.thresh
        rc = dr < args.thresh
        if lc:
            contact[t, [c for c in LEFT_HAND_MUJ]] = 1.0
            lcount += 1
        if rc:
            contact[t, [c for c in RIGHT_HAND_MUJ]] = 1.0
            rcount += 1
        contact[t, 0] = 1.0 if (lc or rc) else 0.0

    torch.save(torch.from_numpy(contact), args.out)
    print(f"wrote {args.out}  shape={tuple(contact.shape)}")
    print(f"  object-contact frames: {int((contact[:,0]>0.5).sum())}/{T}")
    print(f"  left-hand contact: {lcount}/{T}  right-hand contact: {rcount}/{T}")
    print(f"  (thresh={args.thresh}m)")


if __name__ == "__main__":
    main()
