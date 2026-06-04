import json
import os
import os.path
import numpy as np
import torch
from tqdm import tqdm
import smplx
import trimesh
from scipy.spatial.transform import Rotation
import joblib
os.environ['LD_LIBRARY_PATH'] = f"{os.environ.get('CONDA_PREFIX', '')}/lib:" + os.environ.get('LD_LIBRARY_PATH', '')
import trimesh
from smplx import SMPLX, SMPLH
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--recon_file', type=str, help='输入文件夹路径')
parser.add_argument('--save_name', type=str, help='输入文件夹路径',default='')
parser.add_argument('--output_path', type=str, help='输入文件夹路径',default='recon-intermimic_1013result')
parser.add_argument('--start_frame', type=int, help='输入文件夹路径',default=None)
parser.add_argument('--end_frame', type=int, help='输入文件夹路径',default=None)
parser.add_argument('--obj_name', type=str, help='输入文件夹路径')


args = parser.parse_args()


def get_cam(hmr4d_path, smpl_model_path):

    hmr4d_results = torch.load(hmr4d_path)
    smpl = SMPLX(model_path=smpl_model_path, gender='neutral', num_betas=10, use_pca=False)

    body_pose = hmr4d_results['smpl_params_incam']['body_pose']
    bsz = body_pose.shape[0]


    body_pose_global = hmr4d_results['smpl_params_global']['body_pose']
    bsz = body_pose_global.shape[0]



    print(hmr4d_results['smpl_params_incam']['betas'].shape)
    print(hmr4d_results['smpl_params_incam']['body_pose'].shape)
    print(hmr4d_results['smpl_params_incam']['global_orient'].shape)
    print(hmr4d_results['smpl_params_incam']['transl'].shape)


    common_body_pose = body_pose 

    smplout_cam = smpl(
    betas=hmr4d_results['smpl_params_incam']['betas'],
    body_pose=common_body_pose,
    global_orient=hmr4d_results['smpl_params_incam']['global_orient'],
    transl=hmr4d_results['smpl_params_incam']['transl'],
    left_hand_pose=torch.zeros(bsz,45, device=common_body_pose.device),
    right_hand_pose=torch.zeros(bsz,45, device=common_body_pose.device),
    jaw_pose=torch.zeros(bsz,3, device=common_body_pose.device),
    leye_pose=torch.zeros(bsz,3, device=common_body_pose.device),
    reye_pose=torch.zeros(bsz,3, device=common_body_pose.device),
    expression=torch.zeros(bsz,10, device=common_body_pose.device),
    )

    verts_cam = smplout_cam.vertices[0]
    cam_pelvis = smplout_cam.joints[0,0]
    smplout_world = smpl(
    betas=hmr4d_results['smpl_params_incam']['betas'],   # 保持一致
    body_pose=common_body_pose,                           # 保持一致
    global_orient=hmr4d_results['smpl_params_global']['global_orient'],
    transl=hmr4d_results['smpl_params_global']['transl'],
    left_hand_pose=torch.zeros(bsz,45, device=common_body_pose.device),
    right_hand_pose=torch.zeros(bsz,45, device=common_body_pose.device),
    jaw_pose=torch.zeros(bsz,3, device=common_body_pose.device),
    leye_pose=torch.zeros(bsz,3, device=common_body_pose.device),
    reye_pose=torch.zeros(bsz,3, device=common_body_pose.device),
    expression=torch.zeros(bsz,10, device=common_body_pose.device),
    )
    world_pelvis = smplout_world.joints[0,0]


    # 旋转：camera -> world
    R_cw = (Rotation.from_rotvec(hmr4d_results['smpl_params_global']['global_orient'][0].cpu().numpy())
    * Rotation.from_rotvec(hmr4d_results['smpl_params_incam']['global_orient'][0].cpu().numpy()).inv())

    # 平移：camera -> world   （关键：要乘 R_cw）
    t_cw = (world_pelvis.cpu().numpy()
    - R_cw.apply(cam_pelvis.cpu().numpy()))

    cam_trans = t_cw
    cam_rots = R_cw.as_rotvec()
    cam_trans = np.array([0, 0, 0])
    cam_rots = np.array([0, 0, 0])

    return cam_trans, cam_rots

def get_cam_gt():
    cam_trans = np.array([0, 0, 0])
    cam_rots = np.array([0, 0, 0])
    return cam_trans, cam_rots

def get_cam_from_gt():
    gt_cam_path = os.path.join(args.output_path, 'intermimic_gt', 'cam.npz')
    cam_data = np.load(gt_cam_path)
    return cam_data['trans'], cam_data['angles']


MODEL_PATH = './body_models'

smpl_model_male = smplx.create(MODEL_PATH, model_type='smplh',
                          gender="male",
                          use_pca=False,
                          ext='pkl')

smpl_model_female = smplx.create(MODEL_PATH, model_type='smplh',
                          gender="female",
                          use_pca=False,
                          ext='pkl')

smpl = {'male': smpl_model_male, 'female': smpl_model_female}


with open(args.recon_file, 'rb') as f:
    f = joblib.load(f)
    obj_angles, obj_trans = f['obj_angles'], f['obj_trans']
    poses, betas, trans = f['poses'], f['betas'], f['trans']
    gender = 'male'
    start_frame = args.start_frame if args.start_frame is not None else 0
    end_frame = args.end_frame if args.end_frame is not None else obj_angles.shape[0]
    obj_angles = obj_angles[start_frame:end_frame]
    obj_trans = obj_trans[start_frame:end_frame]
    poses = poses[start_frame:end_frame]
    trans = trans[start_frame:end_frame]
    if len(betas.shape) == 2:
        betas = betas[start_frame:end_frame].mean(0)

obj_name = args.obj_name
if not 'gt' in args.save_name:
    obj_angles = obj_angles.transpose(0, 2, 1)
    obj_angles =Rotation.from_matrix(obj_angles).as_rotvec()

obj_trans_incam = obj_trans.copy()
obj_angles_incam = obj_angles.copy()

poses_incam = poses.copy()
trans_incam = trans.copy()

rotation_matrix_x = Rotation.from_euler('x', -np.pi, degrees=False)
hmr_results_path = os.path.join(args.output_path, 'gvhmr/hmr4d_results.pt')
if 'gt' in args.save_name:
    cam_trans, cam_angles = get_cam_gt()
    rotation_cam = Rotation.from_rotvec(cam_angles)
    rotation_cam = rotation_matrix_x * rotation_cam
    cam_angles = rotation_cam.as_rotvec()
    cam_trans = rotation_matrix_x.apply(cam_trans)
else:
    cam_trans, cam_angles = get_cam_from_gt()
    rotation_cam = Rotation.from_rotvec(cam_angles)

frame_times = obj_trans.shape[0]

smpl_model = smpl[gender]


betas = betas[None, :].repeat(frame_times, 0)
print(betas.shape,torch.cat([torch.from_numpy(betas).float(),torch.zeros(frame_times, 16 - betas.shape[1])], dim=1).shape)
smplx_output = smpl_model(body_pose=torch.from_numpy(poses[:, 3:66]).float(),
                            global_orient=torch.from_numpy(poses[:, :3]).float(),
                            left_hand_pose=torch.from_numpy(poses[:, 66:111]).float(),
                            right_hand_pose=torch.from_numpy(poses[:, 111:156]).float(),
                            betas=torch.cat([torch.from_numpy(betas).float(),torch.zeros(frame_times, 16 - betas.shape[1])], dim=1),
                            transl=torch.from_numpy(trans).float(),)
pelvis = smplx_output.joints.detach().numpy()[:, 0, :]
#export first frame mesh
verts = smplx_output.vertices.detach().numpy()[0]
faces = smpl_model.faces



rotvecs = poses[:, :3]
rotations = Rotation.from_rotvec(rotvecs)


# Apply the rotation to the batch of rotations
rotated_rotations = rotation_cam * rotations
pelvis_global = rotation_cam.apply(pelvis) + cam_trans
# Convert the rotated rotations back to rotation vectors
poses[:, :3] = rotated_rotations.as_rotvec()

trans = rotation_cam.apply(trans)

rotvecs2 = obj_angles
rotations2 = Rotation.from_rotvec(rotvecs2)

# Apply the rotation to the batch of rotations
rotated_rotations2 = rotation_cam * rotations2
# Convert the rotated rotations back to rotation vectors
obj_angles = rotated_rotations2.as_rotvec()
obj_trans = rotation_cam.apply(obj_trans) + cam_trans



smplx_output = smpl_model(body_pose=torch.from_numpy(poses[:, 3:66]).float(),
                            global_orient=torch.from_numpy(poses[:, :3]).float(),
                            left_hand_pose=torch.from_numpy(poses[:, 66:111]).float(),
                            right_hand_pose=torch.from_numpy(poses[:, 111:156]).float(),
                            betas=torch.cat([torch.from_numpy(betas).float(),torch.zeros(frame_times, 16 - betas.shape[1])], dim=1),
                            transl=torch.zeros_like(torch.from_numpy(trans)).float())

verts = smplx_output.vertices.detach().numpy()
pelvis = smplx_output.joints.detach().numpy()[:, 0, :]
faces = smpl_model.faces

trans = pelvis_global - pelvis


verts = verts+trans[:,None, :]

obj_path_in_dir = args.recon_file.split('vistracker_result')[0] + obj_name
print(obj_path_in_dir, args.output_path)
os.makedirs(args.output_path, exist_ok=True)
os.system(f"cp -r {obj_path_in_dir} {args.output_path}")
os.system(f'python libs/create_urdf.py --out_path {args.output_path} --obj_name {obj_name}')
obj_path = os.path.join(args.output_path, obj_name, f'{obj_name}.obj')
mesh_obj = trimesh.load(obj_path, force='mesh')
obj_verts, obj_faces = mesh_obj.vertices, mesh_obj.faces


angle_matrix = Rotation.from_rotvec(obj_angles).as_matrix()
obj_verts = mesh_obj.vertices[None, ...]
obj_verts = np.matmul(obj_verts, angle_matrix.transpose(0, 2, 1)) + obj_trans[:, None, :]

min1 = verts[:, ..., 1].min(axis=1)
min2 = obj_verts[:, ..., 1].min(axis=1)

if 'gt' in args.save_name:
    min3 = np.sort(min1)[:5].mean()
    min4 = np.sort(min2)[:5].mean()
    min5 = min(min3, min4)
    diff_fix = min5
    cam_fix = diff_fix
    obj_trans[..., 1] -= diff_fix
    trans[..., 1] -= diff_fix
    cam_trans[..., 1] -= cam_fix

if 'vistracker' in args.save_name:
    min3 = min1.mean(keepdims=True)
    diff_fix = np.where(min1<min3, min1, min3)
    cam_fix = diff_fix.mean()
    diff_fix = np.where(min2<diff_fix, min2, diff_fix)

    obj_trans[..., 1] -= diff_fix
    trans[..., 1] -= diff_fix
    cam_trans[..., 1] -= cam_fix



K = np.array([[979.784423828125, 0, 1018.9523315429688], [0, 979.8400268554688, 779.4866943359375], [0, 0, 1]])
normalized_K = K.copy()
normalized_K[0, 0] /= 2048
normalized_K[1, 1] /= 1536
normalized_K[0, 2] /= 2048
normalized_K[1, 2] /= 1536


obj = {
    'angles': obj_angles,
    'trans': obj_trans,
    'name': obj_name,
}
human = {
    'poses': poses,
    'betas': betas[0],
    'trans': trans,
    'gender': gender,
}

cam_npz= {
    'trans': cam_trans,
    'angles': cam_angles,
    'K': K,
    'normalized_K': normalized_K,
    'size': np.array([1536, 2048]),
}

obj_incam = {
    'angles': obj_angles_incam,
    'trans': obj_trans_incam,
    'name': obj_name,
}
human_incam = {
    'poses': poses_incam,
    'betas': betas[0],
    'trans': trans_incam,
    'gender': gender,
}

kinematic_all = {
    'object': obj,
    'human': human,
    'cam': cam_npz,
    'object_incam': obj_incam,
    'human_incam': human_incam,
}

os.makedirs(os.path.join(args.output_path, 'intermimic'), exist_ok=True)
if args.save_name == '':
    os.makedirs(os.path.join(args.output_path, 'intermimic'), exist_ok=True)
    np.savez(os.path.join(args.output_path, 'intermimic', 'object.npz'), **obj)
    np.savez(os.path.join(args.output_path, 'intermimic', 'human.npz'), **human)
    np.savez(os.path.join(args.output_path, 'intermimic', 'cam.npz'), **cam_npz)
    np.savez(os.path.join(args.output_path, 'intermimic', 'object_incam.npz'), **obj_incam)
    np.savez(os.path.join(args.output_path, 'intermimic', 'human_incam.npz'), **human_incam)
    np.savez(os.path.join(args.output_path, 'intermimic', 'kinematic_all.npz'), **kinematic_all)
else:
    os.makedirs(os.path.join(args.output_path, f'{args.save_name}'), exist_ok=True)
    np.savez(os.path.join(args.output_path, f'{args.save_name}', 'object.npz'), **obj)
    np.savez(os.path.join(args.output_path, f'{args.save_name}', 'human.npz'), **human)
    np.savez(os.path.join(args.output_path, f'{args.save_name}', 'cam.npz'), **cam_npz)
    np.savez(os.path.join(args.output_path, f'{args.save_name}', 'object_incam.npz'), **obj_incam)
    np.savez(os.path.join(args.output_path, f'{args.save_name}', 'human_incam.npz'), **human_incam)
    np.savez(os.path.join(args.output_path, f'{args.save_name}', 'kinematic_all.npz'), **kinematic_all)

