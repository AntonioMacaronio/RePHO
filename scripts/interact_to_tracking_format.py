from isaacgym import gymapi, gymutil, gymtorch
import torch
import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as sRot
import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())
from libs.torch_utils import quat_to_exp_map
import argparse
import smplx
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='0')
parser.add_argument('--type', type=str, required=True)
parser.add_argument('--save_name', type=str, default='intermimic')
parser.add_argument('--load_contact_from', type=str, default=None)
parser.add_argument('--obj_name', type=str, default=None)
parser.add_argument('--thin_object', action='store_true')

args = parser.parse_args()
input_dir = args.input_dir

from libs.penetration_smplh import get_penetration

from smpl_sim.smpllib.smpl_mujoco_new import SMPL_BONE_ORDER_NAMES as joint_names
from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES, SMPLH_BONE_ORDER_NAMES, SMPLX_BONE_ORDER_NAMES, SMPL_MUJOCO_NAMES, SMPLH_MUJOCO_NAMES
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot
from libs.poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState


robot_cfg = {
    "mesh": False,
    "model": args.type,
    "rel_joint_lm": False,
    "upright_start": True,
    "remove_toe": False,
    "real_weight": True,
    "real_weight_porpotion_capsules": True,
    "body_params": {},
    "joint_params": {},
    "geom_params": {},
    "actuator_params": {},
}

smpl_local_robot = LocalRobot(
    robot_cfg,
    data_dir="./body_models/smplh",
)
smpl_local_robot2 = LocalRobot(
    robot_cfg,
    data_dir="./body_models/smplh",
)

smpl_2_mujoco_new = [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 
25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
14, 17, 19, 21,
40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]

R1 = np.array([
    [ 1,  0,  0],
    [ 0,  0,  -1],
    [ 0,  1,  0]
])
R1 = sRot.from_matrix(R1)

SMPL_MODEL_PATH = './body_models'

finger_indices = [
    # 左手
    18, 19, 20,  # L_Index
    21, 22, 23,  # L_Middle
    24, 25, 26,  # L_Pinky
    27, 28, 29,  # L_Ring
    30, 31, 32,  # L_Thumb
    # 右手
    37, 38, 39,  # R_Index
    40, 41, 42,  # R_Middle
    43, 44, 45,  # R_Pinky
    46, 47, 48,  # R_Ring
    49, 50, 51   # R_Thumb
]
finger_indices = [i-1 for i in finger_indices]

hand_indices = [
    # 左手
    17, # L_Wrist
    18, 19, 20,  # L_Index
    21, 22, 23,  # L_Middle
    24, 25, 26,  # L_Pinky
    27, 28, 29,  # L_Ring
    30, 31, 32,  # L_Thumb
    # 右手
    36, # R_Wrist
    37, 38, 39,  # R_Index
    40, 41, 42,  # R_Middle
    43, 44, 45,  # R_Pinky
    46, 47, 48,  # R_Ring
    49, 50, 51   # R_Thumb
]

left_hand_indices = [
    17, # L_Wrist
    18, 19, 20,  # L_Index
    21, 22, 23,  # L_Middle
    24, 25, 26,  # L_Pinky
    27, 28, 29,  # L_Ring
    30, 31, 32,  # L_Thumb
]

right_hand_indices = [
    36, # R_Wrist
    37, 38, 39,  # R_Index
    40, 41, 42,  # R_Middle
    43, 44, 45,  # R_Pinky
    46, 47, 48,  # R_Ring
    49, 50, 51   # R_Thumb
]
non_hand_indices = [i for i in range(52) if i not in hand_indices]
smpl_2_mujoco = [SMPLH_BONE_ORDER_NAMES.index(q) for q in SMPLH_MUJOCO_NAMES if q in SMPLH_BONE_ORDER_NAMES]

mujuco_hand_indices = [smpl_2_mujoco.index(i) for i in hand_indices]
mujuco_non_hand_indices = [smpl_2_mujoco.index(i) for i in non_hand_indices]
mujuco_left_hand_indices = [smpl_2_mujoco.index(i) for i in left_hand_indices]
mujuco_right_hand_indices = [smpl_2_mujoco.index(i) for i in right_hand_indices]


def convert_obj(obj_data):
    obj_motion_dict = {}
    obj_trans = obj_data['trans']
    pose_aa = obj_data['angles']
    pose_aa = (R1 * sRot.from_rotvec(pose_aa)).as_rotvec()
    pose_quat = sRot.from_rotvec(pose_aa).as_quat()
    obj_motion_dict['rot'] = pose_quat
    obj_motion_dict['trans'] = R1.apply(obj_trans)
    return obj_motion_dict


def convert_cam(obj_data):
    obj_motion_dict = {}
    obj_trans = obj_data['trans']
    pose_aa = obj_data['angles']
    pose_aa = (R1 * sRot.from_rotvec(pose_aa)).as_rotvec()
    obj_motion_dict['angles'] = pose_aa
    obj_motion_dict['trans'] = R1.apply(obj_trans)
    obj_motion_dict['start_point'] = obj_motion_dict['trans']
    #end point can be calculated by the direction of the camera and the start point
    direction = sRot.from_rotvec(pose_aa).apply([0, 0, 1])  # Get camera forward direction
    obj_motion_dict['end_point'] = obj_motion_dict['start_point'] + direction
    obj_motion_dict['K'] = obj_data['K']
    return obj_motion_dict


def convert_human_step1(human_data, xml_path):
    double = False
    full_motion_dict = {}
    human_data = human_data
    B = human_data['poses'].shape[0]

    start, end = 0, 0

    pose_aa = human_data['poses'].copy()[start:][:,:156]
    # pose_aa[:,156-90:156] = 0
    root_trans = human_data['trans'].copy()[start:]
    beta = human_data['beta'].copy() if "beta" in human_data else human_data['betas'].copy()
    if len(beta.shape) == 2:
        beta = beta[0]
    gender = human_data.get("gender", "neutral")
    fps = human_data.get("fps", 30.0)

    if isinstance(gender, np.ndarray):
        gender = gender.item()
    if isinstance(gender, bytes):
        gender = gender.decode("utf-8")
    if gender == "neutral":
        gender_number = [0]
    elif gender == "male":
        gender_number = [1]
    elif gender == "female":
        gender_number = [2]
    else:
        import ipdb
        ipdb.set_trace()
        raise Exception("Gender Not Supported!!")
    
    smpl_2_mujoco = [SMPLH_BONE_ORDER_NAMES.index(q) for q in SMPLH_MUJOCO_NAMES if q in SMPLH_BONE_ORDER_NAMES]
    batch_size = pose_aa.shape[0]

    pose_aa[:,0:3] = (R1 * sRot.from_rotvec(pose_aa[:,0:3])).as_rotvec()



    pose_aa_mj = pose_aa.reshape(-1, 52, 3)[..., smpl_2_mujoco, :].copy()
    output_dict = {}
    num = 1
    if double:
        num = 2
    for idx in range(num):
        pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(batch_size, 52, 4)
        pose_quat_raw = pose_quat.copy()

        smpl_local_robot.load_from_skeleton(betas=torch.from_numpy(beta[None,]), gender=gender_number, objs_info=None)
        smpl_local_robot.write_xml(xml_path)
        skeleton_tree = SkeletonTree.from_mjcf(xml_path)
        os.system(f'python ./libs/change_xml.py {xml_path} ./libs/omomo.xml {xml_path}')

        root_trans_offset = root_trans + skeleton_tree.local_translation[0].numpy()
        root_trans_offset = torch.from_numpy(R1.apply(root_trans_offset))
        output_dict['root_pos'] = root_trans_offset

        root_trans = root_trans_offset - skeleton_tree.local_translation[0]


        new_sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,  # This is the wrong skeleton tree (location wise) here, but it's fine since we only use the parent relationship here. 
            torch.from_numpy(pose_quat),
            root_trans_offset,
            is_local=True)
        key_name_dump = 'test'
        if robot_cfg['upright_start']:
            pose_quat_global = (sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy()) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(B, -1, 4)  # should fix pose_quat as well here...

            new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree, torch.from_numpy(pose_quat_global), root_trans_offset, is_local=False)
            pose_quat = new_sk_state.local_rotation.numpy()
            pose_quat1 = (sRot.from_quat(pose_quat_raw[:,:1].reshape(-1, 4)) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(B, -1, 4)
            pose_quat2 = (sRot.from_quat([0.5, 0.5, 0.5, 0.5]) * sRot.from_quat(pose_quat_raw[:,1:].reshape(-1, 4)) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(B, -1, 4)
            pose_quat_new = np.concatenate([pose_quat1, pose_quat2], axis=1)
            print((pose_quat_new - pose_quat).max(), 9999999)
            print((pose_quat_new - pose_quat).min(), 9999999)
            pose_quat = pose_quat_new

            output_dict['pose_quat'] = pose_quat
            output_dict['pose_quat_global'] = pose_quat_global

            ############################################################
            
            if idx == 1:
                left_to_right_index = [0, 5, 6, 7, 8, 1, 2, 3, 4, 9, 10, 11, 12, 13, 19, 20, 21, 22, 23, 14, 15, 16, 17, 18]
                pose_quat_global = pose_quat_global[:, left_to_right_index]
                pose_quat_global[..., 0] *= -1
                pose_quat_global[..., 2] *= -1

                root_trans_offset[..., 1] *= -1
            ############################################################

        new_motion_out = {}
        new_motion_out['pose_quat_global'] = pose_quat_global
        new_motion_out['pose_quat'] = pose_quat
        new_motion_out['trans_orig'] = root_trans
        new_motion_out['root_trans_offset'] = root_trans_offset

    smpl_model = smplx.create(SMPL_MODEL_PATH, 
                            model_type=args.type,
                            gender=gender,
                            use_pca=False,
                            ext='pkl',
                            flat_hand_mean=False)
    pose_aa = torch.from_numpy(pose_aa)
    smplx_output = smpl_model(body_pose=pose_aa[:, 3:66].float(),
                global_orient=pose_aa[:, :3].float(),
                left_hand_pose=(pose_aa[:, 66:111]).float(),
                right_hand_pose=(pose_aa[:, 111:156]).float(),
                betas=torch.cat([torch.from_numpy(beta[None,]).repeat(batch_size, 1).float(),torch.zeros(batch_size, 16 - beta.shape[0])], dim=1),
                transl=root_trans.float())
    joints = smplx_output.joints.detach().numpy()
    pelvis = joints[:, 0].copy()

    output_dict['rg_pos'] = joints[:, smpl_2_mujoco, :]
    output_dict['rb_rot'] = pose_quat_global
    output_dict['root_pos'] = root_trans_offset.numpy()
    output_dict['root_rot'] = pose_quat_global[:, 0, :]
    dof_new = _local_rotation_to_dof_smpl(torch.from_numpy(pose_quat)).numpy()
    output_dict['dof_pos'] = dof_new
    for k, v in output_dict.items():
        print(type(v), k, v.shape)
        output_dict[k] = torch.from_numpy(v).float()
    # import ipdb; ipdb.set_trace()
    return output_dict

def _local_rotation_to_dof_smpl(local_rot):
    B, J, _ = local_rot.shape
    dof_pos = quat_to_exp_map(local_rot[:, 1:])
    return dof_pos.reshape(B, -1)

def convert_final(human_motion_dict, obj_motion_dict, inputdir):
    bsz = human_motion_dict['root_pos'].shape[0]
    data_final = torch.zeros((bsz, 591), dtype=torch.float32)
    data_final[:, :3] = human_motion_dict['root_pos']
    data_final[:, 3:7] = human_motion_dict['root_rot']
    dof_pos = human_motion_dict['dof_pos'].reshape(bsz, -1, 3).detach().cpu()
    dof_pos[:, finger_indices] = 0



    print(dof_pos.shape)
    print(data_final.device, dof_pos.device)
    data_final[:, 9:9+153] = dof_pos.reshape(bsz, 153)

    
    data_final[:, 162:162+52*3] = human_motion_dict['rg_pos'].reshape(bsz, 52*3)
    data_final[:, 331+52:331+52+52*4] = human_motion_dict['rb_rot'].reshape(bsz, 52*4)
    
    data_final[:, 318:318+3] = torch.from_numpy(obj_motion_dict['trans'])
    data_final[:, 321:321+4] = torch.from_numpy(obj_motion_dict['rot'])

    human_path = inputdir + '/human.npz'
    obj_path = inputdir + '/object.npz'
    obj_name = args.obj_name

    obj_verts = '/'.join(inputdir.split('/')[:-1] + [obj_name, obj_name + '.obj'])


    if not args.thin_object:
        penetration = get_penetration(human_path, obj_path, obj_verts)
        np.save(os.path.join(inputdir, 'penetration.npy'), penetration)

    if args.load_contact_from is not None:
        contact_data = torch.load(args.input_dir+'/'+args.load_contact_from+'/contact.pt')
        data_final[:, 330:331+52] = contact_data


    return data_final

def convert_intermimic(inputdir):
    human_data = np.load(inputdir + '/human.npz', allow_pickle=True)
    obj_data = np.load(inputdir + '/object.npz', allow_pickle=True)
    cam_data = np.load(inputdir + '/cam.npz', allow_pickle=True)

    sub_name = 'intermimic'
    xml_path = inputdir+f'/{sub_name}_humanoid.xml'

    obj_motion_dict = convert_obj(obj_data)
    cam_dict = convert_cam(cam_data)
    np.savez(inputdir + '/cam_intermimic.npz', cam_dict)

    human_motion_dict = convert_human_step1(human_data, xml_path)
    data_final = convert_final(human_motion_dict, obj_motion_dict, inputdir)
    out_dir = inputdir+f'/intermimic.pt'
    torch.save(data_final, out_dir)

convert_intermimic(input_dir+'/'+args.save_name)

