#!/usr/bin/env python3
"""Browser-based viser visualizer for RePHO physics rollouts (forward + backward).

Why this exists: this node's IsaacGym camera renderer (libnvf Vulkan plugin) is ABI-incompatible
with the installed NVIDIA 570 driver and segfaults. viser renders client-side in the browser (WebGL),
bypassing the server graphics stack entirely.

Design:
  * Each track uploads a FIXED set of handles ONCE (a joints point-cloud, a bones line-segment set,
    and — for the object — the mesh geometry a single time). Playback MUTATES those handles per frame
    (`.points`, `.colors`, `.position`, `.wxyz`) rather than adding a copy per timestep. This keeps
    the scene at O(#tracks) objects and streams the object mesh across the wire only once, so startup
    is fast even for long clips (the old "add T copies + toggle .visible" design streamed ~3000+
    objects for 4 tracks x 265 frames -> slow load).
  * Shows BOTH the forward and backward rollouts, plus the kinematic reference ("ghost") for each.
    Checkboxes select which tracks are shown; forward/backward can be offset along x (--separate).
  * NOTE: a mesh handle's color cannot be reassigned per frame, so the object's per-frame contact
    color (red on contact) is dropped; object contact is still conveyed by the red hand joints.

Input: rollout `intermimic.pt` saved by `intermimic/run.py --save_states` (shape [T, 592]):
  0:3 root_pos | 3:7 root_rot quat(xyzw) | 162:318 body_pos 52x3 (MuJoCo order, WORLD)
  318:321 obj_pos | 321:325 obj_rot quat(xyzw) | 330:331 obj-in-contact | 331:383 per-body contact
  383:591 body_rot 52x4 | 591 validity flag.  Reference .pt is [T,591] (no flag), same columns.

Usage:
  python scripts/viser_rollout.py \
    --forward <fwd rollout pt> --backward <bwd rollout pt> \
    --ref <ref intermimic.pt> --obj <box.obj> [--port 8080] [--separate]

--------------------------------------------------------------------------------
HOW TO PRODUCE THE ROLLOUT .pt THIS VIEWER READS
--------------------------------------------------------------------------------
This viewer reads SAVED rollout tensors [T,592]; it does NOT attach to a live
training loop. To view a policy mid-training, run a separate EVAL pass on a
checkpoint (on a spare GPU so training is undisturbed), which writes the rollout
via `--save_states`, then point this viewer at it.

The eval pass = `intermimic/run.py --test --save_states`:
  * `--test`            -> mode='test' (single-env playback of the policy)
  * `--save_states`     -> on episode end, writes <output_path>/intermimic.pt
                           [T,592] (cols match this file's header). Then quit()s.
  * DO NOT pass `--enable_camera_sensors` / `--save_images` on a headless box --
    that path uses the IsaacGym Vulkan renderer which segfaults here. `--save_states`
    is independent of it, so state export works without any server-side rendering.
  * `--num_envs 1`, `--stateInit Start`, `--init_range_left 0`; the BACKWARD
    policy additionally needs reverse_time (its train script sets it; do NOT pass
    `--no_reverse_time` for backward, DO pass it for forward).

Checkpoints live at <out_root>/<seq>_dual/{forward,backward}/smplx/nn/mimic_000<EPOCH>.pth
with matching ref_hoi/ref_hoi_<EPOCH>.npz + hoi_data/intermimic_<EPOCH>.pt. Example
(forward policy, epoch 53300, eval on GPU 1 while training runs on GPU 0):

  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"   # isaacgym needs libpython
  D=tracking_intermediate_file/<seq>_dual; E=53300
  OUT=data/demo_data/output/rollout_states/forward/<seq>_live; mkdir -p "$OUT"
  CUDA_VISIBLE_DEVICES=1 python intermimic/run.py --task InterMimic \
    --cfg_env intermimic/data/cfg/omomo_train_917.yaml \
    --cfg_train intermimic/data/cfg/train/rlg/omomo.yaml --headless \
    --output_path "$OUT" --stateInit Start --init_range_left 0 --no_reverse_time \
    --device_id 0 --rl_device cuda:0 \
    --motion_file data/demo_data/output/<seq> --sub_file_name intermimic_vistracker \
    --checkpoint "$D/forward/smplx/nn/mimic_000$E.pth" \
    --hoi_refs_path "$D/forward/ref_hoi/ref_hoi_$E.npz" \
    --hoi_data_path "$D/forward/hoi_data/intermimic_$E.pt" \
    --test --num_envs 1 --save_states
  # -> $OUT/intermimic.pt  (the [T,592] this viewer wants; col -1 = validity/survived flag)

Repeat for backward (drop --no_reverse_time; use the backward/ checkpoint+refs),
then launch this viewer on both. `scripts/infer_forward.sh` is the fuller
image-rendering variant (needs a working camera renderer + epoch>=54500); prefer
the --save_states path above for headless.

The `--ref` file is the KINEMATIC target the policy tracks -- it is the SAME
intermimic.pt produced by step-2 (interact_to_tracking_format.py). Loading it as
the grey ghost lets you compare the physics rollout against the reference, and it
should match the CRISP-Obj unified_human_scene_object_viewer.py view of the same
sequence (both draw the reconstructed human+object).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
import viser

LEFT_HAND_MUJ = list(range(17, 33))
RIGHT_HAND_MUJ = list(range(36, 52))
HAND_MUJ = LEFT_HAND_MUJ + RIGHT_HAND_MUJ


def load_pt(path):
    if path is None or not Path(path).exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False).float().numpy()


def quat_xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]], np.float32)


class Track:
    """One animated entity (human skeleton + object), built ONCE and animated by MUTATING a fixed
    set of handles per frame (joints .points/.colors, bones .points, object .position/.wxyz).

    The previous design added T copies of every asset (frame + joints + bones + object mesh) and
    toggled visibility. With 4 tracks x 265 frames that is ~3000+ scene objects streamed to the
    browser at startup -> long load. Here each track uploads exactly 3 handles (+ the object mesh
    geometry once) and the playback loop just reassigns their per-frame arrays: O(1) scene objects,
    the mesh vertices/faces cross the wire a single time."""

    def __init__(self, server, name, states, bones, overts, ofaces, x_offset=0.0,
                 body_color=(80, 140, 240), is_ref=False):
        self.server = server
        self.name = name
        self.T = states.shape[0]
        self.is_ref = is_ref
        self.bones = bones
        self.x_offset = x_offset

        # ---- precompute per-frame arrays (cheap, no scene traffic) ----
        self.body_pos = states[:, 162:318].reshape(self.T, 52, 3).astype(np.float32).copy()
        self.body_pos[:, :, 0] += x_offset                          # (T,52,3) joint positions
        self.bone_seg = np.stack([self.body_pos[:, bones[:, 0]],
                                  self.body_pos[:, bones[:, 1]]], axis=2)  # (T,nb,2,3)

        self.has_obj = overts is not None
        if self.has_obj:
            self.obj_pos = states[:, 318:321].astype(np.float32).copy(); self.obj_pos[:, 0] += x_offset
            self.obj_rot = states[:, 321:325].astype(np.float32)     # xyzw quats

        # per-frame joint colors (hands turn red on contact; ref is a static grey ghost)
        base = np.tile(np.array(body_color, np.uint8), (self.T, 52, 1))
        if not is_ref:
            contact_h = states[:, 331:383]                           # (T,52) per-body contact
            hot = contact_h[:, HAND_MUJ] > 0.5                       # (T, |hands|)
            for k, h in enumerate(HAND_MUJ):
                base[hot[:, k], h] = (240, 60, 60)
            self.contact_obj = states[:, 330]                        # object-in-contact per frame
        self.joint_cols = base                                       # (T,52,3)
        self.bone_col = (150, 150, 150) if is_ref else (230, 230, 240)

        # ---- create the FIXED handles once (frame 0) ----
        parent = f"/{name}"
        server.scene.add_frame(parent, show_axes=False)
        self.joints_h = server.scene.add_point_cloud(
            f"{parent}/joints", points=self.body_pos[0], colors=self.joint_cols[0],
            point_size=0.02 if is_ref else 0.028, point_shape="circle")
        self.bones_h = server.scene.add_line_segments(
            f"{parent}/bones", points=self.bone_seg[0], colors=self.bone_col,
            line_width=2.0 if is_ref else 3.0)
        self.obj_h = None
        if self.has_obj:
            if is_ref:
                self.obj_h = server.scene.add_mesh_simple(
                    f"{parent}/object", vertices=overts, faces=ofaces, color=(150, 150, 150),
                    wireframe=True, opacity=0.35,
                    wxyz=quat_xyzw_to_wxyz(self.obj_rot[0]), position=self.obj_pos[0])
            else:
                # object mesh geometry uploaded ONCE; color can't be mutated per-frame on a mesh
                # handle, so pick a single manipulation color (contact state still shown via the
                # red hand joints + the object-contact GUI is dropped in favor of the joint cue).
                self.obj_h = server.scene.add_mesh_simple(
                    f"{parent}/object", vertices=overts, faces=ofaces, color=(210, 180, 70),
                    opacity=0.85, wxyz=quat_xyzw_to_wxyz(self.obj_rot[0]), position=self.obj_pos[0])

    def set_frame(self, f):
        f = min(int(f), self.T - 1)
        self.joints_h.points = self.body_pos[f]
        self.joints_h.colors = self.joint_cols[f]
        self.bones_h.points = self.bone_seg[f]
        if self.obj_h is not None:
            self.obj_h.position = self.obj_pos[f]
            self.obj_h.wxyz = quat_xyzw_to_wxyz(self.obj_rot[f])

    def set_visible(self, v: bool):
        self.joints_h.visible = v
        self.bones_h.visible = v
        if self.obj_h is not None:
            self.obj_h.visible = v

    def hide(self):
        self.set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forward", required=True)
    ap.add_argument("--backward", default=None)
    ap.add_argument("--ref", default=None, help="kinematic reference intermimic.pt (ghost)")
    ap.add_argument("--obj", required=True)
    ap.add_argument("--bones", default="/tmp/repho_bones.npy")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--separate", action="store_true",
                    help="offset forward/backward along x so they don't overlap")
    args = ap.parse_args()

    fwd = load_pt(args.forward)
    bwd = load_pt(args.backward)
    ref = load_pt(args.ref)
    bones = np.load(args.bones)
    mesh = trimesh.load(args.obj, force="mesh")
    overts = (np.asarray(mesh.vertices, np.float32) - np.asarray(mesh.vertices, np.float32).mean(0))
    ofaces = np.asarray(mesh.faces, np.int32)

    T = max(fwd.shape[0], bwd.shape[0] if bwd is not None else 0)
    print(f"[viser] forward T={fwd.shape[0]}"
          + (f", backward T={bwd.shape[0]}" if bwd is not None else "")
          + (f", ref T={ref.shape[0]}" if ref is not None else ""))

    server = viser.ViserServer(port=args.port)
    server.scene.add_grid("/grid", width=8, height=8, cell_size=0.5, plane="xy")
    server.scene.set_up_direction("+z")

    off = 1.2 if args.separate else 0.0
    tracks = {}
    tracks["forward"] = Track(server, "forward", fwd, bones, overts, ofaces,
                              x_offset=-off, body_color=(80, 140, 240))
    if ref is not None:
        tracks["ref_fwd"] = Track(server, "ref_fwd", ref, bones, overts, ofaces,
                                  x_offset=-off, body_color=(150, 150, 150), is_ref=True)
    if bwd is not None:
        tracks["backward"] = Track(server, "backward", bwd, bones, overts, ofaces,
                                   x_offset=+off, body_color=(90, 210, 130))
        if ref is not None:
            tracks["ref_bwd"] = Track(server, "ref_bwd", ref, bones, overts, ofaces,
                                      x_offset=+off, body_color=(150, 150, 150), is_ref=True)

    # GUI
    show_fwd = server.gui.add_checkbox("forward (blue)", True)
    show_bwd = server.gui.add_checkbox("backward (green)", bwd is not None)
    show_ref = server.gui.add_checkbox("kinematic ref (grey ghost)", ref is not None)
    gui_frame = server.gui.add_slider("frame", 0, T - 1, 1, 0)
    gui_play = server.gui.add_checkbox("play", True)
    gui_fps = server.gui.add_slider("fps", 1, 60, 1, int(args.fps))

    def _vis_map():
        return {"forward": show_fwd.value, "backward": show_bwd.value,
                "ref_fwd": show_ref.value, "ref_bwd": show_ref.value}

    def set_frame(fi):
        """Advance every VISIBLE track to frame fi (mutates handles in place; no scene rebuild)."""
        fi = int(fi)
        on = _vis_map()
        for key, tr in tracks.items():
            if on.get(key, False):
                tr.set_frame(fi)

    def apply_visibility():
        """Toggle track visibility only (checkbox change) — cheap, no per-frame array push."""
        on = _vis_map()
        for key, tr in tracks.items():
            tr.set_visible(on.get(key, False))
        set_frame(gui_frame.value)   # refresh newly-shown tracks to the current frame

    @gui_frame.on_update
    def _(_):
        set_frame(gui_frame.value)

    for cb in (show_fwd, show_bwd, show_ref):
        cb.on_update(lambda _: apply_visibility())

    apply_visibility()
    print(f"[viser] open http://localhost:{args.port}  (forward the port)")
    while True:
        if gui_play.value:
            gui_frame.value = (int(gui_frame.value) + 1) % T  # fires on_update -> apply()
        time.sleep(1.0 / max(1, gui_fps.value))


if __name__ == "__main__":
    main()
