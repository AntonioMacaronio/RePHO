#!/usr/bin/env python3
"""Browser-based viser visualizer for RePHO physics rollouts (forward + backward).

Why this exists: this node's IsaacGym camera renderer (libnvf Vulkan plugin) is ABI-incompatible
with the installed NVIDIA 570 driver and segfaults. viser renders client-side in the browser (WebGL),
bypassing the server graphics stack entirely.

Design (per review feedback):
  * ALL frames for every track are added to the scene ONCE at startup; animation just toggles
    per-frame `.visible` — no scene-API calls in the playback loop.
  * Shows BOTH the forward and backward rollouts, plus the kinematic reference ("ghost") for each.
    A GUI dropdown selects which tracks are shown; forward/backward can be offset along x so they
    don't overlap, or overlaid.

Input: rollout `intermimic.pt` saved by `intermimic/run.py --save_states` (shape [T, 592]):
  0:3 root_pos | 3:7 root_rot quat(xyzw) | 162:318 body_pos 52x3 (MuJoCo order, WORLD)
  318:321 obj_pos | 321:325 obj_rot quat(xyzw) | 330:331 obj-in-contact | 331:383 per-body contact
  383:591 body_rot 52x4 | 591 validity flag.  Reference .pt is [T,591] (no flag), same columns.

Usage:
  python scripts/viser_rollout.py \
    --forward <fwd rollout pt> --backward <bwd rollout pt> \
    --ref <ref intermimic.pt> --obj <box.obj> [--port 8080] [--separate]
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
    """One animated entity (human skeleton + object) built ONCE, animated by visibility toggling."""

    def __init__(self, server, name, states, bones, overts, ofaces, x_offset=0.0,
                 body_color=(80, 140, 240), is_ref=False):
        self.server = server
        self.name = name
        self.T = states.shape[0]
        self.is_ref = is_ref
        body_pos = states[:, 162:318].reshape(self.T, 52, 3).copy()
        body_pos[:, :, 0] += x_offset
        self.frame_nodes = []  # one parent frame per timestep; toggle .visible

        # Render the object for BOTH physics rollouts and the kinematic reference.
        # The ref's object (cols 318:325) is the noisy VisTracker input — showing it ghosted lets you
        # see the floating/penetrating kinematic object vs. the physics-corrected one.
        has_obj = overts is not None
        if has_obj:
            obj_pos = states[:, 318:321].copy(); obj_pos[:, 0] += x_offset
            obj_rot = states[:, 321:325]
        contact_obj = states[:, 330] if not is_ref else None
        contact_h = states[:, 331:383] if not is_ref else None

        for f in range(self.T):
            parent = f"/{name}/f{f:04d}"
            node = server.scene.add_frame(parent, show_axes=False, visible=(f == 0))
            self.frame_nodes.append(node)
            jp = body_pos[f]
            # joint colors: hands red on contact (skip for ref ghost)
            cols = np.tile(np.array(body_color, np.uint8), (52, 1))
            if not is_ref:
                for h in HAND_MUJ:
                    if contact_h[f, h] > 0.5:
                        cols[h] = (240, 60, 60)
            server.scene.add_point_cloud(f"{parent}/joints", points=jp, colors=cols,
                                         point_size=0.02 if is_ref else 0.028, point_shape="circle")
            seg = np.stack([jp[bones[:, 0]], jp[bones[:, 1]]], axis=1)
            bone_col = (150, 150, 150) if is_ref else (230, 230, 240)
            server.scene.add_line_segments(f"{parent}/bones", points=seg, colors=bone_col,
                                           line_width=2.0 if is_ref else 3.0)
            if has_obj:
                if is_ref:
                    # ghosted kinematic-input object: grey wireframe, translucent
                    server.scene.add_mesh_simple(f"{parent}/object", vertices=overts, faces=ofaces,
                                                 color=(150, 150, 150), wireframe=True,
                                                 wxyz=quat_xyzw_to_wxyz(obj_rot[f]),
                                                 position=obj_pos[f].astype(np.float32), opacity=0.35)
                else:
                    ocol = (220, 70, 70) if contact_obj[f] > 0.5 else (210, 180, 70)
                    server.scene.add_mesh_simple(f"{parent}/object", vertices=overts, faces=ofaces,
                                                 color=ocol, wxyz=quat_xyzw_to_wxyz(obj_rot[f]),
                                                 position=obj_pos[f].astype(np.float32), opacity=0.85)

    def set_frame(self, f):
        f = min(f, self.T - 1)
        for i, node in enumerate(self.frame_nodes):
            node.visible = (i == f)

    def hide(self):
        for node in self.frame_nodes:
            node.visible = False


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

    def apply(fi):
        fi = int(fi)
        on = {"forward": show_fwd.value, "backward": show_bwd.value,
              "ref_fwd": show_ref.value, "ref_bwd": show_ref.value}
        for key, tr in tracks.items():
            if on.get(key, False):
                tr.set_frame(fi)
            else:
                tr.hide()

    @gui_frame.on_update
    def _(_):
        apply(gui_frame.value)

    for cb in (show_fwd, show_bwd, show_ref):
        cb.on_update(lambda _: apply(gui_frame.value))

    apply(0)
    print(f"[viser] open http://localhost:{args.port}  (forward the port)")
    while True:
        if gui_play.value:
            gui_frame.value = (int(gui_frame.value) + 1) % T  # fires on_update -> apply()
        time.sleep(1.0 / max(1, gui_fps.value))


if __name__ == "__main__":
    main()
