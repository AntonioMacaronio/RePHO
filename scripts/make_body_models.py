#!/usr/bin/env python3
"""Assemble RePHO's body_models/ from assets on this box.

RePHO's converters load SMPL-H via smplx (`smplx.create(model_type='smplh', ext='pkl',
use_pca=False)`) and via smpl_sim's SMPLH_Parser (also smplx under the hood). smplx's SMPLH
loader unconditionally reads MANO fields `hands_componentsl/r`, `hands_meanl/r`,
`hands_coeffsl/r` at construction — even with use_pca=False, where the components matrix is
never applied in forward() (only the hand *mean* is, and only when flat_hand_mean=False).

The provided dependencies/smplh/*/model.npz are AMASS-format SMPL-H: correct topology,
skinning, shapedirs(16 betas), posedirs, J_regressor, kintree — but NO MANO hand fields.
So we merge: AMASS SMPL-H body  +  MANO hand fields borrowed from a SMPL-X model (the MANO
hand PCA basis is shared across SMPL-H/SMPL-X). Output = smplx-loadable SMPLH_<G>.pkl.

This is faithful for RePHO's use: it passes explicit 45-dim L/R hand poses with use_pca=False,
so only hands_mean is consumed; components/coeffs are present only to satisfy the loader.
"""
import argparse
import os
import pickle

import numpy as np

AMASS_DIR = "dependencies/smplh"  # neutral/male/female / model.npz
SMPLX_NPZ = "/home/sky/sky_workdir/CRISP-Real2Sim/prep/HMR/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz"
OUT_SMPLH = "body_models/smplh"

HAND_KEYS = ["hands_componentsl", "hands_componentsr",
             "hands_meanl", "hands_meanr",
             "hands_coeffsl", "hands_coeffsr"]


def build_smplh_pkl(gender: str, smplx_hands: dict) -> dict:
    src = os.path.join(AMASS_DIR, gender, "model.npz")
    d = dict(np.load(src, allow_pickle=True))
    model = {}
    for k, v in d.items():
        # decode the few bytes-scalars AMASS stores (bs_style, bs_type)
        if v.dtype.kind == "S" or (v.dtype == object):
            try:
                model[k] = v.item()
            except Exception:
                model[k] = v
        else:
            model[k] = v
    # graft MANO hand fields (shared basis; SMPL-H uses the same 45-dim per hand)
    for k in HAND_KEYS:
        model[k] = smplx_hands[k]
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smplx", default=SMPLX_NPZ, help="SMPLX npz to borrow MANO hand fields from")
    ap.add_argument("--out", default=OUT_SMPLH)
    args = ap.parse_args()

    sx = np.load(args.smplx, allow_pickle=True)
    smplx_hands = {k: sx[k] for k in HAND_KEYS}
    print("Borrowed MANO hand fields from", args.smplx)
    for k in HAND_KEYS:
        print(f"  {k:20s} {smplx_hands[k].shape}")

    os.makedirs(args.out, exist_ok=True)
    for gender in ["neutral", "male", "female"]:
        model = build_smplh_pkl(gender, smplx_hands)
        out_fn = os.path.join(args.out, f"SMPLH_{gender.upper()}.pkl")
        with open(out_fn, "wb") as f:
            pickle.dump(model, f)
        print(f"wrote {out_fn}  (v_template={model['v_template'].shape}, "
              f"shapedirs={model['shapedirs'].shape}, weights={model['weights'].shape})")


if __name__ == "__main__":
    main()
