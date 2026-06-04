#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge MuJoCo XMLs by taking all body/joint `pos` from model A
and applying them onto model B (used as the template for all other parameters).

Usage:
    python merge_pos_from_A_into_B.py A.xml B.xml merged.xml
"""
import sys
import xml.etree.ElementTree as ET
from typing import Dict, Tuple

def collect_positions(root: ET.Element) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Traverse the tree and collect pos attributes for bodies and joints keyed by name.
    Returns:
        body_pos[name] = "x y z"
        joint_pos[name] = "x y z"
    """
    body_pos = {}
    joint_pos = {}

    for elem in root.iter():
        tag = elem.tag.strip().lower()
        name = elem.get("name")
        pos = elem.get("pos")
        if name is None or pos is None:
            continue

        if tag == "body":
            body_pos[name] = pos
        elif tag == "joint":
            joint_pos[name] = pos

    return body_pos, joint_pos


def apply_positions(root: ET.Element,
                    body_pos: Dict[str, str],
                    joint_pos: Dict[str, str]) -> int:
    """
    Apply pos attributes to bodies and joints in-place on the given root.
    Returns number of updates made.
    """
    updates = 0
    for elem in root.iter():
        tag = elem.tag.strip().lower()
        name = elem.get("name")
        if not name:
            continue

        if tag == "body" and name in body_pos:
            if name != "Pelvis":
                elem.set("pos", body_pos[name])
            updates += 1
        elif tag == "joint" and name in joint_pos:
            elem.set("pos", joint_pos[name])
            updates += 1
    return updates


def get_single_root(path: str) -> ET.ElementTree:
    """
    Parse an XML file that should contain a single <mujoco> root.
    (If your file contains multiple models concatenated, split them beforehand.)
    """
    try:
        tree = ET.parse(path)
    except ET.ParseError as e:
        raise SystemExit(f"[ERROR] Failed to parse '{path}': {e}")
    root = tree.getroot()
    if root.tag.lower() != "mujoco":
        raise SystemExit(f"[ERROR] Root of '{path}' is <{root.tag}>, expected <mujoco>.")
    return tree


def main():
    if len(sys.argv) != 4:
        print("Usage: python merge_pos_from_A_into_B.py A.xml B.xml merged.xml")
        sys.exit(1)

    a_path, b_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

    # Load both XMLs
    tree_a = get_single_root(a_path)
    tree_b = get_single_root(b_path)

    root_a = tree_a.getroot()
    root_b = tree_b.getroot()

    # Collect positions from A
    body_pos_A, joint_pos_A = collect_positions(root_a)

    # Apply onto B
    updated = apply_positions(root_b, body_pos_A, joint_pos_A)

    # Write output
    tree_b.write(out_path, encoding="utf-8", xml_declaration=True)
    print(f"[OK] Wrote merged XML to: {out_path} (updated {updated} pos attributes)")

if __name__ == "__main__":
    main()
