#!/usr/bin/env python3
"""Layer release-and-settle rewards on top of the active Panda v3 patch."""

from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "# Project Jiraiya Panda transport_reward_v4"
BACKUP_SUFFIX = ".jiraiya_panda_transport_reward_v4_backup"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one {label} anchor, found {count}.")
    return text.replace(old, new, 1)


def apply(path: Path) -> None:
    text = path.read_text()
    if MARKER in text:
        print("Panda transport-reward v4 patch is already active.")
        return
    if "Project Jiraiya Panda transport_reward_v3" not in text:
        raise RuntimeError("v4 requires the active Project Jiraiya v3 patch.")

    backup = path.with_name(path.name + BACKUP_SUFFIX)
    if backup.exists():
        raise RuntimeError(f"Backup already exists: {backup}; restore v4 first.")
    backup.write_text(text)

    text = replace_once(
        text,
        "              transport_progress=4.0,\n",
        "              transport_progress=4.0,\n"
        "              release_open=3.0,\n"
        "              placement_stable=8.0,\n"
        "              early_release=-2.0,\n",
        "v4 reward scales",
    )

    text = replace_once(
        text,
        '        "horizontal_drag": jp.array(0.0, dtype=float),\n'
        '        "transport_progress": jp.array(0.0, dtype=float),\n',
        '        "horizontal_drag": jp.array(0.0, dtype=float),\n'
        '        "transport_progress": jp.array(0.0, dtype=float),\n'
        '        "near_target_steps": jp.array(0.0, dtype=float),\n'
        '        "release_steps": jp.array(0.0, dtype=float),\n',
        "v4 metrics initialization",
    )

    text = replace_once(
        text,
        '        "transport_progress": jp.array(0.0, dtype=float),\n'
        "    }\n",
        '        "transport_progress": jp.array(0.0, dtype=float),\n'
        '        "near_target_steps": jp.array(0, dtype=jp.int32),\n'
        '        "release_steps": jp.array(0, dtype=jp.int32),\n'
        "    }\n",
        "v4 reset info",
    )

    step_anchor = '    state.info["prev_target_distance"] = target_distance\n'
    step_new = step_anchor + (
        "    gripper_open = jp.clip(jp.mean(data.qpos[-2:]) / 0.04, 0.0, 1.0)\n"
        "    eligible_release = (state.info[\"lifted\"] >= 0.5) & (state.info[\"lifted_steps\"] >= 10)\n"
        "    near_target = target_distance <= 0.08\n"
        '    state.info["near_target_steps"] = jp.where(\n'
        '        near_target & (gripper_open >= 0.8),\n'
        '        state.info["near_target_steps"] + 1,\n'
        "        0,\n"
        "    )\n"
        '    state.info["release_steps"] = jp.where(\n'
        '        eligible_release & (gripper_open >= 0.8),\n'
        '        state.info["release_steps"] + 1,\n'
        "        0,\n"
        "    )\n"
    )
    text = replace_once(text, step_anchor, step_new, "v4 step state")

    reward_anchor = '        "transport_progress": info["transport_progress"],\n'
    reward_new = reward_anchor + (
        '        "release_open": info["lifted"] * jp.clip(info["lifted_steps"] / 10.0, 0.0, 1.0) * (jp.linalg.norm(target_pos - box_pos) <= 0.08).astype(float) * jp.clip(jp.mean(data.qpos[-2:]) / 0.04, 0.0, 1.0),\n'
        '        "placement_stable": jp.clip(info["near_target_steps"] / 10.0, 0.0, 1.0),\n'
        '        "early_release": jp.clip(jp.mean(data.qpos[-2:]) / 0.04, 0.0, 1.0) * info["lifted"] * (1.0 - (jp.linalg.norm(target_pos - box_pos) <= 0.08).astype(float)),\n'
    )
    text = replace_once(text, reward_anchor, reward_new, "v4 reward terms")

    metric_anchor = '        horizontal_drag=horizontal_drag,\n'
    metric_new = metric_anchor + (
        '        near_target_steps=state.info["near_target_steps"].astype(float),\n'
        '        release_steps=state.info["release_steps"].astype(float),\n'
    )
    text = replace_once(text, metric_anchor, metric_new, "v4 metrics update")

    text = text.replace(
        "class PandaPickCube(panda.PandaBase):\n",
        "class PandaPickCube(panda.PandaBase):\n  " + MARKER + "\n",
        1,
    )
    path.write_text(text)
    print(f"Applied Panda transport-reward v4 patch: {path}")


def restore(path: Path) -> None:
    backup = path.with_name(path.name + BACKUP_SUFFIX)
    if not backup.exists():
        raise RuntimeError(f"Missing v4 backup: {backup}")
    path.write_bytes(backup.read_bytes())
    backup.unlink()
    print(f"Restored v3 source: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--playground-dir", type=Path, required=True)
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()
    path = args.playground_dir / "mujoco_playground/_src/manipulation/franka_emika_panda/pick.py"
    if args.restore:
        restore(path)
    else:
        apply(path)


if __name__ == "__main__":
    main()
