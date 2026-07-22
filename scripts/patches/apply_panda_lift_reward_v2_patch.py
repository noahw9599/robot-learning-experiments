#!/usr/bin/env python3
"""Apply or restore the reversible Panda lift-reward experiment."""

from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "# Project Jiraiya Panda lift_reward_v2"
BACKUP_SUFFIX = ".jiraiya_panda_lift_reward_v2_backup"


def source_path(playground_dir: Path) -> Path:
    return playground_dir / "mujoco_playground/_src/manipulation/franka_emika_panda/pick.py"


def restore(path: Path) -> None:
    backup = path.with_name(path.name + BACKUP_SUFFIX)
    if backup.exists():
        path.write_bytes(backup.read_bytes())
        backup.unlink()
        print(f"Restored clean source: {path}")
    elif MARKER in path.read_text():
        raise RuntimeError("Patch marker exists but its backup is missing; refusing to overwrite source.")
    else:
        print("No active Panda lift-reward patch found.")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one {label} anchor, found {count}.")
    return text.replace(old, new, 1)


def apply(path: Path) -> None:
    if MARKER in path.read_text():
        print("Panda lift-reward patch is already active.")
        return

    backup = path.with_name(path.name + BACKUP_SUFFIX)
    if backup.exists():
        raise RuntimeError(f"Backup already exists: {backup}; restore it before applying again.")

    original = path.read_text()
    backup.write_text(original)
    text = original

    text = replace_once(
        text,
        "              gripper_box=4.0,\n",
        "              gripper_box=2.0,\n",
        "gripper-box scale",
    )

    text = replace_once(
        text,
        "robot_target_qpos=0.3,\n",
        "robot_target_qpos=0.3,\n"
        "              # Project Jiraiya Panda lift_reward_v2: prioritize vertical lift over dragging.\n"
        "              box_hand_contact=1.0,\n"
        "              gripper_closed=0.5,\n"
        "              lift_progress=12.0,\n"
        "              lift_hold=12.0,\n"
        "              pre_lift_drag=-4.0,\n",
        "reward-scale anchor",
    )

    text = replace_once(
        text,
        '    metrics = {\n        "out_of_bounds": jp.array(0.0, dtype=float),\n        **{k: 0.0 for k in self._config.reward_config.scales.keys()},\n    }\n',
        '    metrics = {\n'
        '        "out_of_bounds": jp.array(0.0, dtype=float),\n'
        '        **{k: 0.0 for k in self._config.reward_config.scales.keys()},\n'
        '        "lifted": jp.array(0.0, dtype=float),\n'
        '        "lifted_steps": jp.array(0.0, dtype=float),\n'
        '        "box_height_delta": jp.array(0.0, dtype=float),\n'
        '        "horizontal_drag": jp.array(0.0, dtype=float),\n'
        '    }\n',
        "metrics initialization anchor",
    )

    text = replace_once(
        text,
        '    info = {"rng": rng, "target_pos": target_pos, "reached_box": 0.0}\n',
        "    info = {\n"
        '        "rng": rng,\n'
        '        "target_pos": target_pos,\n'
        '        "reached_box": 0.0,\n'
        '        "lifted": jp.array(0.0, dtype=float),\n'
        '        "lifted_steps": jp.array(0, dtype=jp.int32),\n'
        "    }\n",
        "reset-info anchor",
    )

    step_anchor = "    raw_rewards = self._get_reward(data, state.info)\n"
    step_new = (
        "    # Project Jiraiya Panda lift_reward_v2: track lift and pre-lift dragging.\n"
        "    box_pos = data.xpos[self._obj_body]\n"
        "    box_height_delta = jp.maximum(box_pos[2] - self._init_obj_pos[2], 0.0)\n"
        "    horizontal_drag = jp.linalg.norm(box_pos[:2] - self._init_obj_pos[:2])\n"
        "    lifted = box_height_delta >= 0.05\n"
        '    state.info["lifted"] = lifted.astype(float)\n'
        '    state.info["lifted_steps"] = jp.where(\n'
        '        lifted, state.info["lifted_steps"] + 1, 0\n'
        "    )\n"
        + step_anchor
    )
    text = replace_once(text, step_anchor, step_new, "step reward anchor")

    reward_state_anchor = (
        '    target_pos = info["target_pos"]\n'
        "    box_pos = data.xpos[self._obj_body]\n"
    )
    reward_state_new = reward_state_anchor + (
        "    horizontal_drag = jp.linalg.norm(box_pos[:2] - self._init_obj_pos[:2])\n"
    )
    text = replace_once(text, reward_state_anchor, reward_state_new, "reward-state anchor")

    text = replace_once(
        text,
        '    rewards = {\n'
        '        "gripper_box": gripper_box,\n'
        '        "box_target": box_target * info["reached_box"],\n',
        '    rewards = {\n'
        '        "gripper_box": gripper_box,\n'
        '        "box_target": box_target * info["lifted"],\n',
        "target gate",
    )

    reward_anchor = '        "gripper_box": gripper_box,\n'
    reward_new = reward_anchor + (
        '        # Project Jiraiya Panda lift_reward_v2: geometric grasp proxy and anti-drag term.\n'
        '        "box_hand_contact": (jp.linalg.norm(box_pos - gripper_pos) < 0.025).astype(float),\n'
        '        "gripper_closed": jp.clip(\n'
        '            1.0 - jp.mean(data.qpos[-2:]) / 0.04,\n'
        '            0.0,\n'
        '            1.0,\n'
        '        ),\n'
        '        "lift_progress": (\n'
        '            1.0 - jp.exp(-20.0 * jp.maximum(box_pos[2] - self._init_obj_pos[2], 0.0))\n'
        '        ) * info["lifted"],\n'
        '        "lift_hold": jp.clip(info["lifted_steps"] / 10.0, 0.0, 1.0),\n'
        '        "pre_lift_drag": jp.clip(horizontal_drag / 0.05, 0.0, 1.0) * (1.0 - info["lifted"]),\n'
    )
    text = replace_once(text, reward_anchor, reward_new, "reward dictionary anchor")

    metric_anchor = (
        "    state.metrics.update(\n"
        "        **raw_rewards, out_of_bounds=out_of_bounds.astype(float)\n"
        "    )\n"
    )
    metric_new = (
        "    state.metrics.update(\n"
        "        **raw_rewards,\n"
        "        out_of_bounds=out_of_bounds.astype(float),\n"
        '        lifted=state.info["lifted"],\n'
        '        lifted_steps=state.info["lifted_steps"].astype(float),\n'
        "        box_height_delta=box_height_delta,\n"
        "        horizontal_drag=horizontal_drag,\n"
        "    )\n"
    )
    text = replace_once(text, metric_anchor, metric_new, "metrics anchor")

    text = text.replace(
        "class PandaPickCube(panda.PandaBase):\n",
        "class PandaPickCube(panda.PandaBase):\n  " + MARKER + "\n",
        1,
    )
    path.write_text(text)
    print(f"Applied Panda lift-reward patch: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--playground-dir", type=Path, required=True)
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()
    path = source_path(args.playground_dir)
    if not path.exists():
        raise SystemExit(f"Missing source file: {path}")
    if args.restore:
        restore(path)
    else:
        apply(path)


if __name__ == "__main__":
    main()
