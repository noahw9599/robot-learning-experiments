#!/usr/bin/env python3
"""Apply the G1 recovery_reference_v1 task patch.

This is an imitation-style reference-tracking curriculum. It does not use real
demonstration data yet. Instead, it creates a small state-dependent recovery
reference: when the torso pitches forward, bias hip/ankle targets backward; when
it pitches backward, bias them forward. The policy is fine-tuned from the best
v5 checkpoint rather than trained from scratch.
"""

from __future__ import annotations

import argparse
from pathlib import Path


PATCH_MARKER = "# Robot Learning Experiments G1 recovery_reference_v1"
RESET_START = "  def reset(self, rng: jax.Array) -> mjx_env.State:"
STEP_START = "  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:"
OBS_START = "  def _get_obs("

PATCHED_RESET_AND_STEP = '''  def _stand_qpos(self) -> jax.Array:
    """G1 recovery_reference_v1 neutral standing reset pose."""
    qpos = jp.array(self._init_q)
    for joint_name in ("left_hip_pitch_joint", "right_hip_pitch_joint"):
      qpos = qpos.at[self.mj_model.joint(joint_name).qposadr].add(-0.05)
    for joint_name in ("left_ankle_pitch_joint", "right_ankle_pitch_joint"):
      qpos = qpos.at[self.mj_model.joint(joint_name).qposadr].add(0.05)
    return qpos

  def _foot_xy(self, data: mjx.Data) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    left_foot_x = data.geom_xpos[self._left_foot_geom_id, 0]
    left_foot_y = data.geom_xpos[self._left_foot_geom_id, 1]
    right_foot_x = data.geom_xpos[self._right_foot_geom_id, 0]
    right_foot_y = data.geom_xpos[self._right_foot_geom_id, 1]
    return left_foot_x, left_foot_y, right_foot_x, right_foot_y

  def _neutral_pose(self) -> jax.Array:
    return self._stand_qpos()[7:]

  def _recovery_reference_pose(self, data: mjx.Data) -> tuple[jax.Array, jax.Array]:
    """State-dependent corrective reference pose.

    In the v5 diagnostic, positive torso_gravity_x grew during the forward fall.
    This reference applies a small opposing hip/ankle pitch bias proportional to
    pitch and pitch-rate. It is deliberately weak so the policy can still learn
    corrections instead of being forced into a rigid pose.
    """
    pose = self._neutral_pose()
    torso_gravity = self.get_gravity(data, "torso")
    angvel = self.get_global_angvel(data, "pelvis")
    pitch_signal = jp.clip(torso_gravity[0] + 0.12 * angvel[1], -0.45, 0.45)

    for joint_name in ("left_hip_pitch_joint", "right_hip_pitch_joint"):
      adr = self.mj_model.joint(joint_name).qposadr - 7
      pose = pose.at[adr].add(-0.08 * pitch_signal)
    for joint_name in ("left_ankle_pitch_joint", "right_ankle_pitch_joint"):
      adr = self.mj_model.joint(joint_name).qposadr - 7
      pose = pose.at[adr].add(0.08 * pitch_signal)
    for joint_name in ("left_knee_joint", "right_knee_joint"):
      adr = self.mj_model.joint(joint_name).qposadr - 7
      pose = pose.at[adr].add(0.025 * jp.abs(pitch_signal))
    return pose, pitch_signal

  def _recovery_info(self, rng: jax.Array, motor_targets: jax.Array, data: mjx.Data) -> dict[str, Any]:
    left_foot_x, left_foot_y, right_foot_x, right_foot_y = self._foot_xy(data)
    support_mid_x = 0.5 * (left_foot_x + right_foot_x)
    support_mid_y = 0.5 * (left_foot_y + right_foot_y)
    support_front_x = jp.maximum(left_foot_x, right_foot_x)
    support_back_x = jp.minimum(left_foot_x, right_foot_x)
    torso_gravity = self.get_gravity(data, "torso")
    return {
        "rng": rng,
        "step": 0,
        "command": jp.zeros(3),
        "last_act": jp.zeros(self.action_size),
        "last_last_act": jp.zeros(self.action_size),
        "motor_targets": motor_targets,
        "feet_air_time": jp.zeros(2),
        "last_contact": jp.zeros(2, dtype=bool),
        "swing_peak": jp.zeros(2),
        "phase_dt": jp.zeros(1),
        "phase": jp.array([0.0, jp.pi]),
        "push": jp.array([0.0, 0.0]),
        "push_step": 0,
        "push_interval_steps": jp.array(1, dtype=jp.int32),
        "reset_left_foot_x": left_foot_x,
        "reset_left_foot_y": left_foot_y,
        "reset_right_foot_x": right_foot_x,
        "reset_right_foot_y": right_foot_y,
        "support_mid_x": support_mid_x,
        "support_mid_y": support_mid_y,
        "support_front_x": support_front_x,
        "support_back_x": support_back_x,
        "last_abs_pitch": jp.abs(torso_gravity[0]),
        "residual_scale": jp.array(0.06),
    }

  def _recovery_metrics(self) -> dict[str, Any]:
    metrics = {
        "reward/alive": jp.zeros(()),
        "reward/upright": jp.zeros(()),
        "reward/recovery_reference": jp.zeros(()),
        "reward/pitch_control": jp.zeros(()),
        "reward/pitch_reduction": jp.zeros(()),
        "reward/com_sagittal": jp.zeros(()),
        "reward/com_lateral": jp.zeros(()),
        "reward/height": jp.zeros(()),
        "reward/zero_velocity": jp.zeros(()),
        "reward/foot_planted": jp.zeros(()),
        "reward/contact_preservation": jp.zeros(()),
        "reward/support_escape_penalty": jp.zeros(()),
        "reward/action_penalty": jp.zeros(()),
        "reward/action_rate_penalty": jp.zeros(()),
        "reward/recovery_reference_v1": jp.zeros(()),
        "recovery/pitch_signal": jp.zeros(()),
        "recovery/abs_pitch": jp.zeros(()),
        "recovery/pitch_delta": jp.zeros(()),
        "recovery/com_x": jp.zeros(()),
        "recovery/com_y": jp.zeros(()),
        "recovery/com_mid_x": jp.zeros(()),
        "recovery/com_mid_y": jp.zeros(()),
        "recovery/com_ahead_front_x": jp.zeros(()),
        "recovery/com_behind_back_x": jp.zeros(()),
        "recovery/foot_motion": jp.zeros(()),
        "recovery/contact_count": jp.zeros(()),
        "recovery/torso_gravity_x": jp.zeros(()),
        "recovery/torso_gravity_y": jp.zeros(()),
        "recovery/torso_gravity_z": jp.zeros(()),
        "authority/residual_scale": jp.zeros(()),
    }
    metrics["swing_peak"] = jp.zeros(())
    return metrics

  def reset(self, rng: jax.Array) -> mjx_env.State:
    # G1 recovery_reference_v1: deterministic recovery reset.
    qpos = self._stand_qpos()
    qvel = jp.zeros(self.mjx_model.nv)
    motor_targets = self._neutral_pose()
    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        qvel=qvel,
        ctrl=motor_targets,
        impl=self.mjx_model.impl.value,
        naconmax=self._config.naconmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)
    info = self._recovery_info(rng, motor_targets, data)
    contact = jp.array([
        data.sensordata[self.mj_model.sensor_adr[sensor_id]] > 0
        for sensor_id in self._feet_floor_found_sensor
    ])
    obs = self._get_obs(data, info, contact)
    return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), self._recovery_metrics(), info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    reference_pose, pitch_signal = self._recovery_reference_pose(state.data)
    residual_scale = state.info["residual_scale"]
    motor_targets = reference_pose + residual_scale * action * self._config.action_scale
    data = mjx_env.step(self.mjx_model, state.data, motor_targets, self.n_substeps)
    contact = jp.array([
        data.sensordata[self.mj_model.sensor_adr[sensor_id]] > 0
        for sensor_id in self._feet_floor_found_sensor
    ])

    info = dict(state.info)
    info["step"] = state.info["step"] + 1
    info["last_last_act"] = state.info["last_act"]
    info["last_act"] = action
    info["motor_targets"] = motor_targets
    info["command"] = jp.zeros(3)
    obs = self._get_obs(data, info, contact)

    torso_gravity = self.get_gravity(data, "torso")
    torso_gravity_z = torso_gravity[-1]
    root_z = data.qpos[2]
    linvel = self.get_global_linvel(data, "pelvis")
    angvel = self.get_global_angvel(data, "pelvis")
    left_foot_x, left_foot_y, right_foot_x, right_foot_y = self._foot_xy(data)
    support_mid_x = 0.5 * (left_foot_x + right_foot_x)
    support_mid_y = 0.5 * (left_foot_y + right_foot_y)
    support_front_x = jp.maximum(left_foot_x, right_foot_x)
    support_back_x = jp.minimum(left_foot_x, right_foot_x)

    com_x = data.subtree_com[0, 0]
    com_y = data.subtree_com[0, 1]
    com_error_x = jp.square(com_x - support_mid_x)
    com_error_y = jp.square(com_y - support_mid_y)
    com_ahead_front_x = com_x - support_front_x
    com_behind_back_x = support_back_x - com_x
    support_escape = jp.maximum(com_ahead_front_x - 0.08, 0.0) + jp.maximum(com_behind_back_x - 0.08, 0.0)
    abs_pitch = jp.abs(torso_gravity[0])
    pitch_delta = state.info["last_abs_pitch"] - abs_pitch
    info["last_abs_pitch"] = abs_pitch

    qpos_error = data.qpos[7:] - reference_pose
    pose_error = jp.mean(jp.square(qpos_error))
    height_error = jp.abs(root_z - self._stand_qpos()[2])
    foot_motion = (
        jp.square(left_foot_x - state.info["reset_left_foot_x"])
        + jp.square(left_foot_y - state.info["reset_left_foot_y"])
        + jp.square(right_foot_x - state.info["reset_right_foot_x"])
        + jp.square(right_foot_y - state.info["reset_right_foot_y"])
    )
    contact_count = jp.sum(contact.astype(float))
    velocity_penalty = linvel[0] ** 2 + linvel[1] ** 2 + 0.15 * linvel[2] ** 2
    angular_velocity_penalty = angvel[0] ** 2 + angvel[1] ** 2 + 0.2 * angvel[2] ** 2
    action_penalty = jp.mean(jp.square(action))
    action_rate_penalty = jp.mean(jp.square(action - state.info["last_act"]))

    fallen = torso_gravity_z < 0.45
    done = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any() | fallen
    done = done.astype(float)
    alive = 1.0 - done

    upright_reward = jp.clip(torso_gravity_z, 0.0, 1.0)
    recovery_reference_reward = jp.exp(-16.0 * pose_error)
    pitch_control_reward = jp.exp(-9.0 * abs_pitch ** 2)
    pitch_reduction_reward = jp.clip(6.0 * jp.maximum(pitch_delta, 0.0), 0.0, 1.0)
    com_sagittal_reward = jp.exp(-35.0 * com_error_x)
    com_lateral_reward = jp.exp(-220.0 * com_error_y)
    height_reward = jp.clip(1.0 - jp.maximum(height_error - 0.035, 0.0) / 0.10, 0.0, 1.0)
    zero_velocity_reward = jp.exp(-4.0 * velocity_penalty)
    foot_planted_reward = jp.exp(-160.0 * foot_motion)
    contact_preservation_reward = contact_count / 2.0

    reward_value = (
        1.2 * alive
        + 3.2 * upright_reward
        + 2.6 * recovery_reference_reward
        + 2.2 * pitch_control_reward
        + 0.8 * pitch_reduction_reward
        + 2.2 * com_sagittal_reward
        + 1.7 * com_lateral_reward
        + 1.3 * height_reward
        + 1.2 * zero_velocity_reward
        + 1.1 * foot_planted_reward
        + 1.5 * contact_preservation_reward
        - 4.5 * support_escape
        - 0.45 * angular_velocity_penalty
        - 0.02 * action_penalty
        - 0.04 * action_rate_penalty
    )

    metrics = dict(state.metrics)
    metrics["reward/alive"] = alive
    metrics["reward/upright"] = upright_reward
    metrics["reward/recovery_reference"] = recovery_reference_reward
    metrics["reward/pitch_control"] = pitch_control_reward
    metrics["reward/pitch_reduction"] = pitch_reduction_reward
    metrics["reward/com_sagittal"] = com_sagittal_reward
    metrics["reward/com_lateral"] = com_lateral_reward
    metrics["reward/height"] = height_reward
    metrics["reward/zero_velocity"] = zero_velocity_reward
    metrics["reward/foot_planted"] = foot_planted_reward
    metrics["reward/contact_preservation"] = contact_preservation_reward
    metrics["reward/support_escape_penalty"] = support_escape
    metrics["reward/action_penalty"] = action_penalty
    metrics["reward/action_rate_penalty"] = action_rate_penalty
    metrics["reward/recovery_reference_v1"] = reward_value
    metrics["recovery/pitch_signal"] = pitch_signal
    metrics["recovery/abs_pitch"] = abs_pitch
    metrics["recovery/pitch_delta"] = pitch_delta
    metrics["recovery/com_x"] = com_x
    metrics["recovery/com_y"] = com_y
    metrics["recovery/com_mid_x"] = support_mid_x
    metrics["recovery/com_mid_y"] = support_mid_y
    metrics["recovery/com_ahead_front_x"] = com_ahead_front_x
    metrics["recovery/com_behind_back_x"] = com_behind_back_x
    metrics["recovery/foot_motion"] = foot_motion
    metrics["recovery/contact_count"] = contact_count
    metrics["recovery/torso_gravity_x"] = torso_gravity[0]
    metrics["recovery/torso_gravity_y"] = torso_gravity[1]
    metrics["recovery/torso_gravity_z"] = torso_gravity_z
    metrics["authority/residual_scale"] = residual_scale
    return mjx_env.State(data, obs, reward_value, done, metrics, info)

'''


def find_g1_file(playground_dir: Path) -> Path:
    direct_candidates = [
        playground_dir / "mujoco_playground" / "_src" / "locomotion" / "g1" / "joystick.py",
        playground_dir / "external" / "mujoco_playground" / "mujoco_playground" / "_src" / "locomotion" / "g1" / "joystick.py",
    ]
    for path in direct_candidates:
        if path.exists():
            text = path.read_text(encoding="utf-8")
            if RESET_START in text and STEP_START in text and "def _get_obs" in text:
                return path

    search_root = playground_dir / "mujoco_playground"
    for path in search_root.rglob("*.py"):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        path_lower = str(path).lower()
        if "locomotion/g1" in path_lower.replace("\\", "/") and RESET_START in text and STEP_START in text:
            return path
    raise FileNotFoundError(f"Could not locate G1 joystick environment under {playground_dir}")


def source_text_for_patch(g1_path: Path) -> str:
    recovery_backup = g1_path.with_suffix(".py.robot_learning_g1_recovery_reference_v1_backup")
    v5_backup = g1_path.with_suffix(".py.robot_learning_g1_reference_gait_v5_backup")
    if recovery_backup.exists():
        return recovery_backup.read_text(encoding="utf-8")
    if v5_backup.exists():
        return v5_backup.read_text(encoding="utf-8")
    return g1_path.read_text(encoding="utf-8")


def apply_patch(g1_path: Path) -> None:
    current_text = g1_path.read_text(encoding="utf-8")
    backup_path = g1_path.with_suffix(".py.robot_learning_g1_recovery_reference_v1_backup")

    if PATCH_MARKER in current_text:
        print(f"G1 recovery_reference_v1 patch already present: {g1_path}")
        return

    source_text = source_text_for_patch(g1_path)
    if not backup_path.exists():
        backup_path.write_text(source_text, encoding="utf-8")

    reset_start = source_text.find(RESET_START)
    step_start = source_text.find(STEP_START, reset_start)
    obs_start = source_text.find(OBS_START, step_start)
    if reset_start == -1 or step_start == -1 or obs_start == -1:
        raise RuntimeError("Could not locate G1 reset/step/_get_obs methods to patch.")

    patched = source_text[:reset_start] + PATCHED_RESET_AND_STEP + source_text[obs_start:]
    g1_path.write_text(patched, encoding="utf-8")
    print(f"Applied G1 recovery_reference_v1 patch: {g1_path}")
    print(f"Backup: {backup_path}")


def restore_patch(g1_path: Path) -> None:
    backup_path = g1_path.with_suffix(".py.robot_learning_g1_recovery_reference_v1_backup")
    if not backup_path.exists():
        raise FileNotFoundError(f"No recovery_reference_v1 backup found for {g1_path}")
    g1_path.write_text(backup_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Restored G1 source from backup: {backup_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--playground-dir", required=True, type=Path)
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()

    g1_path = find_g1_file(args.playground_dir)
    if args.restore:
        restore_patch(g1_path)
    else:
        apply_patch(g1_path)


if __name__ == "__main__":
    main()
