#!/usr/bin/env python3
"""Diagnose trained G1 standing policies with per-step physics metrics.

This script runs a restored PPO policy in the patched G1 standing environment and
records the signals we care about for failure analysis: COM relative to support,
torso pitch/roll, pelvis velocity, foot motion, contact, action size, and reward.

It is meant to run on the AWS MuJoCo Playground machine from the project root,
while the matching stand_balance patch for the checkpoint is active.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
import mediapy as media
import mujoco
import numpy as np
import orbax.checkpoint as ocp
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from mujoco import mjx
from mujoco_playground import registry
from mujoco_playground._src import mjx_env


DEFAULT_OVERRIDES: dict[str, Any] = {
    "impl": "jax",
    "lin_vel_x": [0.0, 0.0],
    "lin_vel_y": [0.0, 0.0],
    "ang_vel_yaw": [0.0, 0.0],
    "push_config.enable": False,
    "noise_config.level": 0.0,
}

JOINTS_TO_TRACK = (
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)


@dataclass(frozen=True)
class PolicyCase:
    name: str
    patch: str
    checkpoint_path: Path
    description: str = ""


def _to_float(value: Any) -> float:
    return float(np.asarray(value))


def _sensor_scalar(env: Any, data: mjx.Data, sensor_name: str) -> float:
    sensor_id = env.mj_model.sensor(sensor_name).id
    adr = env.mj_model.sensor_adr[sensor_id]
    return _to_float(data.sensordata[adr])


def _sensor_vec(env: Any, data: mjx.Data, sensor_name: str) -> np.ndarray:
    sensor_id = env.mj_model.sensor(sensor_name).id
    adr = env.mj_model.sensor_adr[sensor_id]
    dim = env.mj_model.sensor_dim[sensor_id]
    return np.asarray(data.sensordata[adr : adr + dim], dtype=np.float64)


def _geom_pos(data: mjx.Data, geom_id: int) -> np.ndarray:
    return np.asarray(data.geom_xpos[geom_id], dtype=np.float64)


def _joint_qpos_adr(env: Any, joint_name: str) -> int:
    return int(np.asarray(env.mj_model.joint(joint_name).qposadr).item())


def _joint_row(env: Any, data: mjx.Data, action: jax.Array, joint_name: str) -> dict[str, float]:
    qpos_adr = _joint_qpos_adr(env, joint_name)
    action_idx = qpos_adr - 7
    qvel_idx = qpos_adr - 1
    target = np.asarray(state_safe_default_pose(env))[action_idx] + np.asarray(action)[action_idx] * env._config.action_scale
    return {
        f"{joint_name}/qpos": _to_float(data.qpos[qpos_adr]),
        f"{joint_name}/qvel": _to_float(data.qvel[qvel_idx]),
        f"{joint_name}/target": float(target),
        f"{joint_name}/actuator_force": _to_float(data.actuator_force[action_idx]),
    }


def state_safe_default_pose(env: Any) -> jax.Array:
    # Patched stand-balance environments expose _stand_qpos; fallback keeps this
    # usable with the original joystick task during debugging.
    if hasattr(env, "_stand_qpos"):
        return env._stand_qpos()[7:]
    return jp.array(env._default_pose)


def find_latest_checkpoint(path: Path) -> Path:
    if (path / "ppo_network_config.json").exists():
        return path
    ckpts = [p for p in path.iterdir() if p.is_dir() and p.name.isdigit()]
    if not ckpts:
        raise FileNotFoundError(f"No numeric checkpoints found under {path}")
    return sorted(ckpts, key=lambda p: int(p.name))[-1]


def _activation_from_name(name: str | None):
    if name in (None, "swish", "silu"):
        return jax.nn.swish
    if name == "relu":
        return jax.nn.relu
    if name == "tanh":
        return jp.tanh
    raise ValueError(f"Unsupported activation in checkpoint config: {name}")



def convert_running_statistics_state(params: Any) -> Any:
    """Convert Orbax-restored normalizer dicts into Brax dataclass state."""
    if not isinstance(params, (tuple, list)) or not params:
        return params
    normalizer_state = params[0]
    if not isinstance(normalizer_state, dict) or "mean" not in normalizer_state:
        return params
    converted = running_statistics.RunningStatisticsState(
        count=normalizer_state["count"],
        mean=normalizer_state["mean"],
        summed_variance=normalizer_state["summed_variance"],
        std=normalizer_state["std"],
        std_eps=float(np.asarray(normalizer_state.get("std_eps", 0.0))),
    )
    return (converted, *tuple(params[1:]))

def load_policy(env: Any, checkpoint_path: Path, deterministic: bool):
    checkpoint_path = find_latest_checkpoint(checkpoint_path)
    config_path = checkpoint_path / "ppo_network_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing ppo_network_config.json in {checkpoint_path}")

    with config_path.open("r", encoding="utf-8") as fp:
        config = json.load(fp)
    kwargs = dict(config["network_factory_kwargs"])

    policy_obs_key = kwargs.get("policy_obs_key", "state")
    value_obs_key = kwargs.get("value_obs_key", "privileged_state")
    policy_hidden_layer_sizes = tuple(kwargs.get("policy_hidden_layer_sizes", (512, 256, 128)))
    value_hidden_layer_sizes = tuple(kwargs.get("value_hidden_layer_sizes", (512, 256, 128)))
    activation = _activation_from_name(kwargs.get("activation"))
    distribution_type = kwargs.get("distribution_type", "tanh_normal")
    normalize_observations = bool(config.get("normalize_observations", True))

    if normalize_observations:
        preprocess_observations_fn = running_statistics.normalize
    else:
        preprocess_observations_fn = lambda obs, params: obs

    ppo_network = ppo_networks.make_ppo_networks(
        observation_size=env.observation_size,
        action_size=env.action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        policy_hidden_layer_sizes=policy_hidden_layer_sizes,
        value_hidden_layer_sizes=value_hidden_layer_sizes,
        activation=activation,
        policy_obs_key=policy_obs_key,
        value_obs_key=value_obs_key,
        distribution_type=distribution_type,
        noise_std_type=kwargs.get("noise_std_type", "scalar"),
        init_noise_std=float(kwargs.get("init_noise_std", 1.0)),
        state_dependent_std=bool(kwargs.get("state_dependent_std", False)),
        mean_clip_scale=kwargs.get("mean_clip_scale"),
        use_distributional_critic=bool(kwargs.get("use_distributional_critic", False)),
        num_quantiles=int(kwargs.get("num_quantiles", 32)),
    )

    params = ocp.PyTreeCheckpointer().restore(str(checkpoint_path))
    params = convert_running_statistics_state(params)
    make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
    inference_fn = make_inference_fn(params, deterministic=deterministic)
    return jax.jit(inference_fn), checkpoint_path, config


def diagnostic_row(env: Any, state: mjx_env.State, action: jax.Array, step: int, policy_name: str) -> dict[str, Any]:
    data = state.data
    torso_gravity = env.get_gravity(data, "torso")
    pelvis_gravity = env.get_gravity(data, "pelvis")
    left_foot = _geom_pos(data, env._left_foot_geom_id)
    right_foot = _geom_pos(data, env._right_foot_geom_id)
    support_min_x = float(min(left_foot[0], right_foot[0]))
    support_max_x = float(max(left_foot[0], right_foot[0]))
    support_mid_x = 0.5 * (support_min_x + support_max_x)
    support_mid_y = 0.5 * (float(left_foot[1]) + float(right_foot[1]))
    com_x = _to_float(data.subtree_com[0, 0])
    com_y = _to_float(data.subtree_com[0, 1])
    left_force = _sensor_vec(env, data, "left_foot_force")
    right_force = _sensor_vec(env, data, "right_foot_force")
    action_np = np.asarray(action, dtype=np.float64)

    row: dict[str, Any] = {
        "policy": policy_name,
        "step": step,
        "time": _to_float(data.time),
        "root_x": _to_float(data.qpos[0]),
        "root_y": _to_float(data.qpos[1]),
        "root_z": _to_float(data.qpos[2]),
        "com_x": com_x,
        "com_y": com_y,
        "com_z": _to_float(data.subtree_com[0, 2]),
        "support_min_x": support_min_x,
        "support_max_x": support_max_x,
        "support_mid_x": support_mid_x,
        "support_mid_y": support_mid_y,
        "com_minus_support_mid_x": com_x - support_mid_x,
        "com_minus_support_mid_y": com_y - support_mid_y,
        "com_ahead_of_support_max_x": com_x - support_max_x,
        "left_foot_x": float(left_foot[0]),
        "left_foot_y": float(left_foot[1]),
        "left_foot_z": float(left_foot[2]),
        "right_foot_x": float(right_foot[0]),
        "right_foot_y": float(right_foot[1]),
        "right_foot_z": float(right_foot[2]),
        "left_foot_floor_contact": _sensor_scalar(env, data, "left_foot_floor_found"),
        "right_foot_floor_contact": _sensor_scalar(env, data, "right_foot_floor_found"),
        "left_foot_force_x": float(left_force[0]),
        "left_foot_force_y": float(left_force[1]),
        "left_foot_force_z": float(left_force[2]),
        "right_foot_force_x": float(right_force[0]),
        "right_foot_force_y": float(right_force[1]),
        "right_foot_force_z": float(right_force[2]),
        "torso_gravity_x": _to_float(torso_gravity[0]),
        "torso_gravity_y": _to_float(torso_gravity[1]),
        "torso_gravity_z": _to_float(torso_gravity[2]),
        "pelvis_gravity_x": _to_float(pelvis_gravity[0]),
        "pelvis_gravity_y": _to_float(pelvis_gravity[1]),
        "pelvis_gravity_z": _to_float(pelvis_gravity[2]),
        "global_vx": _to_float(env.get_global_linvel(data, "pelvis")[0]),
        "global_vy": _to_float(env.get_global_linvel(data, "pelvis")[1]),
        "global_vz": _to_float(env.get_global_linvel(data, "pelvis")[2]),
        "global_wx": _to_float(env.get_global_angvel(data, "pelvis")[0]),
        "global_wy": _to_float(env.get_global_angvel(data, "pelvis")[1]),
        "global_wz": _to_float(env.get_global_angvel(data, "pelvis")[2]),
        "reward": _to_float(state.reward),
        "done": bool(_to_float(state.done) > 0.0),
        "action_abs_mean": float(np.abs(action_np).mean()),
        "action_abs_max": float(np.abs(action_np).max()),
        "actuator_force_abs_mean": float(np.abs(np.asarray(data.actuator_force)).mean()),
        "actuator_force_abs_max": float(np.abs(np.asarray(data.actuator_force)).max()),
    }
    for joint_name in JOINTS_TO_TRACK:
        row.update(_joint_row(env, data, action, joint_name))
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first_done_step = next((r["step"] for r in rows if r["done"]), None)
    max_forward = max(rows, key=lambda r: r["com_ahead_of_support_max_x"])
    max_abs_pitch = max(rows, key=lambda r: abs(r["torso_gravity_x"]))
    max_vx = max(rows, key=lambda r: abs(r["global_vx"]))
    return {
        "first_done_step": first_done_step,
        "initial": rows[0],
        "final": rows[-1],
        "max_com_ahead_of_support_max_x": {
            "step": max_forward["step"],
            "value": max_forward["com_ahead_of_support_max_x"],
        },
        "max_abs_torso_gravity_x": {
            "step": max_abs_pitch["step"],
            "value": max_abs_pitch["torso_gravity_x"],
        },
        "max_abs_global_vx": {
            "step": max_vx["step"],
            "value": max_vx["global_vx"],
        },
        "sampled_steps": [rows[i] for i in (0, 25, 50, 75, 100, 150, 200, 249) if i < len(rows)],
    }


def run_case(case: PolicyCase, output_dir: Path, steps: int, seed: int, render_every: int, deterministic: bool, render: bool) -> dict[str, Any]:
    env = registry.load(
        "G1JoystickFlatTerrain",
        config=registry.get_default_config("G1JoystickFlatTerrain"),
        config_overrides=DEFAULT_OVERRIDES,
    )
    inference_fn, resolved_checkpoint, network_config = load_policy(env, case.checkpoint_path, deterministic=deterministic)
    step_fn = jax.jit(env.step)
    rng = jax.random.PRNGKey(seed)
    state = env.reset(rng)
    rows: list[dict[str, Any]] = []
    trajectory: list[mjx_env.State] = []

    for step in range(steps):
        rng, action_rng = jax.random.split(rng)
        action, _ = inference_fn(state.obs, action_rng)
        rows.append(diagnostic_row(env, state, action, step, case.name))
        trajectory.append(state)
        state = step_fn(state, action)

    case_dir = output_dir / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    write_csv(case_dir / "metrics.csv", rows)

    video_path = None
    if render:
        scene_option = mujoco.MjvOption()
        frames = env.render(trajectory[::render_every], height=480, width=640, scene_option=scene_option)
        video_path = str(case_dir / "rollout.mp4")
        media.write_video(video_path, frames, fps=1.0 / env.dt / render_every)

    summary = summarize(rows)
    summary.update({
        "name": case.name,
        "patch": case.patch,
        "description": case.description,
        "checkpoint_path": str(resolved_checkpoint),
        "deterministic": deterministic,
        "steps": steps,
        "render_every": render_every,
        "metrics_csv": str(case_dir / "metrics.csv"),
        "video": video_path,
        "network_config": network_config,
    })
    with (case_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    return summary


def parse_case(raw: str) -> PolicyCase:
    parts = raw.split("|", 3)
    if len(parts) < 3:
        raise argparse.ArgumentTypeError("case must be name|patch|checkpoint_path[|description]")
    description = parts[3] if len(parts) > 3 else ""
    return PolicyCase(name=parts[0], patch=parts[1], checkpoint_path=Path(parts[2]), description=description)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--case", action="append", required=False, type=parse_case)
    parser.add_argument("--policy-name")
    parser.add_argument("--patch-name", default="none")
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--steps", default=250, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--render-every", default=1, type=int)
    parser.add_argument(
        "--overrides-json",
        default=json.dumps(DEFAULT_OVERRIDES),
        help="MuJoCo Playground config overrides as flattened JSON.",
    )
    parser.add_argument("--enable-pushes", action="store_true")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy actions instead of deterministic mean actions.")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    if not args.case:
        if not args.policy_name or not args.checkpoint_path:
            parser.error("provide --case or --policy-name with --checkpoint-path")
        args.case = [
            PolicyCase(
                name=args.policy_name,
                patch=args.patch_name,
                checkpoint_path=args.checkpoint_path,
            )
        ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    DEFAULT_OVERRIDES.clear()
    DEFAULT_OVERRIDES.update(json.loads(args.overrides_json))
    if args.enable_pushes:
        DEFAULT_OVERRIDES.update(
            {
                "push_config.enable": True,
                "push_config.interval_range": [5.0, 10.0],
                "push_config.magnitude_range": [0.1, 2.0],
            }
        )
    summaries = []
    seeds = args.seeds or [args.seed]
    for seed in seeds:
        for case in args.case:
            seeded_case = PolicyCase(
                name=f"{case.name}_seed_{seed}",
                patch=case.patch,
                checkpoint_path=case.checkpoint_path,
                description=case.description,
            )
            summaries.append(run_case(seeded_case, args.output_dir, args.steps, seed, args.render_every, deterministic=not args.stochastic, render=not args.no_render))

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summaries, fp, indent=2)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()




