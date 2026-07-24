#!/usr/bin/env python3
"""Evaluate Panda transport with distinct transient and final-state metrics."""

from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path

import jax
import jax.numpy as jp
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import registry, wrapper
from mujoco_playground.config import manipulation_params


ENV_NAME = "PandaPickCube"
LIFT_THRESHOLD = 0.05
HOLD_STEPS = 10
TARGET_DISTANCE = 0.08


def evaluate(
    checkpoint_dir: Path,
    output_dir: Path,
    batches: int,
    batch_size: int,
    seed: int,
) -> dict:
    env_cfg = registry.get_default_config(ENV_NAME)
    ppo_params = manipulation_params.brax_ppo_config(ENV_NAME, "jax")
    ppo_params.num_timesteps = 0
    ppo_params.num_evals = 1
    ppo_params.num_eval_envs = batch_size

    env = registry.load(ENV_NAME, config=env_cfg, config_overrides={"impl": "jax"})
    wrapped_env = wrapper.wrap_for_brax_training(
        env,
        episode_length=ppo_params.episode_length,
        action_repeat=ppo_params.get("action_repeat", 1),
    )

    training_params = dict(ppo_params)
    network_factory_config = training_params.pop("network_factory")
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **network_factory_config
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_fn = functools.partial(
        ppo.train,
        **training_params,
        network_factory=network_factory,
        seed=1,
        restore_checkpoint_path=checkpoint_dir,
        save_checkpoint_path=output_dir / "checkpoints",
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )
    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=lambda *_args, **_kwargs: None,
        policy_params_fn=lambda *_args, **_kwargs: None,
        eval_env=env,
    )
    inference_fn = jax.jit(make_inference_fn(params, deterministic=True))
    obj_body = env._obj_body  # pylint: disable=protected-access

    @jax.jit
    def rollout(rng):
        reset_keys = jax.random.split(rng, batch_size)
        state = wrapped_env.reset(reset_keys)
        initial_height = state.data.xpos[:, obj_body, 2]
        initial_target_position = state.info["target_pos"]
        max_height_delta = jp.zeros((batch_size,))
        hold_run = jp.zeros((batch_size,), dtype=jp.int32)
        max_hold = jp.zeros((batch_size,), dtype=jp.int32)
        min_target_distance = jp.full((batch_size,), jp.inf)
        target_region_seen = jp.zeros((batch_size,), dtype=bool)
        near_target_open_run = jp.zeros((batch_size,), dtype=jp.int32)
        max_near_target_open_run = jp.zeros((batch_size,), dtype=jp.int32)
        post_lift_gripper_open_seen = jp.zeros((batch_size,), dtype=bool)

        def step(carry, _step_index):
            (
                state,
                rng,
                max_height_delta,
                hold_run,
                max_hold,
                min_target_distance,
                target_region_seen,
                near_target_open_run,
                max_near_target_open_run,
                post_lift_gripper_open_seen,
            ) = carry
            rng, action_rng = jax.random.split(rng)
            action_keys = jax.random.split(action_rng, batch_size)
            actions = jax.vmap(inference_fn)(state.obs, action_keys)[0]
            state = wrapped_env.step(state, actions)
            height_delta = state.data.xpos[:, obj_body, 2] - initial_height
            lifted = height_delta >= LIFT_THRESHOLD
            max_height_delta = jp.maximum(max_height_delta, height_delta)
            hold_run = jp.where(lifted, hold_run + 1, 0)
            max_hold = jp.maximum(max_hold, hold_run)
            target_distance = jp.linalg.norm(
                state.info["target_pos"] - state.data.xpos[:, obj_body, :], axis=-1
            )
            gripper_open = jp.mean(state.data.qpos[:, -2:], axis=-1) / 0.04 >= 0.8
            min_target_distance = jp.minimum(min_target_distance, target_distance)
            target_region_seen = target_region_seen | (
                lifted
                & (hold_run >= HOLD_STEPS)
                & (target_distance <= TARGET_DISTANCE)
            )
            post_lift_gripper_open_seen = post_lift_gripper_open_seen | (
                lifted & (hold_run >= HOLD_STEPS) & gripper_open
            )
            near_target_open_run = jp.where(
                gripper_open & (target_distance <= TARGET_DISTANCE),
                near_target_open_run + 1,
                0,
            )
            max_near_target_open_run = jp.maximum(
                max_near_target_open_run, near_target_open_run
            )
            return (
                state,
                rng,
                max_height_delta,
                hold_run,
                max_hold,
                min_target_distance,
                target_region_seen,
                near_target_open_run,
                max_near_target_open_run,
                post_lift_gripper_open_seen,
            ), None

        carry = (
            state,
            rng,
            max_height_delta,
            hold_run,
            max_hold,
            min_target_distance,
            target_region_seen,
            near_target_open_run,
            max_near_target_open_run,
            post_lift_gripper_open_seen,
        )
        carry, _ = jax.lax.scan(step, carry, jp.arange(ppo_params.episode_length))
        (
            state,
            _rng,
            max_height_delta,
            _hold_run,
            max_hold,
            min_target_distance,
            target_region_seen,
            _near_target_open_run,
            max_near_target_open_run,
            post_lift_gripper_open_seen,
        ) = carry
        final_position = state.data.xpos[:, obj_body, :]
        final_distance = jp.linalg.norm(state.info["target_pos"] - final_position, axis=-1)
        lift_success = max_height_delta >= LIFT_THRESHOLD
        hold_success = max_hold >= HOLD_STEPS
        final_target_success = (\n            lift_success\n            & hold_success\n            & post_lift_gripper_open_seen\n            & (final_distance <= TARGET_DISTANCE)\n        )
        transient_near_target_open_success = max_near_target_open_run >= HOLD_STEPS
        return {
            "max_height_delta": max_height_delta,
            "max_hold_steps": max_hold,
            "initial_target_position": initial_target_position,
            "final_target_position": state.info["target_pos"],
            "min_target_distance": min_target_distance,
            "final_target_distance": final_distance,
            "lift_success": lift_success,
            "hold_success": hold_success,
            "final_target_success": final_target_success,
            "target_region_seen": target_region_seen,
            "post_lift_gripper_open_seen": post_lift_gripper_open_seen,
            "max_near_target_open_steps": max_near_target_open_run,
            "transient_near_target_open_success": transient_near_target_open_success,
        }

    batches_out = []
    for batch_index in range(batches):
        metrics = rollout(jax.random.PRNGKey(seed + batch_index))
        batches_out.append({key: value.tolist() for key, value in metrics.items()})

    flat = {key: [item for batch in batches_out for item in batch[key]] for key in batches_out[0]}
    result = {
        "environment": ENV_NAME,
        "checkpoint": str(checkpoint_dir),
        "episodes": batches * batch_size,
        "seed": seed,
        "thresholds": {
            "lift_height_delta_m": LIFT_THRESHOLD,
            "hold_steps": HOLD_STEPS,
            "target_distance_m": TARGET_DISTANCE,
        },
        "lift_success_rate": sum(flat["lift_success"]) / len(flat["lift_success"]),
        "hold_success_rate": sum(flat["hold_success"]) / len(flat["hold_success"]),
        "final_target_success_rate": sum(flat["final_target_success"]) / len(flat["final_target_success"]),
        "mean_max_height_delta_m": sum(flat["max_height_delta"]) / len(flat["max_height_delta"]),
        "mean_max_hold_steps": sum(flat["max_hold_steps"]) / len(flat["max_hold_steps"]),
        "mean_final_target_distance_m": sum(flat["final_target_distance"]) / len(flat["final_target_distance"]),
        "target_region_seen_rate": sum(flat["target_region_seen"]) / len(flat["target_region_seen"]),
        "post_lift_gripper_open_seen_rate": sum(flat["post_lift_gripper_open_seen"]) / len(flat["post_lift_gripper_open_seen"]),
        "transient_near_target_open_rate": sum(flat["transient_near_target_open_success"]) / len(flat["transient_near_target_open_success"]),
        "mean_max_near_target_open_steps": sum(flat["max_near_target_open_steps"]) / len(flat["max_near_target_open_steps"]),
        "mean_min_target_distance_m": sum(flat["min_target_distance"]) / len(flat["min_target_distance"]),
        "episodes_detail": flat,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != "episodes_detail"}, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=10_000)
    args = parser.parse_args()
    evaluate(args.checkpoint_dir, args.output_dir, args.batches, args.batch_size, args.seed)


if __name__ == "__main__":
    main()
