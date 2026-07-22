"""Smoke test for the Project Jiraiya MuJoCo Playground environment."""

from __future__ import annotations

import jax
import mujoco
import mujoco_playground


def main() -> None:
    print(f"jax: {jax.__version__}")
    print(f"jax backend: {jax.default_backend()}")
    print(f"mujoco: {mujoco.__version__}")
    print(f"mujoco_playground: {mujoco_playground.__name__}")
    print("smoke test: success")


if __name__ == "__main__":
    main()
