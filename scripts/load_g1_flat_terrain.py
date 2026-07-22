"""Load the G1 flat-terrain locomotion environment."""

from __future__ import annotations

from mujoco_playground import locomotion


def main() -> None:
    locomotion.load("G1JoystickFlatTerrain", config_overrides={"impl": "jax"})
    print("G1JoystickFlatTerrain load: success")


if __name__ == "__main__":
    main()