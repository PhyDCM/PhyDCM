"""
Example: training a CT model with PhyDCM.

Expected directory structure (DATA_DIR):
- ct/train/<class_name>/*.png|*.jpg|*.bmp|...
- ct/val/<class_name>/*.png|*.jpg|*.bmp|...

Use the environment variable PHYDCM_DATA_DIR to point to your dataset root.
"""
from __future__ import annotations

from phydcm import train_model


def main() -> None:
    train_model("ct")


if __name__ == "__main__":
    main()
