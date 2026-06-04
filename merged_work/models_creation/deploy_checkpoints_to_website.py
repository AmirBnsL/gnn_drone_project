"""Copy trained checkpoints into Website/Theoritical/resources/checkpoints/."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from merged_work.models_creation.package_setup import repo_root

FILES = (
    "best_gatv2_digits.pth",
    "normalization_stats_digits.pt",
    "strict_local_negotiator_best_v1.pt",
    "bertsekas_best_digits.pt",
)


def deploy(src_ckpt_dir: Path, website_ckpt_dir: Path) -> None:
    src_ckpt_dir = Path(src_ckpt_dir)
    website_ckpt_dir = Path(website_ckpt_dir)
    website_ckpt_dir.mkdir(parents=True, exist_ok=True)

    for name in FILES:
        src = src_ckpt_dir / name
        if name.startswith("best_gat") or name.startswith("normalization"):
            if not src.is_file():
                raise FileNotFoundError(f"Missing setpoint artifact: {src}")
            shutil.copy2(src, website_ckpt_dir / name)
            print(f"Copied {name}")
        else:
            dst = website_ckpt_dir / name
            if src.is_file():
                shutil.copy2(src, dst)
                print(f"Copied {name}")
            elif dst.is_file():
                print(f"Kept existing {name}")
            else:
                print(f"Warning: missing {name} in src and website")


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        type=Path,
        default=root / "merged_artifacts" / "checkpoints",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=root / "Website" / "Theoritical" / "resources" / "checkpoints",
    )
    args = parser.parse_args()
    deploy(args.src, args.dst)


if __name__ == "__main__":
    main()
