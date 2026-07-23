from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a standalone Yozakura distribution")
    parser.add_argument("--clean", action="store_true", help="Remove previous build outputs first")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    build_dir = root / "build"
    dist_dir = root / "dist"

    if args.clean:
        shutil.rmtree(build_dir, ignore_errors=True)
        shutil.rmtree(dist_dir, ignore_errors=True)

    subprocess.run(
        [sys.executable, "-m", "PyInstaller", "--noconfirm", str(root / "packaging" / "yozakura.spec")],
        cwd=root,
        check=True,
    )

    executable = dist_dir / "yozakura" / ("yozakura.exe" if sys.platform == "win32" else "yozakura")
    if not executable.exists():
        raise SystemExit(f"Standalone executable was not created: {executable}")

    subprocess.run([str(executable), "--help"], cwd=root, check=True)
    print(f"Standalone distribution: {executable.parent}")


if __name__ == "__main__":
    main()
