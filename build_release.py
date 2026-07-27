"""Build a native Formula AI Lab bundle on the current operating system."""
from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    system = platform.system().lower()
    name = {
        "windows": "Formula-AI-Lab-Windows",
        "darwin": "Formula-AI-Lab-macOS",
        "linux": "Formula-AI-Lab-Linux",
    }.get(system, "Formula-AI-Lab")
    command = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean", "--windowed",
        "--name", name,
        "--add-data", f"{ROOT / 'saved_data'}{';' if system == 'windows' else ':'}saved_data",
        str(ROOT / "main.py"),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    print(f"Built {name} in {ROOT / 'dist'}")


if __name__ == "__main__":
    main()

