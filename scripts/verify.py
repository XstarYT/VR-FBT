"""Run release checks from any working directory using the active Python."""

from pathlib import Path
import shutil
import subprocess
import sys


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    node = shutil.which("node")
    if node is None:
        print("Node.js is required for the phone-page checks. Install it and retry.")
        return 1
    commands = [
        [sys.executable, "-m", "compileall", "-q", "Main.py", "Lib", "vrfbt_calib", "tests", "review", "scripts"],
        [sys.executable, "-m", "pip", "check"],
        [sys.executable, "Main.py", "--check"],
        [sys.executable, "-m", "pytest", "-q"],
        [node, "tests/phone_page.test.cjs"],
        [node, "tests/phone_webrtc_page.test.cjs"],
        [sys.executable, "review/bug_reproductions.py"],
        [node, "review/phone_failure_reproductions.cjs"],
    ]
    for command in commands:
        print(f"\nRunning: {' '.join(command)}", flush=True)
        try:
            result = subprocess.run(command, cwd=root, timeout=300)
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f"Verification failed: {exc}")
            return 1
        if result.returncode:
            return result.returncode
    print("\nAll software checks passed. Physical VRChat acceptance remains a separate check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
