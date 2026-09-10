"""VR-FBT desktop entry point."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VR full-body tracking control center")
    parser.add_argument(
        "--check",
        action="store_true",
        help="run headless configuration/dependency diagnostics and exit",
    )
    args = parser.parse_args(argv)

    if args.check:
        from Lib.Config import run_diagnostics

        ok = run_diagnostics(
            lambda name, passed, detail: print(
                f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}"
            )
        )
        return 0 if ok else 1

    from Lib.SingleInstance import SingleInstance

    with SingleInstance() as instance:
        if not instance.acquired:
            from tkinter import messagebox
            messagebox.showwarning("VR-FBT is already running", "Close the existing VR-FBT window before starting another copy.")
            return 2
        from Lib.GUI import launch
        launch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
