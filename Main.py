"""VR-FBT desktop entry point."""

from __future__ import annotations

import argparse
import logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VR full-body tracking control center")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="run headless configuration/dependency diagnostics and exit",
    )
    mode.add_argument("--support-bundle", metavar="ZIP", help="save a local redacted support ZIP and exit")
    args = parser.parse_args(argv)

    if args.support_bundle:
        from Lib.Config import load_profile, load_settings
        from Lib.Support import export_support_bundle
        settings = profile = None
        snapshot_error = None
        try:
            settings = load_settings()
            profile = load_profile(settings.default_profile)
        except Exception as exc:
            snapshot_error = f"Saved settings could not be loaded: {exc}"
        try:
            path = export_support_bundle(args.support_bundle, settings, profile, snapshot_error=snapshot_error)
            print(f"Support ZIP saved locally: {path}")
            return 0
        except Exception as exc:
            print(f"Support export failed: {exc}")
            return 1

    if args.check:
        from Lib.Config import run_diagnostics

        ok = run_diagnostics(
            lambda name, passed, detail: print(
                f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}"
            )
        )
        return 0 if ok else 1

    from Lib.Logging import configure_logging
    from Lib.SingleInstance import SingleInstance

    path = configure_logging()
    try:
        with SingleInstance() as instance:
            if not instance.acquired:
                from tkinter import messagebox
                messagebox.showwarning("VR-FBT is already running", "Close the existing VR-FBT window before starting another copy.")
                return 2
            from Lib.GUI import launch
            launch()
    except Exception as exc:
        logging.getLogger("vrfbt").exception("Application failed")
        detail = f"{type(exc).__name__}: {exc}\n\nRun python Main.py --check to diagnose the installation."
        if path:
            detail += f"\nDetails: {path}"
        try:
            from tkinter import messagebox
            messagebox.showerror("VR-FBT could not continue", detail)
        except Exception:
            print(detail)
        return 1
    return 0


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    raise SystemExit(main())
