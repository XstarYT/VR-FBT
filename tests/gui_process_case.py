"""Run native GUI cases in the same fresh-process model as the desktop app.

This avoids cross-test Tcl/native-library state. It does not retry, skip or
convert a failing child test into a successful result.
"""

import os
from pathlib import Path
import subprocess
import sys
import unittest


class FreshProcessGuiTest(unittest.TestCase):
    def run(self, result=None):
        name = self.id()
        if not name.startswith("tests."):
            name = "tests." + name
        if os.environ.get("VRFBT_GUI_TEST_CHILD") == name:
            return super().run(result)
        result = self.defaultTestResult() if result is None else result
        result.startTest(self)
        environment = os.environ.copy()
        environment["VRFBT_GUI_TEST_CHILD"] = name
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "unittest", name, "-v"],
                cwd=Path(__file__).resolve().parents[1], env=environment,
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=45,
            )
            if completed.returncode:
                raise AssertionError(
                    f"GUI child test {name} exited {completed.returncode}:\n"
                    f"{completed.stdout[-12000:]}\n{completed.stderr[-12000:]}"
                )
        except AssertionError:
            result.addFailure(self, sys.exc_info())
        except Exception:
            result.addError(self, sys.exc_info())
        else:
            result.addSuccess(self)
        finally:
            result.stopTest(self)
        return result


class FailingChildProbe(FreshProcessGuiTest):
    """Loaded by exact name only, to verify propagation through the runner."""

    def test_expected_probe_failure(self):
        self.fail("intentional child failure must reach the parent")
