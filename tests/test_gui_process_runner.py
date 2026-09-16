import unittest

from tests import gui_process_case


class GuiProcessRunnerTests(unittest.TestCase):
    def test_child_failure_is_reported_as_a_real_parent_failure(self):
        result = unittest.TestResult()
        gui_process_case.FailingChildProbe("test_expected_probe_failure").run(result)
        self.assertEqual(result.testsRun, 1)
        self.assertEqual(len(result.failures), 1)
        self.assertIn("intentional child failure must reach the parent", result.failures[0][1])
        self.assertEqual(result.skipped, [])
