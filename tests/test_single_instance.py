import uuid
import unittest

from Lib.SingleInstance import SingleInstance


class SingleInstanceTests(unittest.TestCase):
    def test_second_copy_is_rejected_until_first_closes(self):
        name = f"Local\\VR-FBT-Test-{uuid.uuid4()}"
        first = SingleInstance(name)
        try:
            self.assertTrue(first.acquired)
            second = SingleInstance(name)
            self.assertFalse(second.acquired)
            second.close()
        finally:
            first.close()
        third = SingleInstance(name)
        try:
            self.assertTrue(third.acquired)
        finally:
            third.close()


if __name__ == "__main__":
    unittest.main()
