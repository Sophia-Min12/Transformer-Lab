"""Smoke test: keeps CI green from the very first push.

(pytest exits non-zero when it collects zero tests — this file guarantees
there is always at least one.)
"""

import sys
import unittest


class TestEnvironment(unittest.TestCase):
    def test_python_version(self):
        self.assertGreaterEqual(sys.version_info, (3, 10))

    def test_numpy_is_available(self):
        import numpy as np

        self.assertEqual(float(np.array([1.0, 2.0]).sum()), 3.0)


if __name__ == "__main__":
    unittest.main()
