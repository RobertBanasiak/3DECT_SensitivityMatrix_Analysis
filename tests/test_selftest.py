import io
import unittest
from contextlib import redirect_stdout

from ect_analysis.selftest import run_self_test


class NumericalSelfTest(unittest.TestCase):
    def test_internal_numerical_self_test(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            run_self_test()
        self.assertIn("SELF-TEST PASSED", output.getvalue())


if __name__ == "__main__":
    unittest.main()
