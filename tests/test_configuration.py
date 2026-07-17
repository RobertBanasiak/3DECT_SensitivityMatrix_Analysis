import unittest

from ect_analysis.cli import validate_args
from main import PYCHARM_CONFIG, pycharm_arguments


class ConfigurationTest(unittest.TestCase):
    def test_pycharm_configuration_is_complete_and_valid(self) -> None:
        args = pycharm_arguments()
        self.assertEqual(set(vars(args)), set(PYCHARM_CONFIG))
        self.assertEqual(
            validate_args(args),
            {"global", "groups", "subsets", "phantoms"},
        )


if __name__ == "__main__":
    unittest.main()
