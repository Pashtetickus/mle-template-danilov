import configparser
import os
import unittest
import pandas as pd
import sys

sys.path.insert(1, os.path.join(os.getcwd(), "src"))

from train import MultiModel

config = configparser.ConfigParser()
config.read("config.ini")


class TestMultiModel(unittest.TestCase):
    """Preform models tests."""

    def setUp(self) -> None:
        """Inits models class."""
        self.multi_model = MultiModel()

    def test_log_reg(self):
        """Preform logreg test."""
        self.assertEqual(self.multi_model.log_reg(), True)

    def test_rand_forest(self):
        """Preform random forest test."""
        self.assertEqual(self.multi_model.rand_forest(use_config=False), True)

    def test_d_tree(self):
        """Preform decision tree test."""
        self.assertEqual(self.multi_model.d_tree(use_config=False), True)


if __name__ == "__main__":
    unittest.main()
