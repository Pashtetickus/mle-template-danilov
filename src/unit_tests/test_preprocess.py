import configparser
import os
import unittest
import pandas as pd
import sys

sys.path.insert(1, os.path.join(os.getcwd(), "src"))

from preprocess import DataMaker

config = configparser.ConfigParser()
config.read("config.ini")


class TestDataMaker(unittest.TestCase):
    """Performs dataset tests."""

    def setUp(self) -> None:
        """Inits datamaker class."""
        self.data_maker = DataMaker()

    def test_get_data(self):
        """Tests get_data method."""
        self.assertEqual(self.data_maker.get_data(), True)

    def test_split_data(self):
        """Tests split_data method."""
        self.assertEqual(self.data_maker.split_data(), True)

    def test_save_splitted_data(self):
        """Tests save method."""
        self.assertEqual(self.data_maker.save_splitted_data(pd.read_csv(
            config["DATA"]["x_data"], index_col=0), config["DATA"]["x_data"]), True)


if __name__ == "__main__":
    unittest.main()
